import os
import time
from copy import copy

import numpy as np
import optuna
import torch
import torch.optim as opt
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.utils import load_json
from models.dataset.dataset_creation import create_dataloader
from models.training.checkpoint import save_checkpoint, load_checkpoint, save_csv
from models.training.metrics import MAPE, RMSE, QError
from models.training.utils import batch_to, flatten_dict, find_early_stopping_metric
from models.zero_shot_models.specific_models.model import zero_shot_models


def train_epoch(epoch_stats, train_loader, model, optimizer, max_epoch_tuples, custom_batch_to=batch_to):
    model.train()

    # run remaining batches
    train_start_t = time.perf_counter()
    losses = []
    errs = []
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if max_epoch_tuples is not None and batch_idx * train_loader.batch_size > max_epoch_tuples:
            break

        input_model, label, sample_idxs = custom_batch_to(batch, model.device, model.label_norm)

        optimizer.zero_grad()
        output = model(input_model)
        loss = model.loss_fxn(output, label)
        if torch.isnan(loss):
            raise ValueError('Loss was NaN')
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        output = output.detach().cpu().numpy().reshape(-1)
        label = label.detach().cpu().numpy().reshape(-1)
        errs = np.concatenate((errs, output - label))
        losses.append(loss)

    mean_loss = np.mean(losses)
    mean_rmse = np.sqrt(np.mean(np.square(errs)))
    # print(f"Train Loss: {mean_loss:.2f}")
    # print(f"Train RMSE: {mean_rmse:.2f}")
    epoch_stats.update(train_time=time.perf_counter() - train_start_t, mean_loss=mean_loss, mean_rmse=mean_rmse)


def validate_model(val_loader, model, epoch=0, epoch_stats=None, metrics=None, max_epoch_tuples=None,
                   custom_batch_to=batch_to, verbose=False, log_all_queries=False):
    model.eval()

    with torch.autograd.no_grad():
        val_loss = torch.Tensor([0])
        labels = []
        preds = []
        probs = []
        sample_idxs = []

        # evaluate test set using model
        test_start_t = time.perf_counter()
        val_num_tuples = 0
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            if max_epoch_tuples is not None and batch_idx * val_loader.batch_size > max_epoch_tuples:
                break

            val_num_tuples += val_loader.batch_size

            input_model, label, sample_idxs_batch = custom_batch_to(batch, model.device, model.label_norm)
            sample_idxs += sample_idxs_batch
            output = model(input_model)

            # sum up mean batch losses
            val_loss += model.loss_fxn(output, label).cpu()

            # inverse transform the predictions and labels
            curr_pred = output.cpu().numpy()
            curr_label = label.cpu().numpy()
            if model.label_norm is not None:
                curr_pred = model.label_norm.inverse_transform(curr_pred)
                curr_label = model.label_norm.inverse_transform(curr_label.reshape(-1, 1))
                curr_label = curr_label.reshape(-1)

            preds.append(curr_pred.reshape(-1))
            labels.append(curr_label.reshape(-1))

        if epoch_stats is not None:
            epoch_stats.update(val_time=time.perf_counter() - test_start_t)
            epoch_stats.update(val_num_tuples=val_num_tuples)
            val_loss = (val_loss.cpu() / len(val_loader)).item()
            print(f'val_loss epoch {epoch}: {val_loss}')
            epoch_stats.update(val_loss=val_loss)

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        if verbose:
            print(f'labels: {labels}')
            print(f'preds: {preds}')
        epoch_stats.update(val_std=np.std(labels))
        if log_all_queries:
            epoch_stats.update(val_labels=[float(f) for f in labels])
            epoch_stats.update(val_preds=[float(f) for f in preds])
            epoch_stats.update(val_sample_idxs=sample_idxs)

        # save best model for every metric
        any_best_metric = False
        if metrics is not None:
            for metric in metrics:
                best_seen = metric.evaluate(metrics_dict=epoch_stats, model=model, labels=labels, preds=preds,
                                            probs=probs)
                if best_seen and metric.early_stopping_metric:
                    any_best_metric = True
                    print(f"New best model for {metric.metric_name}")

    return any_best_metric


def optuna_intermediate_value(metrics):
    for m in metrics:
        if m.early_stopping_metric:
            assert isinstance(m, QError)
            return m.best_seen_value
    raise ValueError('Metric invalid')


def train_model(workload_runs,
                test_workload_runs,
                statistics_file,
                target_dir,
                filename_model,
                optimizer_class_name='Adam',
                optimizer_kwargs=None,
                final_mlp_kwargs=None,
                node_type_kwargs=None,
                model_kwargs=None,
                tree_layer_name='GATConv',
                tree_layer_kwargs=None,
                hidden_dim=32,
                batch_size=32,
                output_dim=1,
                epochs=0,
                device='cpu',
                plan_featurization_name=None,
                max_epoch_tuples=100000,
                param_dict=None,
                num_workers=1,
                early_stopping_patience=20,
                trial=None,
                database=None,
                limit_queries=None,
                limit_queries_affected_wl=None,
                skip_train=False,
                seed=0):
    if model_kwargs is None:
        model_kwargs = dict()

    # seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_test_csv_paths = []
    if test_workload_runs is not None:
        for p in test_workload_runs:
            test_workload = os.path.basename(p).replace('.json', '')
            target_test_csv_paths.append(os.path.join(target_dir, f'test_{filename_model}_{test_workload}.csv'))

    if len(target_test_csv_paths) > 0 and all([os.path.exists(p) for p in target_test_csv_paths]):
        print(f"Model was already trained and tested ({target_test_csv_paths} exists)")
        return

    # create a dataset
    loss_class_name = final_mlp_kwargs['loss_class_name']
    label_norm, feature_statistics, train_loader, val_loader, test_loaders = \
        create_dataloader(workload_runs, test_workload_runs, statistics_file, plan_featurization_name, database,
                          val_ratio=0.15, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=False, limit_queries=limit_queries,
                          limit_queries_affected_wl=limit_queries_affected_wl, loss_class_name=loss_class_name)

    if loss_class_name == 'QLoss':
        metrics = [RMSE(), MAPE(), QError(percentile=50, early_stopping_metric=True), QError(percentile=95),
                   QError(percentile=100)]
    elif loss_class_name == 'MSELoss':
        metrics = [RMSE(early_stopping_metric=True), MAPE(), QError(percentile=50), QError(percentile=95),
                   QError(percentile=100)]

    # create zero shot model dependent on database
    model = zero_shot_models[database](device=device, hidden_dim=hidden_dim, final_mlp_kwargs=final_mlp_kwargs,
                                       node_type_kwargs=node_type_kwargs, output_dim=output_dim,
                                       feature_statistics=feature_statistics, tree_layer_name=tree_layer_name,
                                       tree_layer_kwargs=tree_layer_kwargs,
                                       plan_featurization_name=plan_featurization_name,
                                       label_norm=label_norm,
                                       **model_kwargs)
    # move to gpu
    model = model.to(model.device)
    print(model)
    optimizer = opt.__dict__[optimizer_class_name](model.parameters(), **optimizer_kwargs)

    csv_stats, epochs_wo_improvement, epoch, model, optimizer, metrics, finished = \
        load_checkpoint(model, target_dir, filename_model, optimizer=optimizer, metrics=metrics, filetype='.pt')

    # train an actual model (q-error? or which other loss?)
    while epoch < epochs and not finished and not skip_train:
        print(f"Epoch {epoch}")

        epoch_stats = copy(param_dict)
        epoch_stats.update(epoch=epoch)
        epoch_start_time = time.perf_counter()
        # try:
        train_epoch(epoch_stats, train_loader, model, optimizer, max_epoch_tuples)

        any_best_metric = validate_model(val_loader, model, epoch=epoch, epoch_stats=epoch_stats, metrics=metrics,
                                         max_epoch_tuples=max_epoch_tuples)
        epoch_stats.update(epoch=epoch, epoch_time=time.perf_counter() - epoch_start_time)

        # report to optuna
        if trial is not None:
            intermediate_value = optuna_intermediate_value(metrics)
            epoch_stats['optuna_intermediate_value'] = intermediate_value

            print(f"Reporting epoch_no={epoch}, intermediate_value={intermediate_value} to optuna "
                  f"(Trial {trial.number})")
            trial.report(intermediate_value, epoch)

        # see if we can already stop the training
        stop_early = False
        if not any_best_metric:
            epochs_wo_improvement += 1
            if early_stopping_patience is not None and epochs_wo_improvement > early_stopping_patience:
                stop_early = True
        else:
            epochs_wo_improvement = 0
        if trial is not None and trial.should_prune():
            stop_early = True
        # also set finished to true if this is the last epoch
        if epoch == epochs - 1:
            stop_early = True

        epoch_stats.update(stop_early=stop_early)
        print(f"epochs_wo_improvement: {epochs_wo_improvement}")

        # save stats to file
        csv_stats.append(epoch_stats)

        # save current state of training allowing us to resume if this is stopped
        save_checkpoint(epochs_wo_improvement, epoch, model, optimizer, target_dir,
                        filename_model, metrics=metrics, csv_stats=csv_stats, finished=stop_early)

        epoch += 1

        # Handle pruning based on the intermediate value.
        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned()

        if stop_early:
            print(f"Early stopping kicked in due to no improvement in {early_stopping_patience} epochs")
            break
        # except:
        #     print("Error during epoch. Trying again.")

    # if we are not doing hyperparameter search, evaluate test set
    if trial is None and test_loaders is not None:
        if not (target_dir is None or filename_model is None):
            assert len(target_test_csv_paths) == len(test_loaders)
            for test_path, test_loader in zip(target_test_csv_paths, test_loaders):
                print(f"Starting validation for {test_path}")
                test_stats = copy(param_dict)

                early_stop_m = find_early_stopping_metric(metrics)
                print("Reloading best model")
                model.load_state_dict(early_stop_m.best_model)
                validate_model(test_loader, model, epoch=epoch, epoch_stats=test_stats, metrics=metrics,
                               log_all_queries=True)

                save_csv([test_stats], test_path)

        else:
            print("Skipping saving the test stats")

    if trial is not None:
        return optuna_intermediate_value(metrics)


def train_default(workload_runs,
                  test_workload_runs,
                  statistics_file,
                  target_dir, filename_model,
                  device='cpu',
                  plan_featurization='PostgresTrueCardDetail',
                  loss_class_name='QLoss',
                  max_epoch_tuples=100000,
                  num_workers=1,
                  database=None,
                  seed=0,
                  limit_queries=None,
                  limit_queries_affected_wl=None,
                  max_no_epochs=None,
                  skip_train=False):
    """
    Sets default parameters and trains model
    """

    p_dropout = 0.1
    # general fc out
    fc_out_kwargs = dict(p_dropout=p_dropout, activation_class_name='LeakyReLU', activation_class_kwargs={},
                         norm_class_name='Identity', norm_class_kwargs={}, residual=False, dropout=True,
                         activation=True, inplace=True)
    final_mlp_kwargs = dict(width_factor=1, n_layers=2,
                            loss_class_name=loss_class_name,  # MSELoss
                            loss_class_kwargs=dict())
    tree_layer_kwargs = dict(width_factor=1, n_layers=2)
    node_type_kwargs = dict(width_factor=1, n_layers=2, one_hot_embeddings=True, max_emb_dim=32,
                            drop_whole_embeddings=False)
    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)

    train_kwargs = dict(optimizer_class_name='AdamW',
                        optimizer_kwargs=dict(
                            lr=1e-3,
                        ),
                        final_mlp_kwargs=final_mlp_kwargs,
                        node_type_kwargs=node_type_kwargs,
                        tree_layer_kwargs=tree_layer_kwargs,
                        tree_layer_name='MscnConv',  # GATConv MscnConv
                        plan_featurization_name=plan_featurization,
                        hidden_dim=128,
                        output_dim=1,
                        epochs=200 if max_no_epochs is None else max_no_epochs,
                        early_stopping_patience=20,
                        batch_size=128,
                        max_epoch_tuples=max_epoch_tuples,
                        device=device,
                        num_workers=num_workers,
                        seed=seed,
                        limit_queries=limit_queries,
                        limit_queries_affected_wl=limit_queries_affected_wl,
                        skip_train=skip_train
                        )
    param_dict = flatten_dict(train_kwargs)

    train_model(workload_runs, test_workload_runs, statistics_file, target_dir, filename_model, param_dict=param_dict,
                database=database, **train_kwargs)


def train_readout_hyperparams(workload_runs,
                              test_workload_runs,
                              statistics_file,
                              target_dir, filename_model,
                              hyperparameter_path,
                              device='cpu',
                              max_epoch_tuples=100000,
                              num_workers=1,
                              loss_class_name='QLoss',
                              database=None,
                              seed=0,
                              limit_queries=None,
                              limit_queries_affected_wl=None,
                              max_no_epochs=None,
                              skip_train=False
                              ):
    """
    Reads out hyperparameters and trains model
    """
    print(f"Reading hyperparameters from {hyperparameter_path}")
    hyperparams = load_json(hyperparameter_path, namespace=False)

    p_dropout = hyperparams.pop('p_dropout')
    # general fc out
    fc_out_kwargs = dict(p_dropout=p_dropout,
                         activation_class_name='LeakyReLU',
                         activation_class_kwargs={},
                         norm_class_name='Identity',
                         norm_class_kwargs={},
                         residual=hyperparams.pop('residual'),
                         dropout=hyperparams.pop('dropout'),
                         activation=True,
                         inplace=True)
    final_mlp_kwargs = dict(width_factor=hyperparams.pop('final_width_factor'),
                            n_layers=hyperparams.pop('final_layers'),
                            loss_class_name=loss_class_name,
                            loss_class_kwargs=dict())
    tree_layer_kwargs = dict(width_factor=hyperparams.pop('tree_layer_width_factor'),
                             n_layers=hyperparams.pop('message_passing_layers'))
    node_type_kwargs = dict(width_factor=hyperparams.pop('node_type_width_factor'),
                            n_layers=hyperparams.pop('node_layers'),
                            one_hot_embeddings=True,
                            max_emb_dim=hyperparams.pop('max_emb_dim'),
                            drop_whole_embeddings=False)
    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)

    train_kwargs = dict(optimizer_class_name='AdamW',
                        optimizer_kwargs=dict(
                            lr=hyperparams.pop('lr'),
                        ),
                        final_mlp_kwargs=final_mlp_kwargs,
                        node_type_kwargs=node_type_kwargs,
                        tree_layer_kwargs=tree_layer_kwargs,
                        tree_layer_name=hyperparams.pop('tree_layer_name'),
                        plan_featurization_name=hyperparams.pop('plan_featurization_name'),
                        hidden_dim=hyperparams.pop('hidden_dim'),
                        output_dim=1,
                        epochs=200 if max_no_epochs is None else max_no_epochs,
                        early_stopping_patience=20,
                        max_epoch_tuples=max_epoch_tuples,
                        batch_size=hyperparams.pop('batch_size'),
                        device=device,
                        num_workers=num_workers,
                        seed=seed,
                        limit_queries=limit_queries,
                        limit_queries_affected_wl=limit_queries_affected_wl,
                        skip_train=skip_train
                        )

    assert len(hyperparams) == 0, f"Not all hyperparams were used (not used: {hyperparams.keys()}). Hence generation " \
                                  f"and reading does not seem to fit"

    param_dict = flatten_dict(train_kwargs)
    train_model(workload_runs, test_workload_runs, statistics_file, target_dir, filename_model,
                param_dict=param_dict, database=database, **train_kwargs)
