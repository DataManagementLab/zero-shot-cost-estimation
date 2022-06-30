import argparse
import glob

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from models.preprocessing.feature_statistics import gather_feature_statistics
from models.training.train import train_default, train_readout_hyperparams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workload_runs', default=None, nargs='+')
    parser.add_argument('--test_workload_runs', default=None, nargs='+')
    parser.add_argument('--statistics_file', default=None)
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--loss_class_name', default='QLoss')
    parser.add_argument('--filename_model', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_epoch_tuples', type=int, default=100000)
    parser.add_argument('--max_no_epochs', type=int, default=None)
    parser.add_argument('--limit_queries', type=int, default=None)
    parser.add_argument('--limit_queries_affected_wl', type=int, default=None)
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    parser.add_argument('--gather_feature_statistics', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--plan_featurization', default='PostgresTrueCardDetail')
    parser.add_argument('--hyperparameter_path', default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    if args.gather_feature_statistics:
        # gather_feature_statistics
        workload_runs = []

        for wl in args.workload_runs:
            workload_runs += glob.glob(f'{args.raw_dir}/*/{wl}')

        gather_feature_statistics(workload_runs, args.target)

    if args.train_model:
        if args.hyperparameter_path is None:
            # for testing
            train_default(args.workload_runs, args.test_workload_runs, args.statistics_file, args.target,
                          args.filename_model, plan_featurization=args.plan_featurization, device=args.device,
                          num_workers=args.num_workers, max_epoch_tuples=args.max_epoch_tuples,
                          seed=args.seed, database=args.database, limit_queries=args.limit_queries,
                          limit_queries_affected_wl=args.limit_queries_affected_wl, max_no_epochs=args.max_no_epochs,
                          skip_train=args.skip_train, loss_class_name=args.loss_class_name)
        else:
            train_readout_hyperparams(args.workload_runs, args.test_workload_runs, args.statistics_file, args.target,
                                      args.filename_model, args.hyperparameter_path, device=args.device,
                                      num_workers=args.num_workers, max_epoch_tuples=args.max_epoch_tuples,
                                      seed=args.seed, database=args.database, limit_queries=args.limit_queries,
                                      limit_queries_affected_wl=args.limit_queries_affected_wl,
                                      max_no_epochs=args.max_no_epochs, skip_train=args.skip_train,
                                      loss_class_name=args.loss_class_name)
