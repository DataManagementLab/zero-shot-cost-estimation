import collections
import hashlib
import json
import os

import dgl
import torch
from dgl import DGLHeteroGraph


def recursive_to(iterable, device):
    if isinstance(iterable, (dgl.DGLGraph, DGLHeteroGraph)):
        iterable.to(device)
    if isinstance(iterable, torch.Tensor):
        iterable.data = iterable.data.to(device)
    elif isinstance(iterable, collections.abc.Mapping):
        for v in iterable.values():
            recursive_to(v, device)
    elif isinstance(iterable, (list, tuple)):
        for v in iterable:
            recursive_to(v, device)


def batch_to(batch, device, label_norm):
    graph, features, label, sample_idxs = batch

    # normalize the labels for training
    if label_norm is not None:
        label = label_norm.transform(label.reshape(-1,1))
        label = label.reshape(-1)
    label = torch.from_numpy(label)

    recursive_to(features, device)
    recursive_to(label, device)
    # recursive_to(graph, device)
    graph = graph.to(device)
    return (graph, features), label, sample_idxs


def flatten_dict(d, parent_key='', sep='_'):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def find_early_stopping_metric(metrics):
    potential_metrics = [m for m in metrics if m.early_stopping_metric]
    assert len(potential_metrics) == 1
    early_stopping_metric = potential_metrics[0]
    return early_stopping_metric

def save_config(save_dict, target_dir, json_name):
    os.makedirs(target_dir, exist_ok=True)
    target_params_path = os.path.join(target_dir, json_name)
    print(f"Saving best params to {target_params_path}")

    with open(target_params_path, 'w') as f:
        json.dump(save_dict, f)
