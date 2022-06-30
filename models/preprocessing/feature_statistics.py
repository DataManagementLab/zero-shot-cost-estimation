import collections
import json
import os
from enum import Enum

import numpy as np
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


def gather_values_recursively(json_dict, value_dict=None):
    if value_dict is None:
        value_dict = collections.defaultdict(list)

    if isinstance(json_dict, dict):
        for k, v in json_dict.items():
            if not (isinstance(v, list) or isinstance(v, tuple) or isinstance(v, dict)):
                value_dict[k].append(v)
            elif (isinstance(v, list) or isinstance(v, tuple)) and len(v) > 0 and \
                    (isinstance(v[0], int) or isinstance(v[0], float) or isinstance(v[0], str)):
                for v_e in v:
                    value_dict[k].append(v_e)
            else:
                gather_values_recursively(v, value_dict=value_dict)
    elif isinstance(json_dict, tuple) or isinstance(json_dict, list):
        for e in json_dict:
            gather_values_recursively(e, value_dict=value_dict)

    return value_dict


class FeatureType(Enum):
    numeric = 'numeric'
    categorical = 'categorical'

    def __str__(self):
        return self.value


def gather_feature_statistics(workload_run_paths, target):
    """
    Traverses a JSON object and gathers metadata for each key. Depending on whether the values of the key are
    categorical or numerical, different statistics are collected. This is later on used to automate the feature
    extraction during the training (e.g., how to consistently map a categorical value to an index).
    """

    run_stats = []
    for source in tqdm(workload_run_paths):
        assert os.path.exists(source), f"{source} does not exist"
        try:
            with open(source) as json_file:
                run_stats.append(json.load(json_file))
        except:
            raise ValueError(f"Could not read {source}")
    value_dict = gather_values_recursively(run_stats)

    print("Saving")
    # save unique values for categorical features and scale and center of RobustScaler for numerical ones
    statistics_dict = dict()
    for k, values in value_dict.items():
        values = [v for v in values if v is not None]
        if len(values) == 0:
            continue

        if all([isinstance(v, int) or isinstance(v, float) or v is None for v in values]):
            scaler = RobustScaler()
            np_values = np.array(values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(np_values)

            statistics_dict[k] = dict(max=float(np_values.max()),
                                      scale=scaler.scale_.item(),
                                      center=scaler.center_.item(),
                                      type=str(FeatureType.numeric))
        else:
            unique_values = set(values)
            statistics_dict[k] = dict(value_dict={v: id for id, v in enumerate(unique_values)},
                                      no_vals=len(unique_values),
                                      type=str(FeatureType.categorical))

    # save as json
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w') as outfile:
        json.dump(statistics_dict, outfile)
