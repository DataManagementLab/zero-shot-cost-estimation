from cross_db_benchmark.datasets.datasets import database_list
from octopus.script_preparation import strip_commands


def train_spns(ensemble_strategy, rdc_threshold, timeout=4*3600):
    exp_commands = []
    for db in database_list:
        train_cmd = f"""python3.8 deepdb.py 
            --generate_ensemble 
            --dataset {db.db_name} 
            --samples_per_spn 10000000 10000000 1000000 1000000 1000000 
            --ensemble_strategy {ensemble_strategy} 
            --csv_path ../zero-shot-data/datasets/{db.db_name} 
            --hdf_path ../zero-shot-data/deepdb/datasets/{db.db_name}/gen_single_light 
            --ensemble_path ../zero-shot-data/deepdb/datasets/{db.db_name}/spn_ensembles 
            --max_rows_per_hdf_file 100000000 
            --rdc_threshold {rdc_threshold}
            --post_sampling_factor 10 10 5 1 1
            [device_placeholder]
            """
        exp_commands.append(f'timeout {timeout} {train_cmd}')

    exp_commands = strip_commands(exp_commands)
    return exp_commands