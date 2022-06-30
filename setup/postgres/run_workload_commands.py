from cross_db_benchmark.datasets.datasets import ext_database_list
import re


def strip_single_command(cmd):
    cmd = cmd.replace('\n', ' ')
    regex = re.compile(r"\s+", re.IGNORECASE)
    cmd = regex.sub(" ", cmd)
    return cmd


def strip_commands(exp_commands):
    exp_commands = [strip_single_command(cmd) for cmd in exp_commands]
    return exp_commands


def gen_run_workload_commands(workload_name=None, database_conn='user=postgres,password=postgres,host=localhost',
                              database='postgres', cap_workload=10000, query_timeout=30, with_indexes=False,
                              datasets=None, hints=None, db_list=ext_database_list):
    assert workload_name is not None

    cap_workload_cmd = ''
    if cap_workload is not None:
        cap_workload_cmd = f'--cap_workload {cap_workload}'

    index_prefix = ''
    index_cmd = ''
    if with_indexes:
        index_prefix = 'index_'
        index_cmd = '--with_indexes'

    hint_cmd = ''
    if hints is not None:
        hint_cmd = f'--hints {hints}'

    exp_commands = []
    for dataset in db_list:
        if datasets is not None and dataset.db_name not in datasets:
            continue
        exp_commands.append(f"""python3 run_benchmark.py 
          --run_workload
          --query_timeout {query_timeout}
          --source ../zero-shot-data/workloads/{dataset.db_name}/{workload_name}.sql
          --target ../zero-shot-data/runs/raw/{dataset.db_name}/{index_prefix}{workload_name}_[hw_placeholder].json
          --database {database}
          --db_name {dataset.db_name}
          --database_conn {database_conn}
          {cap_workload_cmd}
          {index_cmd}
          {hint_cmd}
          --run_kwargs hardware=[hw_placeholder]
          """)

    exp_commands = strip_commands(exp_commands)
    return exp_commands
