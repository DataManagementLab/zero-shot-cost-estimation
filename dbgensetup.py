import argparse
import multiprocessing
import multiprocessing as mp
import os
import time

import pandas as pd
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.generate_column_stats import generate_stats
from cross_db_benchmark.benchmark_tools.generate_string_statistics import generate_string_stats
from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload
from cross_db_benchmark.benchmark_tools.load_database import load_database
from cross_db_benchmark.benchmark_tools.parse_run import parse_run
from cross_db_benchmark.datasets.datasets import database_list, ext_database_list
from cross_db_benchmark.meta_tools.download_relational_fit import download_from_relational_fit
from cross_db_benchmark.meta_tools.scale_dataset import scale_up_dataset

from run_benchmark import StoreDictKeyPair
from setup.postgres.run_workload_commands import gen_run_workload_commands

workload_defs = {
    # standard workloads will be capped at 5k
    'workload_100k_s1': dict(num_queries=100000,
                             max_no_predicates=5,
                             max_no_aggregates=3,
                             max_no_group_by=0,
                             max_cols_per_agg=2,
                             seed=1),
    # for complex predicates, this will be capped at 5k
    'complex_workload_200k_s1': dict(num_queries=200000,
                                     max_no_predicates=5,
                                     max_no_aggregates=3,
                                     max_no_group_by=0,
                                     max_cols_per_agg=2,
                                     complex_predicates=True,
                                     seed=1),
    # for index workloads, will also be capped at 5k
    'workload_100k_s2': dict(num_queries=100000,
                             max_no_predicates=5,
                             max_no_aggregates=3,
                             max_no_group_by=0,
                             max_cols_per_agg=2,
                             seed=2),
}


def compute(input):
    source, target, d, wl, parse_baseline, cap_queries = input
    no_plans, stats = parse_run(source, target, args.database, min_query_ms=args.min_query_ms, cap_queries=cap_queries,
                                parse_baseline=parse_baseline, parse_join_conds=True)
    return dict(dataset=d, workload=wl, no_plans=no_plans, **stats)


def workload_gen(input):
    source_dataset, workload_path, max_no_joins, workload_args = input
    generate_workload(source_dataset, workload_path, max_no_joins=max_no_joins, **workload_args)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--workload_dir', default='../zero-shot-data/workloads')
    parser.add_argument("--database_conn", dest='database_conn_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--database_kwargs", dest='database_kwarg_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument('--hardware', default='c8220')

    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--parsed_plan_dir', default=None)
    parser.add_argument('--target_stats_path', default=None)
    parser.add_argument('--workloads', nargs='+', default=None)
    parser.add_argument('--min_query_ms', default=100, type=int)
    parser.add_argument('--cap_queries', default=5000, type=int)
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))

    parser.add_argument('--generate_column_statistics', action='store_true')
    parser.add_argument('--generate_string_statistics', action='store_true')
    parser.add_argument('--download_relational_fit', action='store_true')
    parser.add_argument('--scale_datasets', action='store_true')
    parser.add_argument('--load_database', action='store_true')
    parser.add_argument('--generate_workloads', action='store_true')
    parser.add_argument('--print_run_commands', action='store_true')
    parser.add_argument('--parse_all_queries', action='store_true')
    parser.add_argument('--print_zero_shot_commands', action='store_true')

    args = parser.parse_args()

    if args.database_kwarg_dict is None:
        args.database_kwarg_dict = dict()

    if args.generate_column_statistics:
        generate_stats(args.data_dir, args.dataset)

    if args.generate_string_statistics:
        generate_string_stats(args.data_dir, args.dataset)

    if args.download_relational_fit:

        print("Downloading datasets from relational.fit...")
        for rel_fit_dataset_name, dataset_name in tqdm([('Walmart', 'walmart'),
                                                        ('Basketball_men', 'basketball'),
                                                        ('financial', 'financial'),
                                                        ('geneea', 'geneea'),
                                                        ('Accidents', 'accidents'),
                                                        ('imdb_MovieLens', 'movielens'),
                                                        ('lahman_2014', 'baseball'),
                                                        ('Hepatitis_std', 'hepatitis'),
                                                        ('NCAA', 'tournament'),
                                                        ('VisualGenome', 'genome'),
                                                        ('Credit', 'credit'),
                                                        ('employee', 'employee'),
                                                        ('Carcinogenesis', 'carcinogenesis'),
                                                        ('ConsumerExpenditures', 'consumer'),
                                                        ('Seznam', 'seznam'),
                                                        ('FNHK', 'fhnk')]):
            download_from_relational_fit(rel_fit_dataset_name, dataset_name, root_data_dir=args.data_dir)

    if args.scale_datasets:
        # scale if required
        for dataset in database_list:
            if dataset.scale == 1:
                continue

            assert dataset.data_folder != dataset.source_dataset, "For scaling a new folder is required"
            print(f"Scaling dataset {dataset.db_name}")
            curr_source_dir = os.path.join(args.data_dir, dataset.source_dataset)
            curr_target_dir = os.path.join(args.data_dir, dataset.data_folder)
            if not os.path.exists(curr_target_dir):
                scale_up_dataset(dataset.source_dataset, curr_source_dir, curr_target_dir, scale=dataset.scale)

    if args.load_database:
        # load databases
        # also load imdb full dataset to be able to run the full job benchmark
        for dataset in ext_database_list:
            for database in [DatabaseSystem.POSTGRES]:
                curr_data_dir = os.path.join(args.data_dir, dataset.data_folder)
                print(f"Loading database {dataset.db_name} from {curr_data_dir}")
                load_database(curr_data_dir, dataset.source_dataset, database, dataset.db_name, args.database_conn_dict,
                              args.database_kwarg_dict)

    if args.generate_workloads:
        workload_gen_setups = []
        for dataset in ext_database_list:
            for workload_name, workload_args in workload_defs.items():
                workload_path = os.path.join(args.workload_dir, dataset.db_name, f'{workload_name}.sql')
                workload_gen_setups.append((dataset.source_dataset, workload_path, dataset.max_no_joins, workload_args))

        start_t = time.perf_counter()
        proc = multiprocessing.cpu_count() - 2
        p = mp.Pool(initargs=('arg',), processes=proc)
        p.map(workload_gen, workload_gen_setups)
        print(f"Generated workloads in {time.perf_counter() - start_t:.2f} secs")

    if args.print_run_commands:
        experiment_commands = []
        experiment_commands += gen_run_workload_commands(workload_name='workload_100k_s1', cap_workload=5000)
        # 5k queries with indexes
        experiment_commands += gen_run_workload_commands(workload_name='workload_100k_s2', cap_workload=5000,
                                                         with_indexes=True)
        # 5k queries with complex predicates
        experiment_commands += gen_run_workload_commands(workload_name='complex_workload_200k_s1', cap_workload=5000)

        for cmd in experiment_commands:
            print(cmd.replace('[hw_placeholder]', args.hardware))

    if args.parse_all_queries:
        cap_queries = args.cap_queries
        if cap_queries == 'None':
            cap_queries = None

        setups = []
        for wl in args.workloads:
            for db in ext_database_list:
                d = db.db_name
                source = os.path.join(args.raw_dir, d, wl)
                parse_baseline = not 'complex' in wl
                if not os.path.exists(source):
                    print(f"Missing: {d}: {wl}")
                    continue
                target = os.path.join(args.parsed_plan_dir, d, wl)
                setups.append((source, target, d, wl, parse_baseline, cap_queries))

        start_t = time.perf_counter()
        proc = multiprocessing.cpu_count() - 2
        p = mp.Pool(initargs=('arg',), processes=proc)
        eval = p.map(compute, setups)

        eval = pd.DataFrame(eval)
        print()
        print(eval[['dataset', 'workload', 'no_plans']].to_string(index=False))

        print()
        print(eval[['workload', 'no_plans']].groupby('workload').sum().to_string())

        print()
        print(f"Total plans parsed in {time.perf_counter() - start_t:.2f} secs: {eval.no_plans.sum()}")
