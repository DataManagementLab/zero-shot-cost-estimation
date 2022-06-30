import json
import os
import random
import re
import shutil
import time
from json.decoder import JSONDecodeError

from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.load_database import create_db_conn
from cross_db_benchmark.benchmark_tools.postgres.check_valid import check_valid
from cross_db_benchmark.benchmark_tools.utils import load_json

column_regex = re.compile('"(\S+)"."(\S+)"')


def extract_columns(sql):
    return [m for m in column_regex.findall(sql)]


def index_creation_deletion(existing_indexes, sql_part, db_conn, timeout_sec):
    cols = extract_columns(sql_part)
    index_cols = set(existing_indexes.keys())
    no_idxs = len(index_cols.intersection(cols))

    if len(cols) > 0:
        # not a single index for sql available, create one
        if no_idxs == 0:
            t, c = random.choice(cols)
            db_conn.set_statement_timeout(10 * timeout_sec, verbose=False)
            print(f"Creating index on {t}.{c}")
            index_creation_start = time.perf_counter()
            try:
                index_name = db_conn.create_index(t, c)
                existing_indexes[(t, c)] = index_name
                print(f"Creation time: {time.perf_counter() - index_creation_start:.2f}s")
            except Exception as e:
                print(f"Index creation failed {str(e)}")
            db_conn.set_statement_timeout(timeout_sec, verbose=False)

        # indexes for all columns, delete one
        if len(cols) > 1 and no_idxs == len(cols):
            t, c = random.choice(cols)
            print(f"Dropping index on {t}.{c}")
            try:
                index_name = existing_indexes[(t, c)]
                db_conn.drop_index(index_name)
                del existing_indexes[(t, c)]
            except Exception as e:
                print(f"Index deletion failed {str(e)}")


def modify_indexes(db_conn, sql_query, existing_indexes, timeout_sec):
    try:
        if 'GROUP BY ' in sql_query:
            sql_query = sql_query.split('GROUP BY ')[0]
        join_part = sql_query.split(" FROM ")[1].split(" WHERE ")[0]
        where_part = sql_query.split(" FROM ")[1].split(" WHERE ")[1]

        index_creation_deletion(existing_indexes, join_part, db_conn, timeout_sec)
        index_creation_deletion(existing_indexes, where_part, db_conn, timeout_sec)
    except Exception as e:
        print(f"Could not create indexes for {sql_query} ({str(e)})")


def run_pg_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                    repetitions_per_query, timeout_sec, with_indexes=False, cap_workload=None, hints=None,
                    min_runtime=100, no_variants=26):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)

    with open(workload_path) as f:
        content = f.readlines()
    sql_queries = [x.strip() for x in content]

    hint_list = ['' for _ in sql_queries]
    if hints is not None:
        if hints == 'random':
            hint_list = [gen_optimizer_hint_variant(random.randrange(no_variants)) for _ in sql_queries]

        elif hints == 'all':
            hint_list = [gen_optimizer_hint_variant(i) for i in range(no_variants) for _ in sql_queries]
            sql_queries = [sql_q for i in range(no_variants) for sql_q in sql_queries]

        else:
            with open(hints) as f:
                content = f.readlines()
            hint_list = [x.strip() for x in content]
    assert len(hint_list) == len(sql_queries)

    # extract column statistics
    database_stats = db_conn.collect_db_statistics()

    # remove existing indexes from previous workload runs
    if with_indexes:
        db_conn.remove_remaining_fk_indexes()

    # check if workload already exists
    query_list = []
    seen_queries = set()
    time_offset = 0
    if os.path.exists(target_path):
        try:
            last_run = load_json(target_path, namespace=False)
            query_list = last_run['query_list']
            if 'total_time_secs' in last_run:
                time_offset = last_run['total_time_secs']
            for q in query_list:
                seen_queries.add(q['sql'])

            if cap_workload is not None:
                print("Checking existing files")
                for q in tqdm(query_list):
                    if check_valid(q, min_runtime=min_runtime, verbose=False):
                        cap_workload -= 1
                        if cap_workload == 0:
                            print(f"Read existing files already reached sufficient number of queries")
                            return
                print(f"Read existing files and reduced workload cap to {cap_workload}")
        except JSONDecodeError:
            print("Could not read json")

    # set a timeout to make sure long running queries do not delay the entire process
    db_conn.set_statement_timeout(timeout_sec)

    existing_indexes = dict()

    # extract query plans
    start_t = time.perf_counter()
    valid_queries = 0
    for i, sql_query in enumerate(tqdm(sql_queries)):
        if sql_query in seen_queries:
            continue

        if with_indexes:
            modify_indexes(db_conn, sql_query, existing_indexes, timeout_sec)

        hint = hint_list[i]
        curr_statistics = db_conn.run_query_collect_statistics(sql_query, repetitions=repetitions_per_query,
                                                               prefix=hint)
        curr_statistics.update(sql=sql_query)
        curr_statistics.update(hint=hint)
        query_list.append(curr_statistics)

        run_stats = dict(query_list=query_list,
                         database_stats=database_stats,
                         run_kwargs=run_kwargs,
                         total_time_secs=time_offset + (time.perf_counter() - start_t))

        # save to json
        # write to temporary path and then move
        if len(query_list) % 50 == 0:
            save_workload(run_stats, target_path)

        # check whether sufficient valid queries are available
        if cap_workload is not None:
            if check_valid(curr_statistics, min_runtime=min_runtime):
                valid_queries += 1
            elapsed_sec = time.perf_counter() - start_t
            remaining_h = 0
            if valid_queries > 0:
                remaining_h = (cap_workload - valid_queries) / valid_queries * elapsed_sec / 3600

            print(f"Valid Queries {valid_queries}/{cap_workload} "
                  f"(est. remaining hrs: {remaining_h:.3f}, elapsed secs: {elapsed_sec:.2f})")
            if valid_queries >= cap_workload:
                return

    # Finally remove all indexes again
    if with_indexes:
        db_conn.remove_remaining_fk_indexes()

    run_stats = dict(query_list=query_list,
                     database_stats=database_stats,
                     run_kwargs=run_kwargs,
                     total_time_secs=time_offset + (time.perf_counter() - start_t))
    save_workload(run_stats, target_path)

    print(f"Executed workload {workload_path} in {time_offset + time.perf_counter() - start_t:.2f} secs")


def save_workload(run_stats, target_path):
    target_temp_path = os.path.join(os.path.dirname(target_path), f'{os.path.basename(target_path)}_temp')
    with open(target_temp_path, 'w') as outfile:
        json.dump(run_stats, outfile)
    shutil.move(target_temp_path, target_path)
