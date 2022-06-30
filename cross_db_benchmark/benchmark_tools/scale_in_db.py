from cross_db_benchmark.benchmark_tools.load_database import create_db_conn


def scale_in_db(data_dir, dataset, database, db_name, database_conn_args, database_kwarg_dict, no_prev_replications):
    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)
    db_conn.replicate_tuples(dataset, data_dir, no_prev_replications)


