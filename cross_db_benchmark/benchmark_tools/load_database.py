from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.database_connection import PostgresDatabaseConnection


def create_db_conn(database, db_name, database_conn_args, database_kwarg_dict):
    if database == DatabaseSystem.POSTGRES:
        return PostgresDatabaseConnection(db_name=db_name, database_kwargs=database_conn_args, **database_kwarg_dict)
    else:
        raise NotImplementedError(f"Database {database} not yet supported.")


def load_database(data_dir, dataset, database, db_name, database_conn_args, database_kwarg_dict, force=False):
    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)
    db_conn.load_database(dataset, data_dir, force=force)
