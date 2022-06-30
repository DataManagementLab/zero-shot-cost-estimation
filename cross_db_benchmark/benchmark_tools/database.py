from enum import Enum


class DatabaseSystem(Enum):
    POSTGRES = 'postgres'

    def __str__(self):
        return self.value


class DatabaseConnection:
    def __init__(self, db_name=None, database_kwargs=None):
        self.db_name = db_name
        self.database_kwargs = database_kwargs

    def drop(self):
        raise NotImplementedError

    def load_database(self, data_dir, dataset, force=False):
        raise NotImplementedError

    def replicate_tuples(self, dataset, data_dir, no_prev_replications):
        raise NotImplementedError

    def set_statement_timeout(self, timeout_sec):
        raise NotImplementedError

    def run_query_collect_statistics(self, sql, repetitions=3):
        raise NotImplementedError

    def collect_db_statistics(self):
        raise NotImplementedError

    def transform_dicts(self, column_stats_names, column_stats_rows):
        return [{k: v for k, v in zip(column_stats_names, row)} for row in column_stats_rows]
