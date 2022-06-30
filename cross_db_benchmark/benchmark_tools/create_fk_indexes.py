from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.load_database import create_db_conn
from cross_db_benchmark.benchmark_tools.utils import load_schema_json


def create_fk_indexes(dataset, database, db_name, database_conn_args, database_kwarg_dict):
    # check if tables are a connected acyclic graph
    schema = load_schema_json(dataset)
    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)

    idx_sql = []
    for r_id, r in enumerate(schema.relationships):
        table_left, col_left, table_right, col_right = r
        cname = col_left
        if isinstance(col_left, tuple) or isinstance(col_left, list):
            cname = "_".join(col_left)
            col_left = ", ".join([f'"{c}"' for c in col_left])

        sql = f"create index {cname}_{table_left} on \"{table_left}\"({col_left});"
        idx_sql.append(sql)

    idx_sql.append("Vacuum Analyze;")
    for sql in tqdm(idx_sql):
        try:
            db_conn.submit_query(sql, db_created=True)
            print(sql)
        except:
            print(f"Skipping {sql}")
            continue
