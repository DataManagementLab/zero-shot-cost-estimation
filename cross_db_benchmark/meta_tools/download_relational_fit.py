import os
import shutil

from cross_db_benchmark.benchmark_tools.utils import load_schema_json


def download_from_relational_fit(rel_fit_dataset_name, dataset_name, root_data_dir='../zero-shot-data/datasets'):
    schema = load_schema_json(dataset_name)
    data_dir = os.path.join(root_data_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    for table in schema.tables:
        retries = 0
        target_table_file = os.path.join(data_dir, f'{table}.csv')

        if not os.path.exists(target_table_file):
            # `{table}`
            while not download_successful(table) and retries < 3:
                retries += 1
                download_cmd = f'echo "select * from \`{table}\`;" | mysql --host=relational.fit.cvut.cz --user=guest --password=relational {rel_fit_dataset_name} > {table}.csv'
                print(download_cmd)
                os.system(download_cmd)

                download_successful(table)

            if not download_successful(table):
                raise ValueError(f"Could not download table {table}")

            shutil.move(f'{table}.csv', target_table_file)
        else:
            print(f"Skipping download for {table}")


def download_successful(table):
    if not os.path.exists(f"{table}.csv"):
        return False

    filesize = os.path.getsize(f"{table}.csv")
    return filesize > 0
