import os
import re


def extract_column_names(table_defs):
    column_names = dict()

    single_table_defs = table_defs.split("create table")
    for single_table in single_table_defs:
        alphanumeric_sequences = re.findall('\w+', single_table)
        if len(alphanumeric_sequences) > 0:
            table_name = alphanumeric_sequences[0]
            cols = [col.strip() for col in re.findall('\n\s+\w+', single_table)]
            if 'drop' in cols:
                cols.remove('drop')
            column_names[table_name] = cols

    return column_names


source_path = '../../../../../tpch-dbgen'
target = '../../../../../zero-shot-data/datasets/tpc_h'
os.makedirs(target, exist_ok=True)
sql_ddl_path = '../schema_sql/postgres.sql'
assert os.path.exists(sql_ddl_path)
assert os.path.exists(source_path)

with open(sql_ddl_path, 'r') as file:
    table_defs = file.read()
    # This is a rather improvised function. It does not properly parse the sql but instead assumes that columns
    # start with a newline followed by whitespaces and table definitions start with CREATE TABLE ...
    column_names = extract_column_names(table_defs)

print(column_names)

for table in ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]:
    print(f"Creating headers for {table}")
    with open(os.path.join(target, f'{table}.csv'), 'w') as outfile:
        with open(os.path.join(source_path, f'{table}.csv')) as infile:
            outfile.write('|'.join(column_names[table]) + '\n')
            for line in infile:
                outfile.write(line)
