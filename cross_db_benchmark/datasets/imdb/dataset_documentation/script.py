import os
import re


def extract_column_names(table_defs):
    column_names = dict()

    single_table_defs = table_defs.split("CREATE TABLE")
    for single_table in single_table_defs:
        alphanumeric_sequences = re.findall('\w+', single_table)
        if len(alphanumeric_sequences) > 0:
            table_name = alphanumeric_sequences[0]
            cols = [col.strip() for col in re.findall('\n\s+\w+', single_table)]
            if 'DROP' in cols:
                cols.remove('DROP')
            column_names[table_name] = cols

    return column_names


imdb_no_header_path = '../../../../../zero-shot-data/datasets/imdb_no_header'
imdb_path = '../../../../../zero-shot-data/datasets/imdb'
sql_ddl_path = '../schema_sql/postgres.sql'
assert os.path.exists(sql_ddl_path)
assert os.path.exists(imdb_no_header_path)

with open(sql_ddl_path, 'r') as file:
    table_defs = file.read()
    # This is a rather improvised function. It does not properly parse the sql but instead assumes that columns
    # start with a newline followed by whitespaces and table definitions start with CREATE TABLE ...
    column_names = extract_column_names(table_defs)

print(column_names)

for table in ["kind_type", "title", "cast_info", "company_name", "company_type", "info_type", "keyword",
              "movie_companies", "movie_info_idx", "movie_keyword", "movie_info", "person_info", "char_name",
              "aka_name", "name"]:
    print(f"Creating headers for {table}")
    with open(os.path.join(imdb_path, f'{table}.csv'), 'w') as outfile:
        with open(os.path.join(imdb_no_header_path, f'{table}.csv')) as infile:
            outfile.write(','.join(column_names[table]) + '\n')
            for line in infile:
                outfile.write(line)
