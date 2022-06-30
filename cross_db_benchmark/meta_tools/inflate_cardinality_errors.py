import json
import os

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.parse_run import dumper
from cross_db_benchmark.benchmark_tools.postgres.inflate_cardinality_errors import inflate_card_errors_pg
from cross_db_benchmark.benchmark_tools.utils import load_json


def inflate_cardinality_errors(source_path, target_path, card_error_factor, database):
    assert os.path.exists(source_path)
    run_stats = load_json(source_path)

    if database == DatabaseSystem.POSTGRES:
        inflate_func = inflate_card_errors_pg
    else:
        raise NotImplementedError

    for p in run_stats.parsed_plans:
        inflate_func(p, card_error_factor)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w') as outfile:
        json.dump(run_stats, outfile, default=dumper)
