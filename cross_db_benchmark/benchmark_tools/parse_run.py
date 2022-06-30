import json
import os

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.combine_plans import combine_traces
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_plans
from cross_db_benchmark.benchmark_tools.utils import load_json


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


def parse_run(source_paths, target_path, database, min_query_ms=100, max_query_ms=30000,
              parse_baseline=False, cap_queries=None, parse_join_conds=False, include_zero_card=False,
              explain_only=False):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if database == DatabaseSystem.POSTGRES:
        parse_func = parse_plans
        comb_func = combine_traces
    else:
        raise NotImplementedError(f"Database {database} not yet supported.")

    if not isinstance(source_paths, list):
        source_paths = [source_paths]

    assert all([os.path.exists(p) for p in source_paths])
    run_stats = [load_json(p) for p in source_paths]
    run_stats = comb_func(run_stats)

    parsed_runs, stats = parse_func(run_stats, min_runtime=min_query_ms, max_runtime=max_query_ms,
                                    parse_baseline=parse_baseline, cap_queries=cap_queries,
                                    parse_join_conds=parse_join_conds,
                                    include_zero_card=include_zero_card, explain_only=explain_only)

    with open(target_path, 'w') as outfile:
        json.dump(parsed_runs, outfile, default=dumper)
    return len(parsed_runs['parsed_plans']), stats
