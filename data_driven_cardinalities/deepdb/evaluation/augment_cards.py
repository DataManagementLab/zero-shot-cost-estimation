import collections
import itertools
import json
import logging
import os
from json import JSONDecodeError
from time import perf_counter

import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.parse_run import dumper
from cross_db_benchmark.benchmark_tools.utils import load_json
from data_driven_cardinalities.deepdb.ensemble_compilation.graph_representation import Query, QueryType
from data_driven_cardinalities.deepdb.ensemble_compilation.spn_ensemble import read_ensemble
from models.training.checkpoint import save_csv

logger = logging.getLogger(__name__)


def augment_cardinalities(schema, ensemble_location, src, target, scale=1, **spn_kwargs):
    # load ensemble
    spn_ensemble = None
    if ensemble_location is not None:
        spn_ensemble = read_ensemble(ensemble_location, build_reverse_dict=True)

    try:
        run = load_json(src, namespace=True)
    except JSONDecodeError:
        raise ValueError(f"Error reading {src}")

    print(src)
    q_stats = []

    # find out if this an non_inclusive workload (< previously replaced by <=)
    non_inclusive = False
    if any([b in src for b in ['job-light', 'scale', 'synthetic']]):
        non_inclusive = True
        print("Assuming NON-INCLUSIVE workload")

    est_pg = 0
    est_deepdb = 0
    for q_id, p in enumerate(tqdm(run.parsed_plans)):

        if spn_ensemble is not None:
            p.plan_parameters.est_pg = 0
            p.plan_parameters.est_deepdb = 0

            augment_bottom_up(schema, p, q_id, run.database_stats, spn_ensemble, q_stats, p, scale,
                              non_inclusive=non_inclusive, **spn_kwargs)
            est_pg += p.plan_parameters.est_pg
            est_deepdb += p.plan_parameters.est_deepdb

            if q_id > 1 and q_id % 1000 == 0:
                report_stats(est_deepdb, est_pg, q_stats)

        if spn_ensemble is not None:
            def augment_prod(p):
                if len(p.children) == 0:
                    p.plan_parameters.dd_est_children_card = 1
                else:
                    child_card = 1
                    for c in p.children:
                        child_card *= c.plan_parameters.dd_est_card
                        augment_prod(c)
                    p.plan_parameters.dd_est_children_card = child_card

            augment_prod(p)
        else:
            def augment_prod(p):
                _, pg_est_card = get_act_est_card(p.plan_parameters)
                p.plan_parameters.dd_est_card = pg_est_card
                if len(p.children) == 0:
                    p.plan_parameters.dd_est_children_card = 1
                else:
                    child_card = 1
                    for c in p.children:
                        augment_prod(c)
                        child_card *= c.plan_parameters.dd_est_card
                    p.plan_parameters.dd_est_children_card = child_card

            augment_prod(p)

    report_stats(est_deepdb, est_pg, q_stats)

    # save augmented plans
    target_dir = os.path.dirname(target)
    os.makedirs(target_dir, exist_ok=True)
    with open(target, 'w') as outfile:
        json.dump(run, outfile, default=dumper)

    # save csv
    filename = f'stats_{os.path.basename(target).replace(".json", "")}.csv'
    if len(q_stats) > 0:
        save_csv(q_stats, os.path.join(target_dir, filename))


def report_stats(est_deepdb, est_pg, q_stats):
    if len(q_stats) > 0:
        def report_percentiles(key):
            vals = np.array([q_s[key] for q_s in q_stats])
            print(f"{key}: p50={np.median(vals):.2f} p95={np.percentile(vals, 95):.2f} "
                  f"p99={np.percentile(vals, 99):.2f} pmax={np.max(vals):.2f}")

        report_percentiles('q_errors_pg')
        report_percentiles('q_errors_deepdb')
        report_percentiles('latencies')
        print(f"{est_deepdb / (est_deepdb + est_pg) * 100:.2f}% estimated using DeepDB")


def augment_bottom_up(schema, plan, q_id, database_statistics, spn_ensemble, q_stats, top_p, scale, non_inclusive=False,
                      **spn_kwargs):
    workers_planned = vars(plan.plan_parameters).get('workers_planned')
    if workers_planned is None:
        workers_planned = 0
    # assert workers_planned is not None

    aggregation_below = 'Aggregate' in plan.plan_parameters.op_name

    # augment own tables
    tables = set()
    t_idx = vars(plan.plan_parameters).get('table')
    if t_idx is not None:
        table_stats = database_statistics.table_stats[t_idx]
        if hasattr(table_stats, 'relname'):
            table_name = table_stats.relname
        elif hasattr(table_stats, 'table'):
            table_name = table_stats.table
        else:
            raise NotImplementedError
        tables.add(table_name)

    # fill own filter conditions
    filter_conditions = set()
    filter_columns = vars(plan.plan_parameters).get('filter_columns')
    if filter_columns is not None:
        def augment_filter_recursively(fc):
            if len(fc.children) == 0 and fc.operator != 'AND':
                c_idx = fc.column
                col_stas = database_statistics.column_stats[c_idx]
                if hasattr(col_stas, 'tablename'):
                    table_name = col_stas.tablename
                    column_name = col_stas.attname
                elif hasattr(col_stas, 'table_name'):
                    table_name = col_stas.table_name
                    column_name = col_stas.column_name
                else:
                    raise NotImplementedError

                filter = (table_name, column_name, fc.operator, fc.literal)
                filter_conditions.add(filter)
            else:
                assert fc.operator == 'AND', "DeepDB does not support disjunctions"
                for c_fc in fc.children:
                    augment_filter_recursively(c_fc)

        augment_filter_recursively(filter_columns)

    for c in plan.children:
        c_aggregation_below, c_tables, c_filter_conditions = augment_bottom_up(schema, c, q_id, database_statistics,
                                                                               spn_ensemble, q_stats,
                                                                               top_p, scale,
                                                                               non_inclusive=non_inclusive,
                                                                               **spn_kwargs)
        aggregation_below |= c_aggregation_below
        tables.update(c_tables)
        filter_conditions.update(c_filter_conditions)

    # evaluate query
    act_card, pg_est_card = get_act_est_card(plan.plan_parameters)

    query_parsed = True
    q = None
    try:
        q = convert_query(schema, tables, filter_conditions, non_inclusive)
    except:
        # print("Could not parse query")
        query_parsed = False

    # query not supported
    if not query_parsed:
        plan.plan_parameters.dd_est_card = pg_est_card
        top_p.plan_parameters.est_pg += 1

    # group by not directly supported
    elif aggregation_below:
        plan.plan_parameters.dd_est_card = pg_est_card
        top_p.plan_parameters.est_pg += 1

    # we do not care about really small cardinalities
    elif (act_card is not None and pg_est_card <= 1000 and act_card <= 1000):
        plan.plan_parameters.dd_est_card = pg_est_card
        top_p.plan_parameters.est_pg += 1

    else:
        card_start_t = perf_counter()

        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000

        if plan.plan_parameters.op_name in {'Parallel Seq Scan', 'Hash Join', 'Nested Loop', 'Seq Scan', 'Materialize',
                                            'Hash', 'Parallel Hash', 'Merge Join', 'Gather', 'Gather Merge',
                                            'Hash Right Join', 'Hash Left Join', 'Nested Loop Left Join',
                                            'Merge Left Join', 'Merge Right Join'} \
                or plan.plan_parameters.op_name.startswith('XN ') \
                or plan.plan_parameters.op_name in {'Broadcast', 'Distribute'}:

            try:
                _, factors, cardinality_predict, factor_values = spn_ensemble \
                    .cardinality(q, return_factor_values=True, join_keys=schema.scaled_join_keys, scale=scale,
                                 **spn_kwargs)
                cardinality_predict *= scale

                # scale by workers for parallel operators
                op_name = plan.plan_parameters.op_name
                if workers_planned > 0 and (op_name.startswith('Parallel')):
                    cardinality_predict /= (workers_planned + 1)

                # if q_err(cardinality_predict, act_card) > 1000:
                #     print(f"\nPG: {pg_est_card:.2f} ({q_err(pg_est_card, act_card):.2f})")
                #     print(f"DeepDB: {cardinality_predict:.2f} ({q_err(cardinality_predict, act_card):.2f})")
                #     print(f"Act: {act_card:.2f}")
                #     print(q.conditions)
                #     print(plan.plan_parameters.op_name)

                if act_card is not None:
                    q_err_deepdb = q_err(cardinality_predict, act_card)
                    q_err_pg = q_err(pg_est_card, act_card)
                else:
                    q_err_deepdb = 1
                    q_err_pg = 1

                # this was probably a bug, anyway rarely happens
                if q_err_deepdb > 100 * q_err_pg:
                    plan.plan_parameters.dd_est_card = pg_est_card
                    top_p.plan_parameters.est_pg += 1
                else:
                    plan.plan_parameters.dd_est_card = cardinality_predict
                    top_p.plan_parameters.est_deepdb += 1

                    q_stats.append({
                        'query_id': q_id,
                        'q_errors_pg': q_err_pg,
                        'q_errors_deepdb': q_err_deepdb,
                        'latencies': latency_ms
                    })

            except Exception as e:
                plan.plan_parameters.dd_est_card = pg_est_card
                top_p.plan_parameters.est_pg += 1
                print(f"Warning: could not predict ({e})")

        # ignore this in the stats since pg semantics for cardinalities are different for this operator
        elif plan.plan_parameters.op_name in {'Index Only Scan', 'Index Scan', 'Parallel Index Only Scan',
                                              'Bitmap Index Scan', 'Parallel Bitmap Heap Scan', 'Bitmap Heap Scan',
                                              'Sort', 'Parallel Index Scan', 'BitmapAnd'}:
            plan.plan_parameters.dd_est_card = pg_est_card
            top_p.plan_parameters.est_pg += 1
        else:
            raise NotImplementedError(plan.plan_parameters.op_name)

    return aggregation_below, tables, filter_conditions


def get_act_est_card(params):
    if hasattr(params, 'act_card'):
        act_card = params.act_card
        pg_est_card = params.est_card
    elif hasattr(params, 'est_rows'):
        act_card = params.act_avg_rows
        pg_est_card = params.est_rows
    # only estimated available
    elif hasattr(params, 'est_card'):
        # pretend that postgres is true
        act_card = None
        pg_est_card = params.est_card
    else:
        print(params)
        raise NotImplementedError
    return act_card, pg_est_card


def q_err(cardinality_predict, cardinality_true):
    if cardinality_predict == 0 and cardinality_true == 0:
        q_error = 1.
    elif cardinality_true == 0:
        q_error = 1.
    elif cardinality_predict == 0:
        q_error = cardinality_true
    else:
        q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
    return q_error


def convert_query(schema, tables, filter_conditions, non_inclusive):
    filter_conditions = list(filter_conditions)

    query = Query(schema)
    query.query_type = QueryType.CARDINALITY

    # add tables
    query.table_set = tables

    cond_per_col = collections.defaultdict(set)
    # add filter conditions
    for t, c, o, v in filter_conditions:
        cond_per_col[(t, c)].add((o, v))

    # add relationships
    if len(tables) > 1:
        for table_i, table_j in itertools.combinations(tables, 2):
            for r in schema.table_dictionary[table_i].incoming_relationships:
                assert r.end == table_i
                if table_j == r.start:
                    query.add_join_condition(r.identifier)
                    remove_transitive_conds(cond_per_col, filter_conditions, r)

            for r in schema.table_dictionary[table_i].outgoing_relationships:
                assert r.start == table_i
                if table_j == r.end:
                    query.add_join_condition(r.identifier)
                    remove_transitive_conds(cond_per_col, filter_conditions, r)

        assert len(query.relationship_set) == len(query.table_set) - 1, "Unkown relationships joined"

    # add filter conditions
    for t, c, o, v in filter_conditions:
        if str(v).strip() == '::text':
            continue

        if isinstance(v, str):
            v = v.strip()
            if v.endswith('::text'):
                v = v.replace('::text', '')

            if v.endswith('::double precision'):
                v = v.replace('::double precision', '')
            v = v.strip("'")

        if non_inclusive:
            if o == '<=':
                o = '<'
            elif o == '>=':
                o = '>'

        query.add_where_condition(t, c + o + str(v))

    return query


def remove_transitive_conds(cond_per_col, filter_conditions, r):
    # for equality join, have the same condition only once
    conds_1 = cond_per_col[(r.end, r.end_attr)]
    conds_2 = cond_per_col[(r.start, r.start_attr)]
    intersecting_conds = list(conds_1.intersection(conds_2))
    if len(intersecting_conds) > 0:
        for o, v in intersecting_conds:
            filter_conditions.remove((r.end, r.end_attr, o, v))
            cond_per_col[(r.end, r.end_attr)].remove((o, v))
            cond_per_col[(r.start, r.start_attr)].remove((o, v))
