import collections
import re

import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator
from cross_db_benchmark.benchmark_tools.postgres.plan_operator import PlanOperator
from cross_db_benchmark.benchmark_tools.postgres.utils import plan_statistics

planning_time_regex = re.compile('planning time: (?P<planning_time>\d+.\d+) ms')
ex_time_regex = re.compile('execution time: (?P<execution_time>\d+.\d+) ms')
init_plan_regex = re.compile("InitPlan \d+ \(returns \$\d\)")
join_columns_regex = re.compile('\w+\.\w+ ?= ?\w+\.\w+')


def create_node(lines_plan_operator, operators_current_level):
    if len(lines_plan_operator) > 0:
        last_operator = PlanOperator(lines_plan_operator)
        operators_current_level.append(last_operator)
        lines_plan_operator = []
    return lines_plan_operator


def count_left_whitespaces(a):
    return len(a) - len(a.lstrip(' '))


def parse_recursively(parent, plan, offset, depth):
    lines_plan_operator = []
    i = offset
    operators_current_level = []
    while i < len(plan):
        # new operator
        if plan[i].strip().startswith('->'):

            # create plan node for previous one
            lines_plan_operator = create_node(lines_plan_operator, operators_current_level)

            # if plan operator is deeper
            new_depth = count_left_whitespaces(plan[i])
            if new_depth > depth:
                assert len(operators_current_level) > 0, "No parent found at this level"
                i = parse_recursively(operators_current_level[-1], plan, i, new_depth)

            # one step up in recursion
            elif new_depth < depth:
                break

            # new operator in current depth
            elif new_depth == depth:
                lines_plan_operator.append(plan[i])
                i += 1

        else:
            lines_plan_operator.append(plan[i])
            i += 1

    create_node(lines_plan_operator, operators_current_level)

    # any node in the recursion
    if parent is not None:
        parent.children = operators_current_level
        return i

    # top node
    else:
        # there should only be one top node
        assert len(operators_current_level) == 1
        return operators_current_level[0]


def parse_plan(analyze_plan_tuples, analyze=True, parse=True):
    plan_steps = analyze_plan_tuples
    if isinstance(analyze_plan_tuples[0], tuple) or isinstance(analyze_plan_tuples[0], list):
        plan_steps = [t[0] for t in analyze_plan_tuples]

    # for some reason this is missing in postgres
    # in order to parse this, we add it
    plan_steps[0] = '->  ' + plan_steps[0]

    ex_time = 0
    planning_time = 0
    planning_idx = -1
    if analyze:
        for i, plan_step in enumerate(plan_steps):
            plan_step = plan_step.lower()
            ex_time_match = planning_time_regex.match(plan_step)
            if ex_time_match is not None:
                planning_idx = i
                planning_time = float(ex_time_match.groups()[0])

            ex_time_match = ex_time_regex.match(plan_step)
            if ex_time_match is not None:
                ex_time = float(ex_time_match.groups()[0])

        assert ex_time != 0 and planning_time != 0
        plan_steps = plan_steps[:planning_idx]

    root_operator = None
    if parse:
        root_operator = parse_recursively(None, plan_steps, 0, 0)

    return root_operator, ex_time, planning_time


def parse_plans(run_stats, min_runtime=100, max_runtime=30000, parse_baseline=False, cap_queries=None,
                parse_join_conds=False, include_zero_card=False, explain_only=False):
    # keep track of column statistics
    column_id_mapping = dict()
    table_id_mapping = dict()
    partial_column_name_mapping = collections.defaultdict(set)

    database_stats = run_stats.database_stats
    # enrich column stats with table sizes
    table_sizes = dict()
    for table_stat in database_stats.table_stats:
        table_sizes[table_stat.relname] = table_stat.reltuples

    for i, column_stat in enumerate(database_stats.column_stats):
        table = column_stat.tablename
        column = column_stat.attname
        column_stat.table_size = table_sizes[table]
        column_id_mapping[(table, column)] = i
        partial_column_name_mapping[column].add(table)

    # similar for table statistics
    for i, table_stat in enumerate(database_stats.table_stats):
        table = table_stat.relname
        table_id_mapping[table] = i

    # parse individual queries
    parsed_plans = []
    avg_runtimes = []
    no_tables = []
    no_filters = []
    op_perc = collections.defaultdict(int)
    for q in tqdm(run_stats.query_list):

        # either only parse explain part of query or skip entirely
        curr_explain_only = explain_only
        # do not parse timeout queries
        if hasattr(q, 'timeout') and q.timeout:
            continue

        alias_dict = dict()
        if not curr_explain_only:
            if q.analyze_plans is None:
                continue

            if len(q.analyze_plans) == 0:
                continue

            # subqueries are currently not supported
            analyze_str = ''.join([l[0] for l in q.verbose_plan])
            if 'SubPlan' in analyze_str or 'InitPlan' in analyze_str:
                continue

            # subquery is empty due to logical constraints
            if '->  Result  (cost=0.00..0.00 rows=0' in analyze_str:
                continue

            # check if it just initializes a plan
            if isinstance(q.analyze_plans[0][0], list):
                analyze_plan_string = ''.join(l[0] for l in q.analyze_plans[0])
            else:
                analyze_plan_string = ''.join(q.analyze_plans)
            if init_plan_regex.search(analyze_plan_string) is not None:
                continue

            # compute average execution and planning times
            ex_times = []
            planning_times = []
            for analyze_plan in q.analyze_plans:
                _, ex_time, planning_time = parse_plan(analyze_plan, analyze=True, parse=False)
                ex_times.append(ex_time)
                planning_times.append(planning_time)
            avg_runtime = sum(ex_times) / len(ex_times)

            # parse the plan as a tree
            analyze_plan, _, _ = parse_plan(q.analyze_plans[0], analyze=True, parse=True)

            # parse information contained in operator nodes (different information in verbose and analyze plan)
            analyze_plan.parse_lines_recursively(alias_dict=alias_dict, parse_baseline=parse_baseline,
                                                 parse_join_conds=parse_join_conds)

        # elif timeout:
        #     avg_runtime = float(2 * max_runtime)

        else:
            avg_runtime = 0

        # only explain plan (not executed)
        verbose_plan, _, _ = parse_plan(q.verbose_plan, analyze=False, parse=True)
        verbose_plan.parse_lines_recursively(alias_dict=alias_dict, parse_baseline=parse_baseline,
                                             parse_join_conds=parse_join_conds)

        if not curr_explain_only:
            # merge the plans with different information
            analyze_plan.merge_recursively(verbose_plan)

        else:
            analyze_plan = verbose_plan

        tables, filter_columns, operators = plan_statistics(analyze_plan)

        analyze_plan.parse_columns_bottom_up(column_id_mapping, partial_column_name_mapping, table_id_mapping,
                                             alias_dict=alias_dict)

        analyze_plan.plan_runtime = avg_runtime

        def augment_no_workers(p, top_no_workers=0):
            no_workers = p.plan_parameters.get('workers_planned')
            if no_workers is None:
                no_workers = top_no_workers

            p.plan_parameters['workers_planned'] = top_no_workers

            for c in p.children:
                augment_no_workers(c, top_no_workers=no_workers)

        augment_no_workers(analyze_plan)

        if not curr_explain_only:
            # check if result is None
            if analyze_plan.min_card() == 0 and not include_zero_card:
                continue

            if min_runtime is not None and avg_runtime < min_runtime:
                continue

            if avg_runtime > max_runtime:
                continue

        # add joins for MSCN baseline
        if parse_baseline:
            join_conds = []
            join_str = q.sql.split(' FROM ')[1]
            # search for explicit join conditions
            if ' JOIN ' in join_str:
                if ' WHERE ' in join_str:
                    join_str = join_str.split(' WHERE ')[0]

                for p_join_str in join_str.split(' JOIN ')[1:]:
                    p_join_str = p_join_str.split(' ON ')[1]
                    join_cond = normalize_join_condition(p_join_str)
                    join_conds.append(join_cond)
            else:
                # check for join comparisons in the where clause
                for filter_m in join_columns_regex.finditer(join_str):
                    p_join_str = filter_m.group()
                    join_cond = normalize_join_condition(p_join_str)
                    join_conds.append(join_cond)

            tables, _, _ = plan_statistics(analyze_plan, skip_columns=True)
            if len(tables) != len(join_conds) + 1:
                print(f"No tables {len(tables)} != no join conds + 1 {len(join_conds) + 1}")
            analyze_plan.join_conds = join_conds

        # collect statistics
        avg_runtimes.append(avg_runtime)
        no_tables.append(len(tables))
        for _, op in filter_columns:
            op_perc[op] += 1
        # log number of filters without counting AND, OR
        no_filters.append(len([fc for fc in filter_columns if fc[0] is not None]))

        parsed_plans.append(analyze_plan)

        if cap_queries is not None and len(parsed_plans) >= cap_queries:
            print(f"Parsed {cap_queries} queries. Stopping parsing.")
            break

        if parse_baseline:
            def list_columns(n, columns):
                if n['operator'] != str(LogicalOperator.AND) and n['operator'] != str(LogicalOperator.OR):
                    columns.append((n['column'], n['operator'], n['literal']))
                for c in n['children']:
                    list_columns(c, columns)

            def list_col_rec(n, columns):
                filter_column = n.plan_parameters.get('filter_columns')
                if filter_column is not None:
                    list_columns(filter_column, columns)
                for c in n.children:
                    list_col_rec(c, columns)

            columns = []
            list_col_rec(analyze_plan, columns)
            if ' JOIN ' in q.sql:
                if ' WHERE ' not in q.sql:
                    exp_no_filter = 0
                else:
                    exp_no_filter = q.sql.split(' WHERE ')[1].count(' AND ') + 1

                if not exp_no_filter <= len(columns):
                    print(f"Warning: did not find enough filters exp_no_filter: {exp_no_filter} ({q.sql}), "
                          f"columns: {columns}")

    # statistics in seconds
    print(f"Table statistics: "
          f"\n\tmean: {np.mean(no_tables):.1f}"
          f"\n\tmedian: {np.median(no_tables)}"
          f"\n\tmax: {np.max(no_tables)}")
    print("Operators statistics (appear in x% of queries)")
    for op, op_count in op_perc.items():
        print(f"\t{str(op)}: {op_count / len(avg_runtimes) * 100:.0f}%")
    print(f"Runtime statistics: "
          f"\n\tmedian: {np.median(avg_runtimes) / 1000:.2f}s"
          f"\n\tmax: {np.max(avg_runtimes) / 1000:.2f}s"
          f"\n\tmean: {np.mean(avg_runtimes) / 1000:.2f}s")
    print(f"Parsed {len(parsed_plans)} plans ({len(run_stats.query_list) - len(parsed_plans)} had zero-cardinalities "
          f"or were too fast).")

    parsed_runs = dict(parsed_plans=parsed_plans, database_stats=database_stats,
                       run_kwargs=run_stats.run_kwargs)

    stats = dict(
        runtimes=str(avg_runtimes),
        no_tables=str(no_tables),
        no_filters=str(no_filters)
    )

    return parsed_runs, stats


def normalize_join_condition(p_join_str):
    join_conds = p_join_str.split('AND')
    join_conds = [normalize_single_join_condition(jc.strip()) for jc in join_conds]
    join_conds = sorted(join_conds)
    join_conds = ' AND '.join(join_conds)
    return join_conds


def normalize_single_join_condition(p_join_str):
    join_cond = p_join_str.split('=')
    assert len(join_cond) == 2
    for i in [0, 1]:
        join_cond[i] = join_cond[i].strip()
    join_cond = sorted(join_cond)
    join_cond = f'{join_cond[0]} = {join_cond[1]}'
    return join_cond
