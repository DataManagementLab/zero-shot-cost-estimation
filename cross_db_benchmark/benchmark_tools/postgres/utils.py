import math


def plan_statistics(plan_op, tables=None, filter_columns=None, operators=None, skip_columns=False, conv_to_dict=False):
    if tables is None:
        tables = set()
    if operators is None:
        operators = set()
    if filter_columns is None:
        filter_columns = set()

    params = plan_op.plan_parameters

    if conv_to_dict:
        params = vars(params)

    if 'table' in params:
        tables.add(params['table'])
    if 'op_name' in params:
        operators.add(params['op_name'])
    if 'filter_columns' in params and not skip_columns:
        list_columns(params['filter_columns'], filter_columns)

    for c in plan_op.children:
        plan_statistics(c, tables=tables, filter_columns=filter_columns, operators=operators, skip_columns=skip_columns,
                        conv_to_dict=conv_to_dict)

    return tables, filter_columns, operators


def child_prod(p, feature_name, default=1):
    child_feat = [c.plan_parameters.get(feature_name) for c in p.children
                  if c.plan_parameters.get(feature_name) is not None]
    if len(child_feat) == 0:
        return default
    return math.prod(child_feat)


def list_columns(n, columns):
    columns.add((n.column, n.operator))
    for c in n.children:
        list_columns(c, columns)
