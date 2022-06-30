# different features used for plan nodes, filter columns etc. used for postgres plans

class PostgresTrueCardDetail:
    PLAN_FEATURES = ['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresEstSystemCardDetail:
    PLAN_FEATURES = ['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresDeepDBEstSystemCardDetail:
    PLAN_FEATURES = ['dd_est_card', 'est_width', 'workers_planned', 'op_name', 'dd_est_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']
