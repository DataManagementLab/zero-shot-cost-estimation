from models.dataset.plan_featurization import postgres_plan_featurizations
from models.zero_shot_models.zero_shot_model import ZeroShotModel


class PostgresZeroShotModel(ZeroShotModel):
    """
    Zero-shot cost estimation model for postgres.
    """
    def __init__(self, plan_featurization_name=None, **zero_shot_kwargs):
        plan_featurization, encoders = None, None
        if plan_featurization_name is not None:
            plan_featurization = postgres_plan_featurizations.__dict__[plan_featurization_name]

            # define the MLPs for the different node types in the graph representation of queries
            encoders = [
                ('column', plan_featurization.COLUMN_FEATURES),
                ('table', plan_featurization.TABLE_FEATURES),
                ('output_column', plan_featurization.OUTPUT_COLUMN_FEATURES),
                ('filter_column', plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES),
                ('plan', plan_featurization.PLAN_FEATURES),
                ('logical_pred', plan_featurization.FILTER_FEATURES),
            ]

        # define messages passing which is peculiar for postgres
        prepasses = [dict(model_name='column_output_column', e_name='col_output_col')]
        tree_model_types = ['column_output_column']

        super().__init__(plan_featurization=plan_featurization, encoders=encoders, prepasses=prepasses,
                         add_tree_model_types=tree_model_types, **zero_shot_kwargs)
