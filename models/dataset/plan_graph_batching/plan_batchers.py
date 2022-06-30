from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from models.dataset.plan_graph_batching.postgres_plan_batching import postgres_plan_collator

plan_collator_dict = {
    DatabaseSystem.POSTGRES: postgres_plan_collator
}
