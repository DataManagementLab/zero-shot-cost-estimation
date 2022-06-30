from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from models.zero_shot_models.specific_models.postgres_zero_shot import PostgresZeroShotModel

# dictionary with tailored model for each database system (we learn one model per system that generalizes across
#   databases (i.e., datasets) but on the same database system)
zero_shot_models = {
    DatabaseSystem.POSTGRES: PostgresZeroShotModel
}
