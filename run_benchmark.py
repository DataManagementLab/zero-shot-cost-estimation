import argparse

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.run_workload import run_workload


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    parser.add_argument('--db_name', default=None)
    parser.add_argument("--database_conn", dest='database_conn_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--database_kwargs", dest='database_kwarg_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--run_kwargs", dest='run_kwargs_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument('--query_timeout', default=30, type=int)
    parser.add_argument('--target', default=None)
    parser.add_argument('--source', default=None)
    parser.add_argument('--repetitions_per_query', default=1, type=int)
    parser.add_argument('--cap_workload', default=None, type=int)
    parser.add_argument('--min_query_ms', default=100, type=int)
    parser.add_argument('--with_indexes', action='store_true')
    parser.add_argument('--run_workload', action='store_true')

    args = parser.parse_args()

    if args.database_kwarg_dict is None:
        args.database_kwarg_dict = dict()

    force = True

    if args.run_workload:
        run_workload(args.source, args.database, args.db_name, args.database_conn_dict, args.database_kwarg_dict,
                     args.target, args.run_kwargs_dict, args.repetitions_per_query, args.query_timeout,
                     with_indexes=args.with_indexes, cap_workload=args.cap_workload, min_runtime=args.min_query_ms)
