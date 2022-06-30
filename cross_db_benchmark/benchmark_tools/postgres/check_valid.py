from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_plan


def check_valid(curr_statistics, min_runtime=100, verbose=True):
    # Timemouts are also a valid signal in learning
    if 'timeout' in curr_statistics and curr_statistics['timeout']:
        if verbose:
            print("Invalid since it ran into a timeout")
        return False

    try:
        analyze_plans = curr_statistics['analyze_plans']

        if analyze_plans is None or len(analyze_plans) == 0:
            if verbose:
                print("Unvalid because no analyze plans are available")
            return False

        analyze_plan = analyze_plans[0]

        analyze_plan, ex_time, _ = parse_plan(analyze_plan, analyze=True, parse=True)
        analyze_plan.parse_lines_recursively()

        if analyze_plan.min_card() == 0:
            if verbose:
                print("Unvalid because of zero cardinality")
            return False

        if ex_time < min_runtime:
            if verbose:
                print("Unvalid because of too short runtime")
            return False

        return True
    except:
        if verbose:
            print("Unvalid due to error")
        return False
