def combine_traces(runs):
    start_plan = runs[0]
    for p in runs[1:]:
        start_plan.query_list += p.query_list
        start_plan.total_time_secs += p.total_time_secs

    return start_plan
