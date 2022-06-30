def inflate_card_errors_pg(p, factor):
    # inflate the errors (both over- and underestimation)
    params = p.plan_parameters
    if params.act_card > params.est_card:
        q_err = params.act_card / params.est_card
        q_err = (q_err - 1) * factor + 1
        err_card = params.act_card / q_err

    else:
        q_err = params.est_card / params.act_card
        q_err = (q_err - 1) * factor + 1
        err_card = params.act_card * q_err

    if err_card < 1:
        err_card = 1
    err_card = float(int(err_card))

    params.est_card = err_card
    params.act_card = err_card

    prod = 1
    for c in p.children:
        prod *= inflate_card_errors_pg(c, factor)

    params.est_children_card = prod
    params.act_children_card = prod

    return err_card
