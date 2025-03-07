function abstractmcmc()
    nside = 5
    mynet_init = init_env_square_nodiag(nside)
    add_pop_flows!(mynet_init)

    mhp = prepare_metropolitan_hastings(deepcopy(mynet_init))
    K = 5
    num_edges = size(mynet_init.edges, 1)
    edges_to_upgrade_num = sample(1:num_edges, K; replace = false)
    edges_to_upgrade_init = collect(1:num_edges) .∈ Ref(edges_to_upgrade_num)

    β = 100.0

    obj_fun = let mynet_init = mynet_init, β = β
        edges_to_upgrade -> exp(β * travel_welfare(edges_to_upgrade, mynet_init))
    end

    model = HDOP.ValueModel1(obj_fun)

    proposal = HDOP.GenericProposal1(swap_edges_to_upgrade)

    spl = HDOP.MetropolisHastings1(proposal)

    chain = sample(model, spl, 1000, chain_type=Chains, initial_params = edges_to_upgrade_init)

    policy_best = last(chain).params

    mynet_best = deepcopy(mynet_init)
    upgrade_highways!(mynet_best, policy_best)
    mynet_best
end