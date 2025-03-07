using HighDimensionalOptimalPolicies: traces
using HighDimensionalOptimalPolicies: pigeons


function prepare_ps()
    nside = 5
    K = 5
    mynet = init_env_square_nodiag(nside)
    add_pop_flows!(mynet)


    obj_fun = let mynet = mynet
        edges_to_upgrade -> travel_welfare(edges_to_upgrade, mynet)
    end

    next_policy = swap_edges_to_upgrade

    num_edges = size(mynet.edges, 1)
    edges_to_upgrade_num = sample(1:num_edges, K; replace = false)
    edges_to_upgrade_init = collect(1:num_edges) .∈ Ref(edges_to_upgrade_num)

    β = 100.0

    pt = pigeons(
        target = MyLogPotential(obj_fun, 50.0),
        reference = MyLogPotential(obj_fun, 0.0),
        extended_traces = true,
        record = [traces],
        n_chains = 5,
        n_rounds = 4,
        explorer = MyIndependenceSampler(next_policy))
end

function ps()

end