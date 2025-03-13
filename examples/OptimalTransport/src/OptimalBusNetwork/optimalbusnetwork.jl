gr()

struct TravelNetwork
    coords::Vector{NTuple{2, Float64}}
    pops::Vector{Float64}
    adjcosts::Matrix{Float64}
    edges::Vector{NTuple{2, Int}}
    flows::Matrix{Float64}
    num_upgrades::Int
end

function TravelNetwork(coords, pops, adjcosts, num_upgrades)
    flows = get_flows(pops, adjcosts)

    edges = filter(t -> isfinite(adjcosts[t]), CartesianIndices(adjcosts))

    TravelNetwork(coords, pops, adjcosts, edges, flows, num_upgrades)
end

function pairwisecost(adjcosts; θ = 8.0)
    B = inv(I - Symmetric(adjcosts) .^(-θ))

    τ = B .^(-1/θ)
end

function get_flows(pops, adjcosts; θ = 8.0, β = 1.0)
    τ = pairwisecost(adjcosts; θ)

    flows = similar(adjcosts)

    for i in eachindex(pops)
        costs_to_other_nodes = @view τ[i, :]
        πs = (costs_to_other_nodes .^ (-0.5) .* pops) .^ β
        πs ./= sum(πs)
        flows[i, :] = pops[i] * πs
    end

    flows
end

function get_edges_dict(net)
    n_coords = length(net.coords)
    edges_dict = Dict()
    for i in 1:n_coords
        for j in i:n_coords
            c = net.adjcosts[i, j]
            if isfinite(c)
                edges_dict[(i, j)] = c
            end
        end
    end
    edges_dict
end

"""
    swap_edges_to_upgrade(edges_to_upgrade)

"""
swap_edges_to_upgrade(edges_to_upgrade) = swap_edges_to_upgrade(Random.default_rng(), edges_to_upgrade)

function swap_edges_to_upgrade(rng, edges_to_upgrade)
    improved_edges = findall(edges_to_upgrade)
    unimproved_edges = findall(edges_to_upgrade .== false)

    edge_to_drop = sample(rng, improved_edges)
    edge_to_add = sample(rng, unimproved_edges)

    new_edges_to_upgrade = copy(edges_to_upgrade)
    new_edges_to_upgrade[edge_to_drop] = false
    new_edges_to_upgrade[edge_to_add] = true

    return new_edges_to_upgrade
end

function get_n̄(net)
    (;edges = net)
    num_edges = length(net.edges)
end

function plot_network(
    net::TravelNetwork;
    title = "",
    min_width = 0.5,
    max_width = 10.0,
    max_marker_size = 4.0,
    edges_to_upgrade = nothing,
    average_edges_to_upgrade = nothing)

    if !isnothing(edges_to_upgrade) && !isnothing(average_edges_to_upgrade)
        throw(ArgumentError("Cannot supply both edges_to_upgrade and average_edges_to_ugrade"))
    end

    (; coords, pops, adjcosts, edges, flows) = net

    plot(; title)

    edges_dict = get_edges_dict(net)
    cs = values(edges_dict)
    min_cost = minimum(cs)
    max_cost = maximum(cs)
    if max_cost == min_cost
        max_cost = 1.1 * min_cost
    end
    max_cost_diff = max_cost - min_cost
    max_line_diff = (max_width - min_width)
    for k in keys(edges_dict)
        i, j = k
        coord_i = coords[i]
        coord_j = coords[j]
        c = edges_dict[k]

        linewidth = ((c - min_cost) / max_cost_diff) * max_line_diff + max_width
        plot!([coord_i, coord_j]; color = "grey", linewidth, label = "")
    end

    if edges_to_upgrade !== nothing
        for ei in eachindex(edges_to_upgrade)
            e = edges_to_upgrade[ei]
            i, j = edges[ei]
            if e == true
                coord_i = coords[i]
                coord_j = coords[j]
                plot!([coord_i, coord_j]; color = "black", linewidth = max_width, label = "")
            end
        end
    end

    if average_edges_to_upgrade !== nothing
        # Just start over again with a new plot
        # TODO: Figure out a better way to do this
        plot(; title)
        min_cost = minimum(average_edges_to_upgrade)
        max_cost = maximum(average_edges_to_upgrade)
        if max_cost == min_cost
            max_cost = 1.1 * min_cost
        end
        max_cost_diff = max_cost - min_cost
        max_line_diff = (max_width - min_width)
        for edges_itr in 1:length(edges)
            edge = edges[edges_itr]
            i, j = edge
            coord_i = coords[i]
            coord_j = coords[j]
            c = average_edges_to_upgrade[edges_itr]

            linewidth = ((c - min_cost) / max_cost_diff) * max_line_diff
            plot!([coord_i, coord_j]; color = "black", linewidth, label = "")
        end
    end

    marker_sizes = (pops ./ maximum(pops)) .* max_marker_size
    Plots.scatter!(coords; markersize = marker_sizes, label = "")
end

function random_travel_network(n_coords; num_upgrades = nothing)
    coords_unsorted = [(rand(), rand()) for i in 1:n_coords]
    coords = sort(coords_unsorted, lt = (x, y) -> (x[1] < y[1]))

    tri = DelaunayTriangulation.triangulate(coords)

    adjcosts = fill(Inf, n_coords, n_coords)
    for e in DelaunayTriangulation.each_solid_edge(tri)
        u, v = DelaunayTriangulation.edge_vertices(e)
        p, q = DelaunayTriangulation.get_point(tri, u, v)

        d = norm(p .- q)
        adjcosts[u, v] = 2.0
        adjcosts[v, u] = 2.0
    end
    adjcosts = 2 .* adjcosts ./ minimum(adjcosts)

    # Give a large population to the 5 closest coordinates on the
    # line y = x
    dist_from_line = map(coords) do (x, y)
        abs(x - y)
    end
    dist_rank = StatsBase.ordinalrank(dist_from_line)

    pops = fill(1.0, n_coords)
    for i in eachindex(pops)
        if (dist_rank[i] / n_coords) <= 0.2
            pops[i] = 50.0
        end
    end
    pops = pops ./ sum(pops)

    if isnothing(num_upgrades)
        num_upgrades = floor(Int, n_coords * 0.2)
    end

    TravelNetwork(coords, pops, adjcosts, num_upgrades)
end

function square_travel_network(n_sides; num_upgrades = nothing)
    coords_ints = map(Iterators.product(1:n_sides, 1:n_sides)) do (i, j)
        (i, j)
    end |> vec

    n_coords = length(coords_ints)

    adjcosts = map(Iterators.product(coords_ints, coords_ints)) do (i, j)
        xi, yi = i
        xj, yj = j
        if (xj == (xi + 1) || xj == (xi - 1)) && (yi == yj)
            2.0
        elseif (yj == (yi + 1) || yj == (yi - 1)) && (xi == xj)
            2.0
        else
            Inf
        end
    end

    coords = map(coords_ints) do c
        c ./ n_sides
    end

    # Give a large population to the 5 closest coordinates on the
    # line y = x
    dist_from_line = map(coords) do (x, y)
        abs(x - y)
    end
    dist_rank = StatsBase.ordinalrank(dist_from_line)

    pops = fill(1.0, n_coords)
    for i in eachindex(pops)
        if (dist_rank[i] / n_coords) <= 0.2
            pops[i] = 50.0
        end
    end
    pops = pops ./ sum(pops)

    if isnothing(num_upgrades)
        num_upgrades = floor(Int, n_coords * 0.2)
    end

    TravelNetwork(coords, pops, adjcosts, num_upgrades)
end

function get_upgraded_network(edges_to_upgrade, net::TravelNetwork; cost_ratio = 0.75)
    (; edges, adjcosts) = net

    adjcosts_new = copy(adjcosts)
    for i in eachindex(edges_to_upgrade)
        e = edges_to_upgrade[i]
        if e == true
            ind = CartesianIndex(edges[i])
            adjcosts_new[ind] = adjcosts[ind] * cost_ratio
        end
    end
    new_net = @set net.adjcosts = adjcosts_new
    new_net
end

get_initial_upgrade(net) = get_initial_upgrade(default_rng(), net)
function get_initial_upgrade(rng, net::TravelNetwork)
    (; edges, num_upgrades) = net
    n = length(edges)
    edges_to_upgrade = fill(false, n)
    inds_to_upgrade = sample(rng, 1:n, num_upgrades; replace = false)
    edges_to_upgrade[inds_to_upgrade] .= true
    edges_to_upgrade
end

function welfare(net::TravelNetwork)
    (; adjcosts, flows) = net
    τ = pairwisecost(adjcosts)

    # average costs, weighted by population flows
    # Recall that flows are exogenous in this set-up.
    return (-1) * sum(flows .* τ) / sum(flows)
end

lchoose(a,b) = SpecialFunctions.logabsbinomial(a,b)[1]

function get_log_n̄(net)
    (; num_upgrades, edges) = net
    lchoose(length(edges), num_upgrades)
end

function test_travel_network()
    #net = square_travel_network(5)
    net = random_travel_network(10)

    initfun, objfun, nextfun = let net = net
        initfun = rng -> get_initial_upgrade(rng, net)
        objfun = edges_to_upgrade -> begin
            new_net = get_upgraded_network(edges_to_upgrade, net)
            welfare(new_net)
        end
        nextfun = (rng, edges_to_upgrade) -> swap_edges_to_upgrade(rng, edges_to_upgrade)
        initfun, objfun, nextfun
    end

    β = 500.0

    out = HDOP.get_best_policy(HDOP.PigeonsSolver(); initfun, objfun, nextfun, β)
    last_policy = HDOP.get_last_policy(out)
    average_policy = HDOP.get_average_policy(out)
    #plot_network(net; edges_to_upgrade = last_policy)
    plot_network(net; average_edges_to_upgrade = average_policy)

    (; net, out)
#=    best_edges_to_upgrade_pigeons = best_edges_to_upgrade_pigeons .== 1

    best_edges_to_upgrade_mcmc = HDOP.get_best_policy(HDOP.MCMCSolver(); initfun, objfun, nextfun, β)

    best_edges_to_upgrade_temperedmcmc = HDOP.get_best_policy(HDOP.TemperedMCMCSolver(); initfun, objfun, nextfun, β)

    @show sum(best_edges_to_upgrade_pigeons)
    @show sum(best_edges_to_upgrade_mcmc)
    @show sum(best_edges_to_upgrade_temperedmcmc)

    p_pigeons = plot_network(net; edges_to_upgrade = best_edges_to_upgrade_pigeons, title = "Pigeons.jl")
    p_mcmc = plot_network(net; edges_to_upgrade = best_edges_to_upgrade_mcmc, title = "Metropolis Hastings")
    p_temperedmcmc = plot_network(net; edges_to_upgrade = best_edges_to_upgrade_temperedmcmc, title = "TemperedMCMC.jl")
    plot(p_pigeons, p_mcmc, p_temperedmcmc)=#
end

function plot_sol(net, out)
    last_policy = HDOP.get_last_policy(out)
    average_policy = HDOP.get_average_policy(out)
    plot_network(net; average_edges_to_upgrade = average_policy)
end



