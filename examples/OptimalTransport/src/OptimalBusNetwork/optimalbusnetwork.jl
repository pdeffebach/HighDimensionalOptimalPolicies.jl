# Edges are stored without direction.
struct TravelNetwork
    coords::Vector{NTuple{2, Float64}}
    pops::Vector{Float64}
    adjcosts::Matrix{Float64}
    edges::Vector{NTuple{2, Int}}
    flows::Matrix{Float64}
    n_upgrades::Int
end

function Base.show(io::IO, t::TravelNetwork)
    n_coords = length(t.coords)
    n_edges = length(t.edges)
    s = "Travel network with $n_coords nodes and $n_edges edges"
    print(io, s)
end

function TravelNetwork(coords, pops, adjcosts, n_upgrades)
    flows = get_flows(pops, adjcosts)
    n_coords = length(coords)
    edges = Tuple{Int, Int}[]
    for i in 1:n_coords
        for j in i:n_coords
            c = adjcosts[i, j]
            if isfinite(c)
                push!(edges, (i, j))
            end
        end
    end
    TravelNetwork(coords, pops, adjcosts, edges, flows, n_upgrades)
end

function pairwisecost(adjcosts; θ = 8.0)
    B = inv(I - Symmetric(adjcosts) .^(-θ))

    τ = B .^(-1/θ)
end

function get_flows(pops, adjcosts; θ = 8.0, max_invtemp = 1.0)
    τ = pairwisecost(adjcosts; θ)

    flows = similar(adjcosts)

    for i in eachindex(pops)
        costs_to_other_nodes = @view τ[i, :]
        πs = (costs_to_other_nodes .^ (-0.5) .* pops) .^ max_invtemp
        πs ./= sum(πs)
        flows[i, :] = pops[i] * πs
    end

    flows
end

"""
Returns a vector of edges where (i, j) is treated as
identical to (j, i)
"""
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

function swap_edges_to_upgrade(rng, edges_to_upgrade, n_edges_to_swap)
    improved_edges = findall(edges_to_upgrade)
    unimproved_edges = findall(edges_to_upgrade .== false)

    edge_to_drop = sample(rng, improved_edges, n_edges_to_swap, replace = false)
    edge_to_add = sample(rng, unimproved_edges, n_edges_to_swap, replace = false)

    new_edges_to_upgrade = copy(edges_to_upgrade)
    new_edges_to_upgrade[edge_to_drop] .= false
    new_edges_to_upgrade[edge_to_add] .= true

    return new_edges_to_upgrade
end

function plot_network(
    net::TravelNetwork;
    title = "",
    min_line_width = 0.5,
    max_line_width = 8.0,
    min_marker_size = 5.0,
    max_marker_size = 10.0,
    edges_to_upgrade = nothing,
    average_edges_to_upgrade = nothing,
    edges_to_upgrade_vec = nothing)

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
    max_line_diff = (max_line_width - min_line_width)
    for k in keys(edges_dict)
        i, j = k
        coord_i = coords[i]
        coord_j = coords[j]
        c = edges_dict[k]

        linewidth = ((c - min_cost) / max_cost_diff) * max_line_diff + min_line_width
        plot!([coord_i, coord_j]; color = "grey", linewidth, label = "")
    end

    if edges_to_upgrade !== nothing
        for ei in eachindex(edges_to_upgrade)
            e = edges_to_upgrade[ei]
            i, j = edges[ei]
            if e == true
                coord_i = coords[i]
                coord_j = coords[j]
                plot!([coord_i, coord_j]; color = "black", linewidth = max_line_width, label = "")
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
        max_line_diff = (max_line_width - min_line_width)
        for edges_itr in 1:length(edges)
            edge = edges[edges_itr]
            i, j = edge
            coord_i = coords[i]
            coord_j = coords[j]
            c = average_edges_to_upgrade[edges_itr]

            linewidth = ((c - min_cost) / max_cost_diff) * max_line_diff + min_line_width
            plot!([coord_i, coord_j]; color = "black", linewidth, label = "")
        end
    end

    # A vector of vectors
    if edges_to_upgrade_vec !== nothing
        line_colors = distinguishable_colors(length(edges_to_upgrade_vec), colorant"blue")
        for e_upsi in eachindex(edges_to_upgrade_vec)
            e_ups = edges_to_upgrade_vec[e_upsi]
            line_color = line_colors[e_upsi]
            for ei in eachindex(e_ups)
                e = e_ups[ei]
                i, j = edges[ei]
                if e == true
                    coord_i = coords[i]
                    coord_j = coords[j]
                    plot!([coord_i, coord_j]; color = line_color, linewidth = max_line_width, label = "", alpha = 0.5)
                end
            end
        end
    end

    max_pop = maximum(pops)
    min_pop = minimum(pops)
    if max_pop == min_pop
        max_pop = 1.1 * min_pop
    end
    max_pop_diff = (max_pop - min_pop)
    max_size_diff = max_marker_size - min_marker_size
    marker_sizes = @. ((pops - min_pop) / max_pop_diff) * max_size_diff + min_marker_size
    Plots.scatter!(coords; markersize = marker_sizes, label = "", color = "firebrick2", xlim = [-0.05, 1.05], ylim = [-0.05, 1.05])
end

function random_travel_network(n_coords; n_upgrades = nothing, frac_upgrades = nothing, diagonal_path = true, unequal_pops = true)
    coords_unsorted = [(rand(), rand()) for i in 1:n_coords]
    coords = sort(coords_unsorted, lt = (x, y) -> (x[1] < y[1]))

    tri = DelaunayTriangulation.triangulate(coords)

    adjcosts = fill(Inf, n_coords, n_coords)
    n_edges = 0
    for e in DelaunayTriangulation.each_solid_edge(tri)
        u, v = DelaunayTriangulation.edge_vertices(e)
        p, q = DelaunayTriangulation.get_point(tri, u, v)

        d = norm(p .- q)
        adjcosts[u, v] = 2.0
        adjcosts[v, u] = 2.0
        n_edges += 1
    end
    adjcosts = 2 .* adjcosts ./ minimum(adjcosts)

    # Give a large population to the 5 closest coordinates on the
    # line y = x
    if unequal_pops == true
        if diagonal_path == true
            dist_from_line = map(coords) do (x, y)
                abs(x - y)
            end
            dist_rank = StatsBase.ordinalrank(dist_from_line)

            pops = fill(1.0, n_coords)
            for i in eachindex(pops)
                if (dist_rank[i] / n_coords) <= 1/5
                    pops[i] = 5.0
                end
            end
            pops = pops ./ sum(pops)
        else
            pops = fill(1.0, n_coords)
            inds = sample(1:n_coords, floor(Int, n_coords / 5))
            pops[inds] .= 5
            pops = pops ./ sum(pops)
        end
    else
        pops = fill(1.0, n_coords)
        pops = pops ./ sum(pops)
    end

    if isnothing(n_upgrades)
        n_upgrades = ceil(Int, n_edges * frac_upgrades)
    end

    TravelNetwork(coords, pops, adjcosts, n_upgrades)
end

function square_travel_network(n_coords_approx; n_upgrades = nothing, frac_upgrades = nothing, diagonal_path = true, unequal_pops = true)
    n_sides = ceil(Int, sqrt(n_coords_approx))
    coords_ints = map(Iterators.product(1:n_sides, 1:n_sides)) do (i, j)
        (i, j)
    end |> vec

    n_coords = length(coords_ints)
    n_edges = 0
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

    # Give a large population to the 5 closest coordinates on the
    # line y = x
    if unequal_pops == true
        if diagonal_path == true
            dist_from_line = map(coords) do (x, y)
                abs(x - y)
            end
            dist_rank = StatsBase.ordinalrank(dist_from_line)

            pops = fill(1.0, n_coords)
            for i in eachindex(pops)
                if (dist_rank[i] / n_coords) <= 1/5
                    pops[i] = 5.0
                end
            end
            pops = pops ./ sum(pops)
        else
            pops = fill(1.0, n_coords)
            inds = sample(1:n_coords, floor(Int, n_coords / 5))
            pops[inds] .= 5
            pops = pops ./ sum(pops)
        end
    else
        pops = fill(1.0, n_coords)
        pops = pops ./ sum(pops)
    end

    # why do we not calculate n_edges before?
    n_edges = sum(isfinite, adjcosts)
    if isnothing(n_upgrades)
        n_upgrades = ceil(Int, n_edges * frac_upgrades)
    end

    TravelNetwork(coords, pops, adjcosts, n_upgrades)
end

function get_upgraded_network(edges_to_upgrade, net::TravelNetwork; cost_ratio = 0.75)
    (; edges, adjcosts) = net

    adjcosts_new = copy(adjcosts)
    for i in eachindex(edges_to_upgrade)
        e = edges_to_upgrade[i]
        if e == true
            ind = CartesianIndex(edges[i])
            i, j = Tuple(ind)
            adjcosts_new[i, j] = adjcosts[i, j] * cost_ratio
            adjcosts_new[j, i] = adjcosts[j, i] * cost_ratio
        end
    end
    new_net = @set net.adjcosts = adjcosts_new
    new_net
end

get_initial_upgrade(net) = get_initial_upgrade(default_rng(), net)
function get_initial_upgrade(rng, net::TravelNetwork)
    (; edges, n_upgrades) = net
    n = length(edges)
    edges_to_upgrade = fill(false, n)
    inds_to_upgrade = sample(rng, 1:n, n_upgrades; replace = false)
    edges_to_upgrade[inds_to_upgrade] .= true
    edges_to_upgrade
end

function get_welfare(net::TravelNetwork)
    (; adjcosts, flows) = net
    τ = pairwisecost(adjcosts)

    # average costs, weighted by population flows
    # Recall that flows are exogenous in this set-up.
    return (-1) * sum(flows .* τ) / sum(flows)
end

function get_welfare(net::TravelNetwork, edges_to_upgrade)
    new_net = get_upgraded_network(edges_to_upgrade, net)
    get_welfare(new_net)
end

lchoose(a,b) = SpecialFunctions.logabsbinomial(a,b)[1]

function get_log_n(net)
    (; n_upgrades, edges) = net
    lchoose(length(edges), n_upgrades)
end

function plot_sol(net, out; ind = nothing)
    if ind == nothing
        average_policy = get_average_policy(out)
    else
        average_policy = get_average_policy(out; ind)
    end
    plot_network(net; average_edges_to_upgrade = average_policy)
end

using HighDimensionalOptimalPolicies.Pigeons

function run_solver(solver, net; n_edges_to_swap = nothing, kwargs...)
    initfun, objfun, nextfun = let net = net
        initfun = rng -> get_initial_upgrade(rng, net)
        objfun = edges_to_upgrade -> begin
            get_welfare(net, edges_to_upgrade)
        end
        nextfun = (rng, edges_to_upgrade) -> swap_edges_to_upgrade(rng, edges_to_upgrade, n_edges_to_swap)
        initfun, objfun, nextfun
    end

    out = get_best_policy(solver; initfun, objfun, nextfun, kwargs...)

    (; net, out)
end

function test_travel_network(solver;
    square = false,
    n_coords = 10,
    diagonal_path = false,
    unequal_pops = false,
    frac_upgrades = 0.2,
    kwargs...)

    net = if square == true
            square_travel_network(n_coords; diagonal_path, unequal_pops, frac_upgrades)
        else
            random_travel_network(n_coords; diagonal_path, unequal_pops, frac_upgrades)
    end

    test_travel_network(solver, net; kwargs...)
end

function test_travel_network(solver, net;
    max_invtemp = nothing,
    invtemps_curvature = nothing,
    invtemps = nothing,
    n_inner_rounds = 1024,
    n_invtemps = 10,
    n_edges_to_swap = 1,
    display_plot = false,
    kwargs...)

    (;net, out) = run_solver(solver, net;
        max_invtemp,
        invtemps_curvature,
        invtemps,
        n_inner_rounds,
        n_invtemps,
        n_edges_to_swap,
        kwargs...)

    average_policy = get_average_policy(out)
    p = plot_network(net; average_edges_to_upgrade = average_policy)
    if display_plot == true
        display(p)
    end

    (; net, out)
end

function plot_average_policy(net, out)
    average_policy = get_average_policy(out)
    p = plot_network(net; average_edges_to_upgrade = average_policy)
end

function plot_n_policies(net, out; n_policies = 6)
    policy_vec = get_policy_vec(out)
    policy_vec_sub = sample(policy_vec, n_policies)
    plot_network(net; edges_to_upgrade_vec = policy_vec_sub)
end

function plot_objective_time(out; ind = 1, last_half = false)
    objs = get_objective_vec(out; ind)
    if last_half == true
        inds = eachindex(objs)[floor(Int, length(objs) / 2):end]
        objs = objs[inds]
    else
        inds = eachindex(objs)
    end
    plot(inds, objs,
        label = false,
        color = "black",
        xlab = "Iteratution",
        ylab = "Objective")
end

function plot_mixing_stats(net, out; inds = nothing)
    invtemps = get_invtemps(out)
    log_n = get_log_n(net)
    mixing_stats = map(eachindex(invtemps)) do ind
        test_mixing(out, log_n; ind)
    end

    if !isnothing(inds)
        invtemps = invtemps[inds]
        mixing_stats = mixing_stats[inds]
    end

    plot(invtemps, mixing_stats,
        color = "black",
        xlab = "Inverse temperature",
        ylab = "Mixing t-statistic",
        label = "")
end

function plot_all_objectives(out)
    p = plot()
    invtemps = get_invtemps(out)

    rng = Random.default_rng()
    x0 =  out.input.initfun(rng)
    random_objs = map(1:1000) do _
        x0 = out.input.nextfun(rng, x0)
        out.input.objfun(x0)
    end

    for ind in 1:(length(invtemps))
        obj = get_objective_vec(out; ind)
        label = invtemps[ind]
        density!(p, obj; alpha = 0.5, label = label, line_z = invtemps[ind], palette = :thermal)
    end
  #  density!(p, log10.(-random_objs); alpha = 0.5, label = "Random", xscale = :log10)
    p
end

function optimalbusnetwork()
    n_coords = 5^2
    diagonal_path = false
    unequal_pops = true
    net = square_travel_network(n_coords;
      diagonal_path = false,
      unequal_pops = true,
      frac_upgrades = 0.2)
    plot_network(net)

    (_, out) = test_travel_network(
      PTMCMCSolver(),
      net;
      max_invtemp = 50,
      invtemps_curvature = 2.0,
      n_inner_rounds = 10^3,
      n_invtemps = 10,
      n_swap_rounds = 100,
      n_edges_to_swap = 1)
    (; net, out)
end


