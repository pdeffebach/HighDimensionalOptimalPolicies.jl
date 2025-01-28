module OptimalTransport

using UnPack

using Plots

using LinearAlgebra, StatsBase

using HighDimensionalOptimalPolicies

include("turing.jl")

include("abstractchains.jl")

"""
A travel network throughout the city. A weighted graph
in which nodes are locations throughout the city and
weights represent travel times between locations.

Let N be the total number of nodes and E be the total number
of edges

  * `nside`: Number of sides in square travel network
  * `adj`: An ``N \\times N`` Adjacency matrix of nodes
  * `adj_dict`: A dictionary summarizing the adjacency matrix
  * `edges`: An ``E \\times 2`` matrix where each row is an
    columns represent the start and end note of that edge.
  * `coords`: An ``N \\times 2`` matrix where each row is a
    node and each row represents an x and y coordinate of that
    node. x and y coordinates are integers, for simplicity.
  * `adj_cost`: A ``N \\times N`` matrix representing travel
    costs between nodes. Travel costs are `Inf` if direct travel
    is impossible.
  * `adj_cost_original`: Original adjustment cost matrix, before
    updating.
  * `pop`: A length ``N`` vector representing the population residing
    at each node.
  * `pop_sq`: A matrix representation of `pop`. This is useful when
    nodes are arrayed in a grid and we want to visualize populations.
  * `flows`: An ``N \\times N`` matrix of flows between nodes.
"""
mutable struct TravelNetwork
    nside::Int64

    locs::Matrix{Int64}
    adj::Matrix{Int64}
    adj_dict::Dict{Int64, Vector{Int64}}
    edges::Matrix{Int64}
    coords::Matrix{Int64}

    adj_cost::Matrix{Float64}
    adj_cost_original::Matrix{Float64}

    pop::Vector{Float64}
    pop_sq::Matrix{Float64}
    flows::Matrix{Float64}

    TravelNetwork() = new()
end

function add_edge!(edges_dict, (v1, v2))
    if haskey(edges_dict, v1)
        push!(edges_dict[v1], v2)
    else
        edges_dict[v1] = [v2]
    end
end

"""
    init_env_square_nodiag(nside)

Initialize the environment of a square transportation
network with `nside` nodes along each row.
"""
function init_env_square_nodiag(nside::Int)
    locs = reshape(collect(1:nside^2), (nside, nside))

    # Define adjacency matrix
    adj = zeros(Float64, nside^2, nside^2)
    adj_dict = Dict{Int64, Vector{Int64}}()
    edges = zeros(Int64, 2, 0)


    # Add horizontal edges
    # For any node, except that on the furthest right,
    # draw an edge connecting it to the node on the right.
    for irow=1:nside
        for icol=1:(nside-1)
            v1 = locs[irow, icol]
            v2 = locs[irow, icol+1]

            adj[v1, v2] = adj[v2, v1] = 1.0

            add_edge!(adj_dict, (v1, v2))
            add_edge!(adj_dict, (v2, v1))

            edges = hcat(edges, [v1, v2])
        end
    end

    # Add vertical edges
    # For any node, except the node on the bottom, draw
    # an edge connecting it to the node below it.
    for irow=1:(nside-1)
        for icol=1:nside
            v1 = locs[irow, icol]
            v2 = locs[irow+1, icol]

            adj[v1, v2] = adj[v2, v1] = 1.0

            add_edge!(adj_dict, (v1, v2))
            add_edge!(adj_dict, (v2, v1))

            edges = hcat(edges, [v1, v2])
        end
    end

    # Add coordinates for plotting
    coords = zeros(Float64, nside^2, 2)
    for irow=1:nside
        for icol=1:nside
            v1 = locs[irow, icol]
            coords[v1, :] .= (irow, icol)
        end
    end

    # Add adjacenty cost
    # The cost between any two nodes is simply 2
    adj_cost = adj .* 2.0
    adj_cost[adj_cost .== 0] .= Inf


    # Construct the TravelNetwork and fill it in
    mynet = TravelNetwork()
    mynet.nside = nside
    mynet.locs = locs
    mynet.adj = adj
    mynet.adj_dict = adj_dict
    mynet.edges = Matrix(transpose(edges))
    mynet.coords = coords
    mynet.adj_cost = adj_cost
    mynet.adj_cost_original = copy(adj_cost)

    return mynet
    # return locs, adj, adj_dict, Matrix(transpose(edges)), coords, adj_cost
end


"""
    compute_costs(adj_cost::Matrix{Float64}; θ::Float64=8.0)

Compute the expected cost of travelling between two nodes using the
Allen and Arkolakis (2022) method. See Equation 4. On page 15 they
describe how Equation 4 can be re-written in matrix notation.

This is a convenient closed form solution which is the solution
to an optimal routing problem.

`θ` represents the shape parameter of the Frechet distribution
of idiosyncratic shocks along routes.
"""
function compute_costs(adj_cost::Matrix; θ = 8.0)

    B = inv(I - Symmetric(adj_cost) .^(-θ))

    τ = B .^(-1/θ)

    return τ
end

"""
    init_pop_flows(mynet::TravelNetwork; β=1.0)

Assign residential population and flows to the transportation
network. This has nothing to do withthe optimal transport
problem itself, and will be considered exogenous in the
optimization problem.

We give more population to areas far from the city center.

`β` represents dispersion in the Frechet parameter for commuting
between any two locations.

I don't know what the `-0.5` means here.
"""
function add_pop_flows!(mynet::TravelNetwork; β=1.0)
    @unpack nside = mynet

    # Define the population living at each node.
    row_mat = repeat(vec(1:nside), outer=(1, nside))
    col_mat = Matrix(transpose(row_mat))

    pop_sq = abs.(row_mat .+ col_mat .- nside .- 1.0) .^ 1.25
    pop = reshape(pop_sq, nside^2)

    ### Define flows
    # first, compute costs based on baseline
    τ = compute_costs(mynet.adj_cost_original)

    flows = zeros(Float64, nside^2, nside^2)

    for i=1:(nside*nside)
        costs_to_other_nodes = τ[i, :]
        πs = (costs_to_other_nodes .^ (-0.5) .* pop) .^ (β)
        πs ./= sum(πs)
        flows[i, :] = pop[i] * πs
    end

    # display(reshape(flows[1, :], (nside, nside)))

    mynet.pop_sq = pop_sq
    mynet.pop = pop
    mynet.flows = flows
end

function plot_network(
    mynet::TravelNetwork;
    plot_numbers=true,
    plot_pop=true,
    title="",
    filepath="",
    min_width=1.0,
    max_width=6.0,
    max_marker_size=15.0)

    nside, adj_dict, adj_cost, coords, pop = mynet.nside, mynet.adj_dict, mynet.adj_cost, mynet.coords, mynet.pop

    # scatter(coords[:, 1], coords[:, 2], color=:black, label="")

    plot(title=title)

    all_costs = []
    for v1=keys(adj_dict)
        for v2=adj_dict[v1]
            push!(all_costs, adj_cost[v1, v2])
        end
    end
    min_cost = minimum(all_costs)
    max_cost = maximum(all_costs)
    if min_cost == max_cost
        max_cost = 1.1 * min_cost
    end
    # println(min_cost)
    # println(max_cost)

    ### EDGES
    for v1=keys(adj_dict)
        for v2=adj_dict[v1]
            c1 = coords[v1, :]
            c2 = coords[v2, :]

            mylinewidth = - (adj_cost[v1, v2] - min_cost)/(max_cost - min_cost) * (max_width - min_width) + max_width

            plot!([c1[1], c2[1]], [c1[2], c2[2]], label="", color="gray", alpha=0.5, linewidth=mylinewidth)
        end
    end

    marker_sizes = pop ./ maximum(pop) .* max_marker_size

    if plot_numbers && plot_pop
        scatter!(coords[:, 1], coords[:, 2], markersize=marker_sizes, color=:blue,
                series_annotations=text.(1:nside^2, :top, :red), label="",
                xaxis=false, yaxis=false, yticks=[], xticks=[]) |> display
    elseif plot_pop
        scatter!(coords[:, 1], coords[:, 2], markersize=marker_sizes, color=:blue, label="",
                    xaxis=false, yaxis=false, yticks=[], xticks=[]) |> display
    else
        plot!(xaxis=false, yaxis=false, yticks=[], xticks=[]) |> display
    end

    if filepath != ""
        png(filepath)
    end

    return
end

"""
    upgrade_highways!(mynet::TravelNetwork, edges_to_upgrade; cost_ratio=0.75)

Lower the cost of travel between two nodes.
"""
function upgrade_highways!(mynet::TravelNetwork, edges_to_upgrade; cost_ratio=0.75)

    mynet.adj_cost .= mynet.adj_cost_original

    # adj_cost, edges = mynet.adj_cost, mynet.edges

    for i=1:length(edges_to_upgrade)
        if edges_to_upgrade[i] == true
            v1 = mynet.edges[i, 1]
            v2 = mynet.edges[i, 2]

            mynet.adj_cost[v1, v2] *= cost_ratio
            mynet.adj_cost[v2, v1] *= cost_ratio
        end
    end

    @assert all(mynet.adj_cost .> 1.0)
end

"""
    reset_highways!(mynet)

Resets the transportation cost between edges to their
original values.

Used for debugging.
"""
function reset_highways!(mynet)
    mynet.adj_cost .= mynet.adj_cost_original

    nothing
end

"""
    travel_welfare(edges_to_upgrade, mynet)

The welfare of a given transportation network, accounting for
upgrades.
"""
function travel_welfare(edges_to_upgrade, mynet)

    upgrade_highways!(mynet, edges_to_upgrade; cost_ratio=0.75)

    τ = compute_costs(mynet.adj_cost)

    # average costs, weighted by population flows
    # Recall that flows are exogenous in this set-up.
    return (-1) * sum(mynet.flows .* τ) / sum(mynet.flows)
end


"""
    swap_edges_to_upgrade(edges_to_upgrade)


"""
function swap_edges_to_upgrade(edges_to_upgrade)
    improved_edges = findall(edges_to_upgrade)
    unimproved_edges = findall(edges_to_upgrade .== false)

    edge_to_drop = sample(improved_edges)
    edge_to_add = sample(unimproved_edges)

    new_edges_to_upgrade = copy(edges_to_upgrade)
    new_edges_to_upgrade[edge_to_drop] = false
    new_edges_to_upgrade[edge_to_add] = true

    return new_edges_to_upgrade
end

# Get ready for the Metropolitan Hastings algorithm
function prepare_metropolitan_hastings(mynet)
    obj_fun = let mynet = mynet
        edges_to_upgrade -> travel_welfare(edges_to_upgrade, mynet)
    end

    next_policy = swap_edges_to_upgrade

    β = 100.0

    mhp = HighDimensionalOptimalPolicies.MetropolitanHastingsProblem(
        obj_fun, next_policy, β)
end

function main()
    nside = 5
    mynet_init = init_env_square_nodiag(nside)
    add_pop_flows!(mynet_init)

    mhp = prepare_metropolitan_hastings(deepcopy(mynet_init))
    K = 5
    num_edges = size(mynet_init.edges, 1)
    edges_to_upgrade_num = sample(1:num_edges, K; replace = false)
    edges_to_upgrade_init = collect(1:num_edges) .∈ Ref(edges_to_upgrade_num)

    policy_best = HighDimensionalOptimalPolicies.solve(edges_to_upgrade_init, mhp; max_itr = 1000)

    mynet_best = deepcopy(mynet_init)
    upgrade_highways!(mynet_best, policy_best)

    mynet_best
end



end # module OptimalTransport
