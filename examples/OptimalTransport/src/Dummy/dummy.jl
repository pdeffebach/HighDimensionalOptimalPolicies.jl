struct BestPolicies{V, I}
    x::V
    num_upgrades::I
end

function get_log_n(bp::BestPolicies)
    num_edges = length(first(bp.x))
    num_upgrades = bp.num_upgrades
    SpecialFunctions.logabsbinomial(num_edges, num_upgrades)[1]
end

function Base.show(io::IO, bp::BestPolicies)
    num_optimal = length(bp.x)
    num_edges = length(first(bp.x))
    num_upgrades = bp.num_upgrades
    s = """
    BestPolicies with $num_optimal candidates with $num_edges
    edges and $num_upgrades upgrades.
    """
    print(io, s)
end

function make_best_policies(num_edges, num_upgrades, num_optimal)
    x = map(1:num_optimal) do _
        #inds = sample(1:num_edges, num_upgrades; replace = false)
        #t = fill(false, num_edges)
        #t[inds] .= true
        #t
        t = rand(LogNormal(2.0), num_edges)
        t = t ./ norm(t)
    end
    BestPolicies(x, num_upgrades)
end

function get_welfare(bp::BestPolicies, t)
    ds = map(bp.x) do policy
        dot(policy, t)
    end
    maximum(ds)
end

function plot_output(bp, out)
    vs = get_policy_vec(out; ind = 1)
    v = mean(last_half(vs))
    v = sort(v; lt = >)
    plot(range(0, 1; length = length(v)), v,
        label = "",
        color = "black",
        xlab = "Edge",
        ylab = "Fraction upgraded")
end

function get_optimal_policies(num_edges, num_upgrades, num_optimal)
    bp = make_best_policies(num_edges, num_upgrades, num_optimal)

    initfun = let num_edges = num_edges, num_upgrades = num_upgrades
        rng -> begin
            inds = sample(rng, 1:num_edges, num_upgrades; replace = false)
            t = fill(false, num_edges)
            t[inds] .= true
            t
        end
    end

    nextfun = (rng, t) -> begin
        swap_edges_to_upgrade(rng, t, 5)
    end

    objfun = let bp = bp
        t -> get_welfare(bp, t)
    end

    out = get_best_policy(
        PTMCMCSolver();
        initfun,
        nextfun,
        objfun,
        max_invtemp = 10.0,
        invtemps_curvature = 2.0,
        n_inner_rounds = 10^4,
        n_invtemps = 10)

    (; bp, out)
end

function dummy()
    num_edges = 100
    num_upgrades = 10
    num_optimal = 10

    get_optimal_policies(num_edges, num_upgrades, num_optimal)
end