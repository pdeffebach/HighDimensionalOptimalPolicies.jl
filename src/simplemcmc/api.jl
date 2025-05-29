struct SimpleMCMCSolver <: AbstractPolicySolver end

struct SimpleMCMCSolverOutput{VP, VO, I} <: AbstractPolicyOutput
    params::VP
    objs::VO
    invtemp::I
end

function Base.show(io::IO, ::MIME"text/plain", t::SimpleMCMCSolverOutput)
    s = """
    SimpleMCMCSolverOutput
        Inverse Temperature: $(t.invtemp)
        Final objective value: $(last(t.objs))
        Sample length: $(length(t.objs))
    """
    print(io, s)
end

function Base.show(io::IO, t::SimpleMCMCSolverOutput)
    s = "SimpleMCMCSolverOutput (invtemp = $(t.invtemp))"
    print(io, s)
end

struct MultiSimpleMCMCSolverOutput{V, I}
    v::V
    input::I
end

function get_invtemps(t::MultiSimpleMCMCSolverOutput)
    [vi.invtemp for vi in t.v]
end

function Base.show(io::IO, t::MultiSimpleMCMCSolverOutput)
    max_temp = first(t.input.invtemps)
    num_temps = length(t.v)
    s = """
    MultiSimpleMCMCSolverOutput with $num_temps temperatures and maximum temperature $max_temp
    """
    print(io, s)
end

function simplemcmc_inner(rng, initfun, nextfun, objfun, invtemp; n_inner_rounds = 1000)
    x0  = initfun(rng)
    y0 = objfun(x0)

    xs_out = Vector{typeof(x0)}(undef, n_inner_rounds)
    ys_out = Vector{typeof(y0)}(undef, n_inner_rounds)

    for i in 1:n_inner_rounds
        xs_out[i] = x0
        ys_out[i] = y0
        x1 = nextfun(rng, x0)
        y1 = objfun(x1)
        α = exp(invtemp * (y1 - y0))
        if α >= 1
            x0 = x1
            y0 = y1
        elseif rand(rng) < α
            x0 = x1
            y0 = y1
        else
            # Do nothing
        end
    end
    return (params = last_half(xs_out), objs = last_half(ys_out))
end

function _get_best_policy(s::SimpleMCMCSolver; initfun, nextfun, objfun, invtemps, n_inner_rounds, n_chains)
    input = GenericSolverInput(initfun, nextfun, objfun, invtemps)

    rng = Random.default_rng()

    v = pmap(invtemps) do invtemp
        (; params, objs) = simplemcmc_inner(rng, initfun, nextfun, objfun, invtemp; n_inner_rounds)
        SimpleMCMCSolverOutput(params, objs, invtemp)
    end

    MultiSimpleMCMCSolverOutput(v, input)
end

function get_best_policy(
    s::SimpleMCMCSolver;
    initfun,
    objfun,
    nextfun,
    max_invtemp = nothing,
    invtemps_curvature = nothing,
    invtemps = nothing,
    n_inner_rounds = 1024,
    n_chains = 10)

    if isnothing(max_invtemp) && isnothing(invtemps_curvature)
        if isnothing(invtemps)
            throw(ArgumentError("Need to provide either max_invtemp and invtemps_curvature or invtemps"))
        end
    else
        invtemps = make_invtemps(max_invtemp; invtemps_curvature, length = n_chains)
    end
    _get_best_policy(s; initfun, objfun, nextfun, invtemps, n_inner_rounds, n_chains)
end

# Accessors for the output ###########################################
# TODO: Make this less verbose

function get_policy_vec(out::SimpleMCMCSolverOutput)
    out.params
end

function get_average_policy(out::SimpleMCMCSolverOutput)
    mean(get_policy_vec(out))
end

function get_last_policy(out::SimpleMCMCSolverOutput)
    last(get_policy_vec(out))
end

function get_objective_vec(out::SimpleMCMCSolverOutput)
    # No need for normalizing by the inverse temperature
    # with this implementation
    out.objs
end

function test_mixing(out::SimpleMCMCSolverOutput, log_n; K = nothing)
    ξ = out.invtemp
    obj_vec = get_objective_vec(out)
    if !isnothing(K)
        obj_vec = StatsBase.sample(obj_vec, K)
    end
    obj_max = maximum(obj_vec)
    obj_mean = mean(obj_vec)
    obj_std = std(obj_vec)

    T̂ = obj_max - obj_mean - (log_n / ξ)
    z = T̂ / obj_std
#    p = cdf(Normal(0, 1), z)
end

function get_policy_vec(out::MultiSimpleMCMCSolverOutput; ind = 1)
    get_policy_vec(out.v[ind])
end

function get_average_policy(out::MultiSimpleMCMCSolverOutput; ind = 1)
    get_average_policy(out.v[ind])
end

function get_last_policy(out::MultiSimpleMCMCSolverOutput; ind = 1)
    get_last_policy(out.v[ind])
end

function get_objective_vec(out::MultiSimpleMCMCSolverOutput; ind = 1)
    get_objective_vec(out.v[ind])
end

function test_mixing(out::MultiSimpleMCMCSolverOutput, log_n; K = nothing, ind = 1)
    test_mixing(out.v[ind], log_n; K)
end

######################################################################