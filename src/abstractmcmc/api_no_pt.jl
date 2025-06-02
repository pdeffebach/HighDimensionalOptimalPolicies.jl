
"""
Separate independent Metropolis-Hastings algorithms for
each inverse temperature.
"""
struct MCMCSolver <: AbstractPolicySolver end

"""
Conventional Simulated Annealing algorithm which systematically
increases inverse temperature in a single run.
"""
struct SimulatedAnnealingSolver <: AbstractPolicySolver end

"""
Many independent Simulated Annealing algorithm runs in parallel,
and whose outputs reflect these independent runs.
"""
struct IndependentSimulatedAnnealingSolver <: AbstractPolicySolver end

"""
Contains ouput from the MCMCSolver algorithm for a single
inverse temperature.

$FIELDS
"""
struct MCMCSolverOutput{VP, VO, I} <: AbstractPolicyOutput
    params::VP
    objs::VO
    invtemp::I
end

"""
Contains ouput from the Independent Simulated Annealing algorithm for a
single inverse temperature.

$FIELDS
"""
struct IndependentSimulatedAnnealingSolverOutput{VP, VO, I} <: AbstractPolicyOutput
    params::VP
    objs::VO
    invtemp::I
end

function Base.show(io::IO, ::MIME"text/plain", t::MCMCSolverOutput)
    s = """
    MCMCSolverOutput
        Inverse Temperature: $(t.invtemp)
        Final objective value: $(last(t.objs))
        Sample length: $(length(t.objs))
    """
    print(io, s)
end

function Base.show(io::IO, t::MCMCSolverOutput)
    s = "MCMCSolverOutput (invtemp = $(t.invtemp))"
    print(io, s)
end

function Base.show(io::IO, ::MIME"text/plain", t::IndependentSimulatedAnnealingSolverOutput)
    s = """
    IndependentSimulatedAnnealingSolverOutput
        Inverse Temperature: $(t.invtemp)
        Final objective value: $(last(t.objs))
        Sample length: $(length(t.objs))
    """
    print(io, s)
end

function Base.show(io::IO, t::IndependentSimulatedAnnealingSolverOutput)
    s = "IndependentSimulatedAnnealingSolverOutput (invtemp = $(t.invtemp))"
    print(io, s)
end

"""
Contains outputs from the MCMCSolver algorithm for multiple
inverse temperatures.
"""
struct MultiMCMCSolverOutput{V, I} <: AbstractMultiPolicyOutput
    v::V
    input::I
end

"""
    $SIGNATURES

Get the inverse temperatures for a "multi"-model, i.e. those which
store outputs for multiple inverse temperatures.
"""
function get_invtemps(t::MultiMCMCSolverOutput)
    [vi.invtemp for vi in t.v]
end

function Base.show(io::IO, t::MultiMCMCSolverOutput)
    max_invtemp = first(t.input.invtemps)
    num_temps = length(t.v)
    s = """
    MultiMCMCSolverOutput with $num_temps temperatures and maximum inverse temperature $max_invtemp
    """
    print(io, s)
end

"""
    $SIGNATURES

Run rounds of the Metropolis-Hastings algorithm.
"""
function mcmc_solver_inner(rng, sampler, objfun, invtemp; n_inner_rounds, initial_state = nothing)
    model = TemperedPolicyObjective(objfun, invtemp)

    s = sample(
        rng,
        model,
        sampler,
        n_inner_rounds,
        chain_type = MCMCChains.Chains,
        progress = false,
        initial_state = initial_state)

end

"""
    $SIGNATURES

Get optimal policies for a given high dimensional problem.
"""
function _get_best_policy(s::MCMCSolver; initfun, nextfun, objfun, invtemps, n_inner_rounds, rng)
    input = GenericSolverInput(initfun, nextfun, objfun, invtemps)

    sampler = PolicySampler(initfun, nextfun)

    v = map(invtemps) do invtemp
        chain = mcmc_solver_inner(rng, sampler, objfun, invtemp; n_inner_rounds)
        params = [x.params for x in chain]
        objs = [x.obj for x in chain]
        MCMCSolverOutput(params, objs, invtemp)
    end

    MultiMCMCSolverOutput(v, input)
end

"""
    get_best_policy(
        s::MCMCSolver;
        initfun,
        nextfun,
        objfun,
        max_invtemp = nothing,
        invtemps_curvature = nothing,
        n_invtemps = 10,
        invtemps = nothing,
        n_inner_rounds = 1024,
        rng = Random.default_rng())


Get optimal policies using `MCMCSolver`, which runs separate
Metropolis-Hastings runs for each temperature.

## Arguments

* `initfun`: `initfun(rng)` must return an initial guess
* `nextfun`: `nextfun(rng, state)` must return the next guess,
  conditional on the current state.
* `objfun`: `objfun(state)` must return the welfare of a given state
  (where higher welfare is better)
* `max_invtemp`: The maximum inverse temperature
* `invtemps_curvature`: The inverse temperature curvature, see
  [`make_invtemps`](@ref) for details.
* `n_invtemps`: The number of inverse temperatures.
* `invtemps`: An optional argument to pass a vector of inverse
  temperatures directlry. *Either* `invtemps` *or* `max_invtemp`
  and `invtemps_curvature` may be passed, but not both.
* `n_inner_rounds`: The number of Metropolis-Hastings rounds per
  inverse temperature.
* `rng`: The random number generator used for estimation
"""
function get_best_policy(
    s::MCMCSolver;
    initfun,
    nextfun,
    objfun,
    max_invtemp::Union{Nothing, Real} = nothing,
    invtemps_curvature::Union{Nothing, Real} = nothing,
    n_invtemps::Integer = 10,
    invtemps::Union{Nothing, AbstractVector{<:Real}} = nothing,
    n_inner_rounds::Integer = 1024,
    rng = Random.default_rng())

    if isnothing(max_invtemp) && isnothing(invtemps_curvature)
        if isnothing(invtemps)
            throw(ArgumentError("Need to provide either max_invtemp and invtemps_curvature or invtemps"))
        end
    else
        invtemps = make_invtemps(max_invtemp; invtemps_curvature, length = n_invtemps)
    end
    _get_best_policy(s; initfun, objfun, nextfun, invtemps, n_inner_rounds, rng)
end


"""
    $SIGNATURES

Run a simulated annealing solver with a fixed number of rounds
in between inverse temperatures.
"""
function _get_best_policy(::SimulatedAnnealingSolver; initfun, nextfun, objfun, invtemps, n_inner_rounds, rng)
    input = GenericSolverInput(initfun, nextfun, objfun, invtemps)

    sampler = PolicySampler(initfun, nextfun)

    local x
    initial_state = nothing
    outputs = map(reverse(invtemps)) do invtemp
        chain = mcmc_solver_inner(rng, sampler, objfun, invtemp; n_inner_rounds, initial_state)

        params = [x.params for x in chain]
        objs = [x.obj for x in chain]

        x = MCMCSolverOutput(params, objs, invtemp)

        initial_state = last(params)
        x
    end

    MultiMCMCSolverOutput(reverse(outputs), input)
end

"""
    get_best_policy(
        s::SimulatedAnnealingSolver;
        initfun,
        nextfun,
        objfun,
        max_invtemp = nothing,
        invtemps_curvature = nothing,
        n_invtemps = 10,
        invtemps = nothing,
        n_inner_rounds = 1024,
        rng = Random.default_rng())


Get optimal policies using `SimulatedAnnealingSolver`, which runs
successive Metropolis-Hastings runs, each time increasing
the inverse temperature until the maximum inverse temperature
is reached.

## Arguments

* `initfun`: `initfun(rng)` must return an initial guess
* `nextfun`: `nextfun(rng, state)` must return the next guess,
  conditional on the current state.
* `objfun`: `objfun(state)` must return the welfare of a given state
  (where higher welfare is better)
* `max_invtemp`: The maximum inverse temperature
* `invtemps_curvature`: The inverse temperature curvature, see
  [`make_invtemps`](@ref) for details.
* `n_invtemps`: The number of inverse temperatures.
* `invtemps`: An optional argument to pass a vector of inverse
  temperatures directlry. *Either* `invtemps` *or* `max_invtemp`
  and `invtemps_curvature` may be passed, but not both.
* `n_inner_rounds`: The number of Metropolis-Hastings rounds per
  inverse temperature.
* `rng`: The random number generator used
"""
function get_best_policy(
    s::SimulatedAnnealingSolver;
    initfun,
    nextfun,
    objfun,
    max_invtemp::Union{Nothing, Real} = nothing,
    invtemps_curvature::Union{Nothing, Real} = nothing,
    n_invtemps::Integer = 10,
    invtemps::Union{Nothing, AbstractVector{<:Real}} = nothing,
    n_inner_rounds::Integer = 1024,
    rng = Random.default_rng())

    if isnothing(max_invtemp) && isnothing(invtemps_curvature)
        if isnothing(invtemps)
            throw(ArgumentError("Need to provide either max_invtemp and invtemps_curvature or invtemps"))
        end
    else
        invtemps = make_invtemps(max_invtemp; invtemps_curvature, length = n_invtemps)
    end
    _get_best_policy(s; initfun, nextfun, objfun, invtemps, n_inner_rounds, rng)
end


"""
For a set of independent simulated annealing runs, create an output
which picks the last state and objective value from each run.
"""
function merge_many_independent_outputs(vector_multi::Vector{<:MultiMCMCSolverOutput})
    first_multi = first(vector_multi)
    input = first_multi.input
    invtemps = first_multi.input.invtemps
    new_multi = map(1:length(invtemps)) do i
        params = [last(multi.v[i].params) for multi in vector_multi]
        objs = [last(multi.v[i].objs) for multi in vector_multi]
        invtemp = invtemps[i]
        IndependentSimulatedAnnealingSolverOutput(params, objs, invtemp)
    end

    MultiMCMCSolverOutput(new_multi, input)
end

"""
    $SIGNATURES

Run many independent Simulated Annealing runs, then agglomerate
the results.
"""
function _get_best_policy(
    s::IndependentSimulatedAnnealingSolver;
    initfun,
    nextfun,
    objfun,
    invtemps,
    n_inner_rounds,
    n_independent_runs,
    rng)

    vector_multi = pmap(1:n_independent_runs) do _
        multi = _get_best_policy(SimulatedAnnealingSolver(); initfun, objfun, nextfun, invtemps, n_inner_rounds, rng)
    end
    merge_many_independent_outputs(vector_multi)
end

"""
    get_best_policy(
        s::IndependentSimulatedAnnealingSolver;
        initfun,
        nextfun,
        objfun,
        max_invtemp = nothing,
        invtemps_curvature = nothing,
        n_invtemps = 10,
        invtemps = nothing,
        n_inner_rounds = 1024,
        n_independent_runs = 50,
        rng = Random.default_rng())


Get optimal policies using `SimulatedAnnealingSolver`, which runs
successive Metropolis-Hastings runs, each time increasing
the inverse temperature until the maximum inverse temperature
is reached.

## Arguments

* `initfun`: `initfun(rng)` must return an initial guess
* `nextfun`: `nextfun(rng, state)` must return the next guess,
  conditional on the current state.
* `objfun`: `objfun(state)` must return the welfare of a given state
  (where higher welfare is better)
* `max_invtemp`: The maximum inverse temperature
* `invtemps_curvature`: The inverse temperature curvature, see
  [`make_invtemps`](@ref) for details.
* `n_invtemps`: The number of inverse temperatures.
* `invtemps`: An optional argument to pass a vector of inverse
  temperatures directlry. *Either* `invtemps` *or* `max_invtemp`
  and `invtemps_curvature` may be passed, but not both.
* `n_inner_rounds`: The number of Metropolis-Hastings rounds per
  inverse temperature.
* `n_independent_runs`: The number of independent Simulated Annealing
  algorithm runs.
* `rng`: The random number generator used.
"""
function get_best_policy(
    s::IndependentSimulatedAnnealingSolver;
    initfun,
    nextfun,
    objfun,
    max_invtemp::Union{Nothing, Real} = nothing,
    invtemps_curvature::Union{Nothing, Real} = nothing,
    n_invtemps::Union{Nothing, Integer} = 10,
    invtemps::Union{AbstractVector{<:Real}, Nothing} = nothing,
    n_inner_rounds::Integer = 1024,
    n_independent_runs::Integer = 50,
    rng = Random.default_rng())

    if isnothing(max_invtemp) && isnothing(invtemps_curvature)
        if isnothing(invtemps)
            throw(ArgumentError("Need to provide either max_invtemp and invtemps_curvature or invtemps"))
        end
    else
        invtemps = make_invtemps(max_invtemp; invtemps_curvature, length = n_invtemps)
    end
    _get_best_policy(s; initfun, nextfun, objfun, invtemps, n_inner_rounds, n_independent_runs, rng)
end

# Accessors for the output ###########################################
# TODO: Make this less verbose

function get_policy_vec(out::MCMCSolverOutput; only_last_half = true)
    if only_last_half == true
        last_half(out.params)
    else
        out.params
    end
end

function get_average_policy(out::MCMCSolverOutput; only_last_half = true)
    mean(get_policy_vec(out; only_last_half))
end

function get_last_policy(out::MCMCSolverOutput)
    last(get_policy_vec(out; last_half = false))
end

function get_objective_vec(out::MCMCSolverOutput; only_last_half = true)
    if only_last_half == true
        last_half(out.objs)
    else
        out.objs
    end
end

function get_policy_vec(out::IndependentSimulatedAnnealingSolverOutput; only_last_half = nothing)
    if !isnothing(only_last_half)
        s = "only_last_half is ignored in with IndependentSimulatedAnnealingSolverOutput" *
        "because each draw is independent."
        @warn s
    end
    if only_last_half == true
        last_half(out.params)
    else
        out.params
    end
end

function get_average_policy(out::IndependentSimulatedAnnealingSolverOutput; only_last_half = nothing)
    if !isnothing(only_last_half)
        s = "only_last_half is ignored in with IndependentSimulatedAnnealingSolverOutput" *
        "because each draw is independent."
        @warn s
    end
    mean(get_policy_vec(out; only_last_half))
end

function get_last_policy(out::IndependentSimulatedAnnealingSolverOutput)
    last(get_policy_vec(out; last_half = false))
end

function get_objective_vec(out::IndependentSimulatedAnnealingSolverOutput; only_last_half = nothing)
    if !isnothing(only_last_half)
        s = "only_last_half is ignored in with IndependentSimulatedAnnealingSolverOutput" *
        "because each draw is independent."
        @warn s
    end
    if only_last_half == true
        last_half(out.objs)
    else
        out.objs
    end
end


