"""
Parallel Tempering with Pigeons.jl
"""
struct PigeonsSolver <: AbstractPolicySolver end

"""
Parallel Tempering with Pigeons.jl using child MPI processes
"""
struct PigeonsMPISolver <: AbstractPolicySolver end

"""
Output for a single inverse temperature with Parallel Tempering
using Pigeons.
"""
struct PigeonsSolverOutput{PV, OV, I} <: AbstractPolicyOutput
    params::PV
    objs::OV
    invtemp::I
end

function Base.show(io::IO, t::PigeonsSolverOutput)
    # TODO: Understand if this is the right correction
    final_obj = last(last(t.objs))
    s = """
    PigeonsSolverOutput
        Inverse Temperature: $(t.invtemp)
        Final objective value: $(final_obj)
        Sample length: $(length(t.objs))
    """
    print(io, s)
end

"""
Output for all inverse temperatures with Parallel Tempering using
Pigeons.
"""
struct MultiPigeonsSolverOutput{V, I, PT} <: AbstractMultiPolicyOutput
    v::V
    input::I
    pt::PT
end

function get_invtemps(t::MultiPigeonsSolverOutput)
    [vi.invtemp for vi in t.v]
end

"""
    MultiPigeonsSolverOutput(input, pt)

Given an input (which is simple) and a Pigeons output (which is very
large and complicated), arrange the parameter guesses and states
into an easier-to-use format.

!!! note
    Current implementation calls `objfun` again for *every* policy
    guess. I cannot figure out how to store the non-modified objective
    functions when we run `step!`. Also, when I divide by the inverse
    temperatures, I get odd results.

    However this is very costly, prohibitively so for very long-
    running jobs. We should change this.
"""
function MultiPigeonsSolverOutput(input, pt)
    invtemps = input.invtemps

    sample = Chains(pt)

    # PT stores the chains with the highest temperature
    # last, but we want to store it with the highest
    # temperature first
    v = map(1:length(invtemps)) do ind
        ind_reversed = (length(invtemps) - ind + 1)
        invtemp = invtemps[ind]

        init_guess = input.initfun(Random.default_rng())
        init_type = typeof(init_guess)

        chain = sample[:, :, ind_reversed]

        nt_params = get(chain, section = :parameters)
        num_itrs = length(first(nt_params))
        num_cols = length(nt_params)

        params = map(1:num_itrs) do i
            init_type([nt_params[j][i] for j in 1:num_cols])
        end


        objs = input.objfun.(params)

        PigeonsSolverOutput(params, objs, invtemp)
    end


    MultiPigeonsSolverOutput(v, input, pt)
end

function Base.show(io::IO, t::MultiPigeonsSolverOutput)
    max_temp = first(t.input.invtemps)
    num_temps = length(t.v)
    s = """
    MultiPigeonsSolverOutput with $num_temps temperatures and maximum temperature $max_temp
    """
    print(io, s)
end

"""
Run the pigeons solver.
"""
function _get_best_policy(::PigeonsSolver; initfun, nextfun, objfun, max_invtemp, n_inner_rounds, n_invtemps,  kwargs...)
    H = HighDimensionalProblem(initfun, objfun, nextfun)

    pt = pigeons(
        target = OuterHighDimensionalProblem(max_invtemp, H),
        reference = OuterHighDimensionalProblem(0.0, H),
        record = [traces],
        extended_traces = true,
        n_chains = n_invtemps,
        # n_rounds is given as an exponent in pigeons.
        n_rounds = ceil(Int, log2(n_inner_rounds)),
        explorer = H,
        multithreaded = false,
        show_report = false,
        kwargs...)

    invtemps = reverse(pt.shared.tempering.schedule.grids .* max_invtemp)

    input = GenericSolverInput(initfun, nextfun, objfun, invtemps)

    MultiPigeonsSolverOutput(input, pt)
end

"""
    get_best_policy(
        s::PigeonsSolver;
        initfun,
        nextfun,
        objfun,
        max_invtemp = nothing,
        invtemps_curvature = nothing,
        n_invtemps = 10,
        invtemps = nothing,
        n_inner_rounds = 1024)

* `initfun`: `initfun(rng)` must return an initial guess
* `nextfun`: `nextfun(rng, state)` must return the next guess,
  conditional on the current state.
* `objfun`: `objfun(state)` must return the welfare of a given state
  (where higher welfare is better)
* `max_invtemp`: The maximum inverse temperature
* `n_invtemps`: The number of inverse temperatures.
* `n_inner_rounds`: The number of Metropolis-Hastings steps per
  inverse temperature. **Note:** This is the *total* number of
  states that will be returned.
* `kwargs...`: Passed to `Pigeons.pigeons`, see for details.

## Additional notes

Pigeons.jl does not permit options for either the inverse temperature
curvature or a vector of inverse temperatures. The algorithm in
Syed et al. (2021) creates an "optimal annealing schedule", meaning
it choosees the inverse temperatures to generate an optimal swap
rate beween inverse temperatures.

The optimal annealing schedule is is iteratively set after 2, 4,
8, 16, etc. iterations, where the final half of the iterations are
conducted under a fixed, presumably optimal, annealing schedule.

Additionaly, Pigeons.jl does not allow for for specifying the
number of swap rounds. As implemented, Pigeons.jl performs a swap
step after only a couple Metropolis-Hastings steps, so swaps
happen very frequently.

See `?HighDimensionalOptimalPolicies.Pigeons.Inputs` for more
details
"""
function get_best_policy(
    s::PigeonsSolver;
    initfun,
    nextfun,
    objfun,
    max_invtemp::Union{Nothing, Real} = nothing,
    n_invtemps::Integer = 10,
    n_inner_rounds = 1024,
    invtemps = nothing, # Just to handle errors easier
    invtemps_curvature = nothing, # Just to handle errors easier
    kwargs...)

    if !isnothing(invtemps) || !isnothing(invtemps_curvature)
        throw(ArgumentError("PigeonsSolver only supports passing a single maximum inverse temperature (max_invtemp)"))
    end

    _get_best_policy(s; initfun, objfun, nextfun, max_invtemp, n_inner_rounds, n_invtemps, kwargs...)
end

"""
Get the best policy using Pigeons.jl with a distributed solver.
"""
function _get_best_policy(::PigeonsMPISolver; initfun, objfun, nextfun, max_invtemp, childprocess, n_invtemps, n_inner_rounds, kwargs...)
    H = HighDimensionalProblem(initfun, objfun, nextfun)

    pt = pigeons(
        target = OuterHighDimensionalProblem(max_invtemp, H),
        reference = OuterHighDimensionalProblem(0, H),
        record = [traces],
        extended_traces = true,
        n_chains = n_invtemps,
        n_rounds = ceil(Int, log2(n_inner_rounds)),
        explorer = H,
        multithreaded = true,
        checkpoint = true,
        show_report = false,
        on = childprocess,
        kwargs...)

    pt = Pigeons.load(pt)

    invtemps = reverse(pt.shared.tempering.schedule.grids .* max_invtemp)

    input = GenericSolverInput(initfun, nextfun, objfun, invtemps)

    MultiPigeonsSolverOutput(input, pt)
end

"""
    function get_best_policy(
        s::PigeonsMPISolver;
        initfun,
        objfun,
        nextfun,
        max_invtemp = nothing,
        invtemps_curvature = nothing,
        invtemps = nothing,
        n_inner_rounds = 1024,
        n_chains = 10,
        n_local_mpi_processes = 2,
        n_threads = 2,
        dependencies = [],
        kwargs...)

Pigeons.jl Parallel Tempering algorithm solver parallelized through
a child MPI process.

!!! warning
    Use of Child MPI Processes does not work well with Revise.jl.

* `n_local_mpi_processes`: The number of local MPI processes to start
* `n_threads`: The number of threads on each MPI process
* `dependencies`: Additional modules to pass to each new MPI process
  required for `initfun`, `objfun`, and `nextfun` to work.

See `? HighDimensionalOptimalPolicies.Pigeons.ChildProcess` for
details.
"""
function get_best_policy(
    s::PigeonsMPISolver;
    initfun,
    objfun,
    nextfun,
    max_invtemp = nothing,
    n_invtemps::Integer = 10,
    invtemps_curvature = nothing, # For handling
    invtemps = nothing, # For error handling
    n_inner_rounds::Integer = 1024,
    n_chains::Integer = 10,
    n_local_mpi_processes = 2,
    n_threads = 2,
    dependencies = [],
    kwargs...)

    if !isnothing(invtemps) || !isnothing(invtemps_curvature)
        throw(ArgumentError("PigeonsMPISolver only supports passing a single maximum inverse temperature (max_invtemp)"))
    end

    childprocess =  ChildProcess(
        n_local_mpi_processes = 2,
        n_threads = 2,
        dependencies = vcat(dependencies, [HighDimensionalOptimalPolicies])
    )

    _get_best_policy(s; initfun, objfun, nextfun, max_invtemp, childprocess, n_inner_rounds, n_invtemps, kwargs...)
end

function get_policy_vec(out::PigeonsSolverOutput; only_last_half = nothing)
    if !isnothing(only_last_half)
        s = "only_last_half is ignored in with PigeonsSolverOutput" *
        "because Pigeons.jl already uses a burn-in period."
        @warn s
    end
    out.params
end

function get_average_policy(out::PigeonsSolverOutput; only_last_half = nothing)
    if !isnothing(only_last_half)
        s = "only_last_half is ignored in with PigeonsSolverOutput" *
        "because Pigeons.jl already uses a burn-in period."
        @warn s
    end
    mean(get_policy_vec(out))
end

function get_last_policy(out::PigeonsSolverOutput)
    last(get_policy_vec(out))
end

# TODO: Understand this better
function get_objective_vec(out::PigeonsSolverOutput; only_last_half = true)
    if !isnothing(only_last_half)
        s = "only_last_half is ignored in with PigeonsSolverOutput" *
        "because Pigeons.jl already uses a burn-in period."
        @warn s
    end
    out.objs
end
