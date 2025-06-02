"""
Parallel Tempering to draw optimal policies.
"""
struct PTMCMCSolver <: AbstractPolicySolver end

"""
Stores the ouput for the Parallel Tempering runs from a single
inverse temperature.
"""
struct PTMCMCSolverOutput{V, I} <: AbstractPolicyOutput
    chain::V
    invtemp::I
end

function Base.show(io::IO, ::MIME"text/plain", t::PTMCMCSolverOutput)
    s = """
    PTMCMCSolverOutput
        Inverse Temperature: $(t.invtemp)
        Final objective value: $(last(t.chain).obj)
        Sample length: $(length(t.chain))
    """
    print(io, s)
end

function Base.show(io::IO, t::PTMCMCSolverOutput)
    s = "PTMCMCSolverOutput (invtemp = $(t.invtemp))"
    print(io, s)
end

"""
Stores the output from all temperatures for a Parallel Tempering
run.
"""
struct MultiPTMCMCSolverOutput{V, I} <: AbstractMultiPolicyOutput
    v::V
    input::I
end

function get_invtemps(t::MultiPTMCMCSolverOutput)
    [vi.invtemp for vi in t.v]
end

function Base.show(io::IO, t::MultiPTMCMCSolverOutput)
    max_temp = first(t.input.invtemps)
    num_temps = length(t.v)
    s = """
    MultiPTMCMCSolverOutput with $num_temps temperatures and maximum temperature $max_temp
    """
    print(io, s)
end

"""
Stores basic information needed for the swapping stage of the
parallel tempering algorithm.
"""
struct SwapCandidate{T1, T2, T3}
    invtemp::T2
    params::T1
    obj::T3
end

"""
Run the inner Metropolis-Hastings sampling between swap attempts.
"""
function run_inner_sampling(rng, model, sampler, n_inner_rounds; initial_state = nothing)
    chainsample = sample(
        rng,
        model,
        sampler,
        n_inner_rounds;
        chain_type = MCMCChains.Chains,
        progress = false,
        initial_state = initial_state)
end

"""
For a given chain, extract the information you need to perform
the swapping.
"""
function extract_swapcandidate(model, chainsample)
    last_transition = last(chainsample)
    last_params = last_transition.params
    last_obj = last_transition.obj
    invtemp = model.invtemp

    last_true_obj = model.objfun(last_params)

    SwapCandidate(invtemp, last_params, last_true_obj)
end

"""
    $SIGNATURES

Given a vector of swap candidates, pair up the swap candidates
and assess whether their parameter states should be swapped.

We swap states according to a Metropolis-Hastings step. This
method of swapping is vanilla PT.
"""
function perform_swap(rng, swap_cands; odd_swap = true)
    params = [swap_cand.params for swap_cand in swap_cands]
    N = length(swap_cands)
    if odd_swap == true
        inds = 1:2:(N-1)
    else
        inds = 2:2:(N-1)
    end

    for i in 1:2:(length(swap_cands)-1)
       obj1 = swap_cands[i].obj
       obj2 = swap_cands[i+1].obj

        invtemp1 = swap_cands[i].invtemp
        invtemp2 = swap_cands[i+1].invtemp

        params1 = params[i]
        params2 = params[i+1]

        # Syed equation 6
        accept_ratio = exp( (invtemp1 - invtemp2) * (obj2 - obj1))
        if 1 <= accept_ratio
            params[i] = params2
            params[i+1] = params1
        elseif rand(rng) < accept_ratio
            params[i] = params2
            params[i+1] = params1
        else
            # Do nothing
        end
    end
    params
end

function _get_best_policy(::PTMCMCSolver;
    initfun,
    objfun,
    nextfun,
    invtemps,
    n_inner_rounds,
    n_swap_rounds,
    rng)

    sampler = PolicySampler(initfun, nextfun)

    n_inner_rounds_per_swap = floor(Int, n_inner_rounds / n_swap_rounds)

    input = GenericSolverInput(initfun, nextfun, objfun, invtemps)

    # Make tempered policy models for each inverse temperature
    models = map(invtemps) do invtemp
        TemperedPolicyObjective(objfun, invtemp)
    end

    aggregated_chainsamples = pmap(models) do model
        chainsample = run_inner_sampling(rng, model, sampler, n_inner_rounds_per_swap; initial_state = nothing)
    end

    swap_cands = map(models, aggregated_chainsamples) do model, chainsample
        extract_swapcandidate(model, chainsample)
    end

    # Run the swapping step using the initial states
    local params
    for swap_round in 2:n_swap_rounds
        params = perform_swap(rng, swap_cands; odd_swap = isodd(swap_round))

        chainsamples = pmap(models, params) do model, param
            chainsample = run_inner_sampling(rng, model, sampler, n_inner_rounds_per_swap; initial_state = param)
        end

        for i in eachindex(invtemps)
            append!(aggregated_chainsamples[i], chainsamples[i])
        end

        swap_cands = map(models, chainsamples) do model, chainsample
            extract_swapcandidate(model, chainsample)
        end
    end

    v = map(aggregated_chainsamples, invtemps) do aggregated_chainsample, invtemp
        PTMCMCSolverOutput(aggregated_chainsample, invtemp)
    end

    MultiPTMCMCSolverOutput(v, input)
end

"""
    get_best_policy(
        s::PTMCMCSolver;
        initfun,
        nextfun,
        objfun,
        max_invtemp = nothing,
        invtemps_curvature = nothing,
        n_invtemps = 10,
        invtemps = nothing,
        n_inner_rounds = 1024,
        n_swap_rounds = 100,
        rng = Random.default_rng())

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
* `n_inner_rounds`: The number of Metropolis-Hastings steps per
  inverse temperature. **Note:** This is the *total* number of
  states that will be returned. There length of the Metroplis-
  Hastings runs in between swap rounds is determined by
  `n_swap_rounds`, below. In other words, the number of Metropolis-
  Hastings steps in between swaps is approximately `n_inner_rounds / n_swap_rounds`.
* `n_swap_rounds`: The number of swap rounds.
* `rng`: The number of swap rounds.
"""
function get_best_policy(
    s::PTMCMCSolver;
    initfun,
    nextfun,
    objfun,
    max_invtemp::Union{Nothing, Real} = nothing,
    invtemps_curvature::Union{Nothing, Real} = nothing,
    n_invtemps::Integer = 10,
    invtemps::Union{Nothing, AbstractVector{<:Real}} = nothing,
    n_inner_rounds::Integer = 1024,
    n_swap_rounds::Integer = 100,
    rng = Random.default_rng())

    if isnothing(max_invtemp) && isnothing(invtemps_curvature)
        if isnothing(invtemps)
            throw(ArgumentError("Need to provide either max_invtemp and invtemps_curvature or invtemps"))
        end
    else
        invtemps = make_invtemps(max_invtemp; invtemps_curvature, length = n_invtemps)
    end
    _get_best_policy(s; initfun, objfun, nextfun, invtemps, n_inner_rounds, n_swap_rounds, rng)
end

# Accessors for the outputs ##########################################
# TODO: Make this less verbose
# TODO: Allocate this vector earlier on
function get_policy_vec(out::PTMCMCSolverOutput; only_last_half = true)
    chain = out.chain
    if only_last_half == true
        [t.params for t in last_half(chain)]
    else
        [t.params for t in chain]
    end
end

function get_average_policy(out::PTMCMCSolverOutput; only_last_half = true)
    mean(get_policy_vec(out; only_last_half))
end

function get_last_policy(out::PTMCMCSolverOutput)
    last(get_policy_vec(out))
end

function get_objective_vec(out::PTMCMCSolverOutput; only_last_half = true)
    chain = out.chain
    if only_last_half == true
        [t.obj for t in last_half(chain)]
    else
        [t.obj for t in chain]
    end
end

######################################################################
# Potential improvement to PT code ###################################
######################################################################
#=

This code was taken from a Julia Slack conversation about how to
do the PT algorithm while reducing the need to pass lots of data
back and forth (i.e., the whole chain for each swap).

It involves using Remote Channels.

Me:

I'm dipping my toes into parallel processing and I'm wondering what
the workflow is for saving the history of results on local processes
without sending them back and forth all the time.Let's say we have
the following pmap call

    for outer_loop in 1:5
        a_means = pmap(1:10) do i
            a = rand(100) * i
            a_mean = mean(a)
    end

Eventually I want to keep track of the a = rand(100) * i values for
each worker i and for each outer_loop. But I don't actually want to
send those vectors back to the main process. Rather, I think I want
them to live in the worker process and aggregate the vector there.
Then at the end send the whole history of vectors back to the main
process.Is this a common workflow? Are there packages that help with
this?

Response:

Sorry for the late response. If you can make it work with Dagger
that's of course much more ergonomic. But to elaborate on the
hand-rolled solution I was suggesting earlier, I think the code below
does roughly what you describe. The mean_channel eagerly collects the
pairs of outer_loop => a_mean and aggregates the values for each
outer_loop. Since it's a remote channel that lives on process 1, the
task wrapped by the channel will execute on process 1 as well.
Meanwhile the return of map collects the full data from which the
means were calculated. Those are cached on each worker for the
duration of the outer loop, and then returned to the main process.
=#

#=
using Distributed
addprocs(10)
@everywhere using Statistics

a_means = [Float64[] for _ in 1:5]

mean_channel = RemoteChannel(1) do
    Channel{Pair{Int, Float64}}(Inf) do ch
        for (outer_loop, a_mean) in ch
            @info "Aggregating for iteration $(outer_loop) on process $(myid())"
            push!(a_means[outer_loop], a_mean)
        end
    end
end

all_as = map(1:10) do i
    remotecall_fetch(i+1) do
        as = Vector{Float64}[]
        for outer_loop in 1:5
            @info "Performing computation for iteration $(outer_loop) on $(myid())"
            a = rand(100) * i
            push!(as, a)
            a_mean = mean(a)
            put!(mean_channel, outer_loop => a_mean)
        end
        return as
    end
end

stack(a_means)' == mean(stack(stack.(all_as)); dims = 1)[1, :, :] # sanity check
=#