######################################################################
# AbstractMCMC Interface #############################################
######################################################################

"""
    $SIGNATURES

Return the non-exponentiated objective value of a given state.
"""
function obj(model::TemperedPolicyObjective, params)
   model.objfun(params)
end

"""
    $SIGNATURES

Return the non-exponentiated objective value of a given state, which
has been cached in `t`.
"""
function obj(::TemperedPolicyObjective, t::AbstractPolicyTransition)
    t.obj
end

# Getting and setting parameters in the transition struct ############

"""
$SIGNATURES

Get the current parameters for the current guess.
"""
function AbstractMCMC.getparams(t::PolicyTransition)
    return t.params
end

"""
$SIGNATURES

Create a new transition struct deviating from the current one.

TODO: Understand why `t.accepted` gets passed forward here. I
guess it refers to the last guess, not the current one?
"""
function AbstractMCMC.setparams!!(
    model::AbstractPolicyObjective,
    t::PolicyTransition,
    params)

    return PolicyTransition(
        params,
        obj(model, params),
        t.accepted
    )
end

# Transitions ########################################################

"""
$SIGNATURES

Construct a transition object by calling the objective function.
"""
function transition(
    sampler::AbstractPolicySampler,
    model::AbstractPolicyObjective,
    params,
    accepted)
    objval = obj(model, params)
    return transition(sampler, model, params, objval, accepted)
end

"""
$SIGNATURES

Construct a transition object using a pre-computed objective value.
"""
function transition(
    sampler::AbstractPolicySampler,
    model::AbstractPolicyObjective,
    params,
    objval::Real,
    accepted)
    return PolicyTransition(params, objval, accepted)
end

"""
$SIGNATURES

Return the initial policy guess.
"""
function propose(
    rng::Random.AbstractRNG,
    proposal::AbstractPolicySampler,
    model::AbstractPolicyObjective
)
    return proposal(rng)
end

"""
$SIGNATURES

Returns the next policy guess, conditional on the current one.
"""
function propose(
    rng::Random.AbstractRNG,
    sampler::AbstractPolicySampler,
    model::AbstractPolicyObjective,
    transition_prev::AbstractPolicyTransition,
)
    sampler(rng, transition_prev.params)
end

# Multiple Proposals #################################################

"""
$SIGNATURES

Returns a vector of initial guesses.
"""
function propose(
    rng::Random.AbstractRNG,
    samplers::AbstractArray{<:AbstractPolicySampler},
    model::AbstractPolicyObjective,
)
    return map(samplers) do sampler
        return propose(rng, sampler, model)
    end
end

"""
$SIGNATURES

Returns a vector of guesses, conditional on the current state.
"""
function propose(
    rng::Random.AbstractRNG,
    samplers::AbstractArray{<:AbstractPolicySampler},
    model::AbstractPolicyObjective,
    ts::AbstractArray{<:AbstractPolicyTransition},
)
    return map(samplers, ts) do sampler, t
        return propose(rng, sampler, model, t)
    end
end

"""
$SIGNATURES

Returns a vector of initial guesses, where proposals are named
in some way.
"""
@generated function propose(
    rng::Random.AbstractRNG,
    proposals::NamedTuple{names},
    model::AbstractPolicyObjective,
) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[:($name = propose(rng, proposals.$name, model)) for name in names]
    return expr
end

"""
$SIGNATURES

Returns a vector guesses conditional on the current states, where
the guesses are named in some way.
"""
@generated function propose(
    rng::Random.AbstractRNG,
    proposals::NamedTuple{names},
    model::AbstractPolicyObjective,
    ts::NamedTuple{names},
) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :($name = propose(rng, proposals.$name, model, ts.$name)) for name in names
    ]
    return expr
end

######################################################################
# Chains constructors ################################################
######################################################################

"""
    $SIGNATURES

Take a collection of guesses and their associated objective functions
and acceptances (i.e. `AbstractPolicyTransition`s), and bundle
them together to a vector of named tuples.

In this method, the `Vector` of transitions do not have names attached
to the parameters, so we create a `NamedTuple` where each parameter
has a name and return that vector of named tuples.

We also include the objective values as the last value.

TODO: This is be a really bad idea if we have, say, 10k
parameters. Can we get by with something untyped without
using a DataFrame?
"""
function AbstractMCMC.bundle_samples(
    transitions::Vector{<:AbstractPolicyTransition},
    model::AbstractPolicyObjective,
    sampler::AbstractPolicySampler,
    state,
    chain_type::Type{Vector{NamedTuple}};
    param_names = missing,
    kwargs...
)
    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["param_$i" for i in 1:length(keys(first(transitions).params))]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    # Add in the objective to the names
    push!(param_names, "obj")

    # Turn all the transitions into a vector-of-NamedTuple.
    ks = tuple(Symbol.(param_names)...)
    nts = [NamedTuple{ks}(tuple(t.params..., t.obj)) for t in transitions]

    return nts
end

"""
    $SIGNATURES

Take a collection of guesses and their associated objective functions
and acceptances (`PolicyTransition`)
"""
function AbstractMCMC.bundle_samples(
    ts::Vector{<:PolicyTransition{<:NamedTuple}},
    model::AbstractPolicyObjective,
    sampler::AbstractPolicySampler,
    state, # Not sure why this argument is here
    chain_type::Type{Vector{NamedTuple}};
    param_names = missing,
    kwargs...
)
    # If the element type of ts is NamedTuples, just use the names in the
    # struct.

    # Extract NamedTuples
    nts = map(x -> merge(x.params, (obj=x.obj,)), ts)

    # Return em'
    return nts
end

######################################################################
# MCMC Steps #########################################################
######################################################################

"""
$SIGNATURES

Initiailizes the MCMC algorithm.

Returns a 2-tuple of the initial sample and the initial state. In our
implementation, however, the transition struct contains all
information needed for both the sample and the state, so we just
return the same thing twice.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::TemperedPolicyObjective,
    sampler::PolicySampler
)
    params = propose(rng, sampler, model)
    trans = transition(sampler, model, params, false)
    return trans, trans
end


"""
$SIGNATURES

Initializes the MCMC algorithm with a givin initial parameter vector.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::TemperedPolicyObjective,
    sampler::PolicySampler,
    initial_params::AbstractVector{<:Real}
)
    trans = transition(sampler, model, initial_params, false)
    return trans, trans
end

"""
$SIGNATURES

Constitutes the Metropolis-Hastings algorithm.

Conditional on a current state, draws a policy guess. Then, runs a
Metropolis-Hastings comparison on the exponentiated versions of the
state and the guess, according to the inverse temperature stored in
`model`.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::TemperedPolicyObjective,
    sampler::PolicySampler,
    transition_prev::PolicyTransition;
)

    invtemp = model.invtemp

    if invtemp == 0
        candidate = propose(rng, sampler, model)
        objval_candidate = LogDensityProblems.logdensity(model, candidate)

        # Always accept
        trans = transition(sampler, model, candidate, objval_candidate, true)
        return trans, trans
    end

    # Generate a new proposal.
    candidate = propose(rng, sampler, model, transition_prev)

    # Calculate the log acceptance probability and the log density of
    # the candidate. objval is alreay logged
    objval_candidate = LogDensityProblems.logdensity(model, candidate)

    objval_curr = obj(model, transition_prev)
    logα = invtemp * (objval_candidate - objval_curr)
    accept_ratio = exp(logα)

    # We use <= here so has to always accept the new candidate
    # when invtemp is equal to 0 (and thus avoid the rand call)
    trans = if 1 <= accept_ratio
        transition(sampler, model, candidate, objval_candidate, true)
    elseif rand(rng) < accept_ratio
        transition(sampler, model, candidate, objval_candidate, true)
    else
        params = transition_prev.params
        objval = transition_prev.obj
        transition(sampler, model, params, objval, false)
    end

    return trans, trans
end

######################################################################
# Log Density Interface ##############################################
######################################################################

#=
Not sure why we implement the logdensity interface. It was
necessary for MCMCTempering.jl, but since we aren't using that
any more, we don't strictly need it.
=#
LogDensityProblems.logdensity(model::AbstractPolicyObjective, x) = obj(model, x)
LogDensityProblems.logdensity(model::AbstractPolicyObjective, t::AbstractPolicyTransition) = t.obj

# TODO: Understand what number to put here. Do we need a number?
function LogDensityProblems.dimension(model::AbstractPolicyObjective)
    throw(ArgumentError("LogDensityProblems.dimension not implemented for <: AbstractPolicyObjective"))
end
LogDensityProblems.capabilities(::AbstractPolicyObjective) = LogDensityProblems.LogDensityOrder{0}()