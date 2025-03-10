######################################################################
# AbstractMCMC Interface #############################################
######################################################################

# Handling the objective value #######################################

"""
$SIGNATURES

Get the objective value for the current state.
"""
function obj(model::PolicyObjective, params)
    model.objfun(params)
end

"""
$SIGNATURES

Get the objective value of the current state, which is cached
in the transition `t`, to avoid re-computing the policy objective.
"""
function obj(::PolicyObjective, t::AbstractPolicyTransition)
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
function AbstractMCMC.setparams!!(model::PolicyObjective, t::PolicyTransition, params)
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
function transition(sampler::PolicySampler, model::PolicyObjective, params, accepted)
    objval = obj(model, params)
    return transition(sampler, model, params, objval, accepted)
end

"""
$SIGNATURES

Construct a transition object by caching an existing objective value.
"""
function transition(sampler::PolicySampler, model::PolicyObjective, params, objval::Real, accepted)
    return PolicyTransition(params, objval, accepted)
end

# New proposals for next policy guess ################################

"""
$SIGNATURES

Returns the initial policy guess, but first unwraps the policy
sampler.
"""
function propose(rng::Random.AbstractRNG, sampler::PolicySampler, model::PolicyObjective)
    return propose(rng, sampler.proposal, model)
end

"""
$SIGNATURES

Returns the initial policy guess.
"""
function propose(
    rng::Random.AbstractRNG,
    proposal::PolicyProposalCallable,
    model::PolicyObjective
)
    return proposal(rng)
end

"""
$SIGNATURES

Returns the next policy guess, conditional on the current one,
but first needs to unwrap the sampler.
"""
function propose(
    rng::Random.AbstractRNG,
    sampler::PolicySampler,
    model::PolicyObjective,
    transition_prev::PolicyTransition,
)
    return propose(rng, sampler.proposal, model, transition_prev.params)
end

"""
$SIGNATURES

Returns the next policy guess, conditional on the current one.
"""
function propose(
    rng::Random.AbstractRNG,
    proposal::PolicyProposalCallable,
    model::PolicyObjective,
    t
)
    return proposal(rng, t)
end

# Multiple Proposals #################################################

"""
$SIGNATURES

Returns a vector of initial guesses, after unwrapping the sampler.
"""
function propose(
    rng::Random.AbstractRNG,
    proposals::AbstractArray{<:PolicyProposalCallable},
    model::PolicyObjective,
)
    return map(proposals) do proposal
        return propose(rng, proposal, model)
    end
end

"""
$SIGNATURES

Returns a vector of initial gueses.
"""
function propose(
    rng::Random.AbstractRNG,
    proposals::AbstractArray{<:PolicyProposalCallable},
    model::PolicyObjective,
    ts,
)
    return map(proposals, ts) do proposal, t
        return propose(rng, proposal, model, t)
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
    model::PolicyObjective,
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
    model::PolicyObjective,
    ts,
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
and acceptances (`PolicyTransition`)

In this method, the vector of transitions do not have names attached
to the parameters, so we create a `NamedTuple` where each parameter
has a name and return that vector of named tuples.

We also include the objective values as the last value.
"""
function AbstractMCMC.bundle_samples(
    transitions::Vector{<:PolicyTransition},
    model::PolicyObjective,
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
    model::PolicyObjective,
    sampler::AbstractPolicySampler,
    state,
    chain_type::Type{Vector{NamedTuple}};
    param_names=missing,
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

Initiailizes the MCMC algorithm. Returns a 2-tuple of the initial
sample and the initial state. In our implementation, however,
the transition struct contains all information needed for both
the sample and the state, so we just return the same thing twice.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::PolicyObjective,
    sampler::PolicySampler;
    initial_params=nothing,
    kwargs...
)
    params = initial_params === nothing ? propose(rng, sampler, model) : initial_params
    trans = transition(sampler, model, params, false)
    return trans, trans
end

"""
$SIGNATURES

Continutes the Metropolis-Hastings algorithm.
Returns a 2-tuple of the next sample and the next state.
In our implementation, however, the transition struct contains
all information needed for both the sample and the state, so
we just return the same thing twice.
"""
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::PolicyObjective,
    sampler::PolicySampler,
    transition_prev::PolicyTransition;
    kwargs...
)
    invtemp = sampler.invtemp

    # Generate a new proposal.
    candidate = propose(rng, sampler, model, transition_prev)

    # Calculate the log acceptance probability and the log density of the candidate.
    objval_candidate = obj(model, candidate)

    logα = invtemp * (objval_candidate - obj(model, transition_prev))

    # Decide whether to return the previous params or the new one.
    trans = if -Random.randexp(rng) < logα
        transition(sampler, model, candidate, objval_candidate, true)
    else
        params = transition_prev.params
        objval = transition_prev.obj
        PolicyTransition(params, objval, false)
    end

    return trans, trans
end



