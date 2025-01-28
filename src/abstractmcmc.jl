using AbstractMCMC
using Random
using Infiltrator

abstract type MHSampler1 <: AbstractMCMC.AbstractSampler end

abstract type AbstractTransition1 end

abstract type Proposal1{P} end

struct ValueModel1{F} <: AbstractMCMC.AbstractModel
    objfun :: F
end

struct MetropolisHastings1{D} <: MHSampler1
    proposal :: D
end

struct GenericProposal1{P} <: Proposal1{P}
    proposal :: P
end

function (p::GenericProposal1{<:Function})()
    p.proposal()
end

function (p::GenericProposal1{<:Function})(t)
    p.proposal(t)
end


# Create a very basic Transition type, stores the
# parameter draws, the log probability of the draw,
# and the draw information until this point
struct Transition1{T,L<:Real} <: AbstractTransition1
    params :: T
    obj :: L
    accepted :: Bool
end

function obj(model::ValueModel1, params)
    model.objfun(params)
end
function obj(::ValueModel1, t::Transition1)
    t.obj
end


# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:AbstractTransition1},
    model::ValueModel1,
    sampler::MHSampler1,
    state,
    chain_type::Type{Vector{NamedTuple}};
    param_names=missing,
    kwargs...
)
    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["param_$i" for i in 1:length(keys(ts[1].params))]
    else
        # Deepcopy to be thread safe.
        param_names = deepcopy(param_names)
    end

    push!(param_names, "lp")

    # Turn all the transitions into a vector-of-NamedTuple.
    ks = tuple(Symbol.(param_names)...)
    nts = [NamedTuple{ks}(tuple(t.params..., t.obj)) for t in ts]

    return nts
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Transition1{<:NamedTuple}},
    model::ValueModel1,
    sampler::MHSampler1,
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

# AbstractMCMC.jl interface
function AbstractMCMC.getparams(t::Transition1)
    return t.params
end

function AbstractMCMC.setparams!!(model::ValueModel1, t::Transition1, params)
    return Transition1(
        params,
        obj(model, params),
        t.accepted
    )
end

function propose(rng::Random.AbstractRNG, sampler::MHSampler1, model::ValueModel1)
    return propose(rng, sampler.proposal, model)
end
function propose(
    rng::Random.AbstractRNG,
    sampler::MHSampler1,
    model::ValueModel1,
    transition_prev::Transition1,
)
    return propose(rng, sampler.proposal, model, transition_prev.params)
end

function propose(
    rng::Random.AbstractRNG,
    proposal::Proposal1{<:Function},
    model::ValueModel1
)
    return proposal.proposal()
end

function propose(
    rng::Random.AbstractRNG,
    proposal::Proposal1{<:Function},
    model::ValueModel1,
    t
)
    return proposal.proposal(t)
end

####################
# Multiple proposals
####################

function propose(
    rng::Random.AbstractRNG,
    proposals::AbstractArray{<:Proposal1},
    model::ValueModel1,
)
    return map(proposals) do proposal
        return propose(rng, proposal, model)
    end
end
function propose(
    rng::Random.AbstractRNG,
    proposals::AbstractArray{<:Proposal1},
    model::ValueModel1,
    ts,
)
    return map(proposals, ts) do proposal, t
        return propose(rng, proposal, model, t)
    end
end

@generated function propose(
    rng::Random.AbstractRNG,
    proposals::NamedTuple{names},
    model::ValueModel1,
) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[:($name = propose(rng, proposals.$name, model)) for name in names]
    return expr
end

@generated function propose(
    rng::Random.AbstractRNG,
    proposals::NamedTuple{names},
    model::ValueModel1,
    ts,
) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :($name = propose(rng, proposals.$name, model, ts.$name)) for name in names
    ]
    return expr
end


function transition(sampler::MHSampler1, model::ValueModel1, params, accepted)
    objval = obj(model, params)
    return transition(sampler, model, params, objval, accepted)
end
function transition(sampler::MHSampler1, model::ValueModel1, params, objval::Real, accepted)
    return Transition1(params, objval, accepted)
end

# Define the first sampling step.
# Return a 2-tuple consisting of the initial sample and the initial state.
# In this case they are identical.
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::ValueModel1,
    sampler::MHSampler1;
    initial_params=nothing,
    kwargs...
)
    params = initial_params === nothing ? propose(rng, sampler, model) : initial_params
    trans = transition(sampler, model, params, false)
    return trans, trans
end

# Define the other sampling steps.
# Return a 2-tuple consisting of the next sample and the the next state.
# In this case they are identical, and either a new proposal (if accepted)
# or the previous proposal (if not accepted).
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::ValueModel1,
    sampler::MHSampler1,
    transition_prev::AbstractTransition1;
    kwargs...
)
    # Generate a new proposal.
    candidate = propose(rng, sampler, model, transition_prev)

    # Calculate the log acceptance probability and the log density of the candidate.
    objval_candidate = obj(model, candidate)

    logα = log(objval_candidate) - log(obj(model, transition_prev))

    # Decide whether to return the previous params or the new one.
    trans = if -Random.randexp(rng) < logα
        transition(sampler, model, candidate, objval_candidate, true)
    else
        params = transition_prev.params
        objval = transition_prev.obj
        Transition1(params, objval, false)
    end

    return trans, trans
end

