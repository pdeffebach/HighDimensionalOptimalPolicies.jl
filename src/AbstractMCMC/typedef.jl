######################################################################
# Defining Structs ###################################################
######################################################################

# Abstract types #####################################################

"""
An abstract type for policy sampler.

Stores the function which creates initial policies guesses and draws
the next policy guess conditional on the current policy guess.

In actuality, it stores an a `AbstractPolicyProposalCallable` struct,
which is a callable struct wrapping the actual user-defined guessing
function.
"""
abstract type AbstractPolicySampler <: AbstractMCMC.AbstractSampler end

"""
An abstract type for the policy transition.

Stores the current state, objective value, and whether that
guess was accepted. Provides convenient caching to avoid costly
re-computation of the objective function.
"""
abstract type AbstractPolicyTransition end

"""
Abstract type for the policy prosal.

Stores the "guessing function", which returns initial guesses
and the next guess conditional on the current guess.
"""
abstract type AbstractPolicyProposalCallable{I, N} end

"""
Stores the policy objective, i.e. a function which takes in a
state and returns a real number.
"""
abstract type AbstractPolicyObjective <: AbstractMCMC.AbstractModel end

# Concrete types #####################################################

"""
Stores the proposal function, i.e. the function that guesses the
initial policy and the function that returns the next guess
conditional on the current guess.
"""
struct PolicySampler{P, I} <: AbstractPolicySampler
    proposal::P
    invtemp::I
end

"""
Stores the function that returns an initial guess or the next
guess conditional on the current guess.
"""
struct PolicyProposalCallable{I, N} <: AbstractPolicyProposalCallable{I, N}
    initfun::I
    proposalfun::N
end

function PolicySampler(initfun, nextfun, β)
    pc = PolicyProposalCallable(initfun, nextfun)
    PolicySampler(pc, β)
end

struct PolicyObjective{F} <: AbstractPolicyObjective
    objfun::F
end

"""
Call the underlying guessing function by making
`PolicyProposalCallable` a callable struct.

Returns an initial guess.
"""
function (p::PolicyProposalCallable)(rng)
    p.initfun(rng)
end

"""
Call the underlying guessing function by making
`PolicyProposalCallable` a callable struct.

Returns the next guess conditional on the current guess.
"""
function (p::PolicyProposalCallable)(rng, t)
    p.proposalfun(rng, t)
end

"""
Stores the current guess (params), the objective value, and
whether the proposal was accepted.
"""
struct PolicyTransition{T, L<:Real} <: AbstractPolicyTransition
    params::T
    obj::L
    accepted::Bool
end