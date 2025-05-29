######################################################################
# Abstract types #####################################################
######################################################################

"""
Abstract type for policy sampler, which defines this way both
initial and conditional guesses are drawn.
"""
abstract type AbstractPolicySampler <: AbstractMCMC.AbstractSampler end

"""
Abstract type for the policy transition.

Stores the current state, objective value, and whether that
guess was accepted. Provides convenient caching to avoid costly
re-computation of the objective function.
"""
abstract type AbstractPolicyTransition end

"""
Stores the policy objective, i.e. a function which takes in a
state and returns a real number.
"""
abstract type AbstractPolicyObjective <: AbstractMCMC.AbstractModel end

######################################################################
# Concrete types #####################################################
######################################################################

"""
Stores the proposal function, i.e. the function that guesses the
initial policy and the function that returns the next guess
conditional on the current guess.

$FIELDS
"""
struct PolicySampler{I, N} <: AbstractPolicySampler
    """
    `initfun(rng)` must return a random initial guess from the
    state space.
    """
    initfun::I
    """
    `nextfun(rng, state)` must return a new guess conditional on
    the current state.
    """
    nextfun::N
end


"""
Stores the objective function and a single inverse temperature. Used
for a single MCMC chain.

$FIELDS
"""
struct TemperedPolicyObjective{F, T<:Real} <: AbstractPolicyObjective
    """
    `objfun(state)` must return the welfare value of the current
    state. Higher values indicate higher welfare.
    """
    objfun::F
    """
    The inverse temperature for this chain.
    """
    invtemp::T
end

"""
`p(rng)` Returns an initial guess for policy sampler `p`.
"""
function (p::PolicySampler)(rng)
    p.initfun(rng)
end

"""
`p(rng, state)` returns the next guess conditional on the current
guess for policy sampler `p`.
"""
function (p::PolicySampler)(rng, state)
    p.nextfun(rng, state)
end

"""
Stores the current guess (params), the objective value, and
whether the proposal was accepted.

$FIELDS
"""
struct PolicyTransition{T, L<:Real} <: AbstractPolicyTransition
    """
    Current state of the MCMC chain (called `params` for consistency
    with `AbstractMCMC` API)
    """
    params::T
    """
    Current objective value of the MCMC chain.
    """
    obj::L
    """
    Whether the proposed guess was accepted.
    """
    accepted::Bool
end