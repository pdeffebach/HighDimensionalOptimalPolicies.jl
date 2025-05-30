"""
The sampling strategy of the problem along with the objective
function.
"""
struct HighDimensionalProblem{A, B, C}
    initfun::A
    objfun::B
    nextfun::C
end

"""
Stores the tempered information along with the sampling and
objective information.
"""
struct OuterHighDimensionalProblem{I, P}
    invtemp::I
    problem::P
end

"""
Returns the logged *modified* objective function, that is,
multiplied by the inverse temperature.

`p(x)` returns `invtemp * obj(x)` where `p` is a
`OuterHighDimensionalProblem`.
"""
function (outerprob::OuterHighDimensionalProblem)(x)
    invtemp = outerprob.invtemp
    objfun = outerprob.problem.objfun
    invtemp * objfun(x)
end

"""
Define how to sample i.i.d draws.
"""
function Pigeons.sample_iid!(reference_log_potential::OuterHighDimensionalProblem, replica, shared)
    rng = replica.rng
    state = replica.state
    new_state = reference_log_potential.problem.initfun(rng)
    state .= new_state
    return nothing
end

"""
    Pigeons.step!(hd::HighDimensionalProblem, replica, shared)

The core logic of a Metropolis-Hastings step in Pigeons.jl.
"""
function Pigeons.step!(hd::HighDimensionalProblem, replica, shared)
    state = replica.state
    rng = replica.rng
    # Note: the log_potential is an InterpolatedLogPotential between
    # the target and reference Think of the reference as being
    #
    # exp(0 * W(x))
    #
    # and the ideal distribution we want to draw from as being
    #
    # exp(max_invtemp * W(x))
    #
    # and replica stores a value xi between 0 and 1. Then, this gives
    # a log potential of
    #
    # exp(xi * W(x))
    #
    # Which is what we do in our own PT implementation.
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    log_pr_before = log_potential(state)

    state_next = hd.nextfun(rng, state)
    log_pr_after = log_potential(state_next)
    # accept-reject step

    accept_ratio = exp(log_pr_after - log_pr_before)
    if 1 <= accept_ratio
        state .= state_next
    elseif rand(rng) < accept_ratio
        state .= state_next
    else
        # Do nothing
    end
end

"""
    Pigeons.initialization(outerprob::OuterHighDimensionalProblem, rng::AbstractRNG, ::Int)

Initialize the problem.
"""
Pigeons.initialization(outerprob::OuterHighDimensionalProblem, rng::AbstractRNG, ::Int) =
    outerprob.problem.initfun(rng)

# This was a recommendation by the Pigeons.jl devs for
# how input the inverse temperatures manually.

#=
function Pigeons.adapt_tempering(tempering::Pigeons.NonReversiblePT, reduced_recorders, iterators, variational, state::Vector{<:Real}, chain_indices)
    new_path = Pigeons.update_path_if_needed(tempering.path, reduced_recorders, iterators, variational, state)
    return Pigeons.NonReversiblePT(
        new_path,
        tempering.schedule, # just use the previous schedule,
        Pigeons.communication_barriers(reduced_recorders, tempering.schedule, chain_indices)
    )
end
=#