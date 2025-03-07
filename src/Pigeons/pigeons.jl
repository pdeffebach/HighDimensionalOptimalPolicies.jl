struct HighDimensionalProblem{A, B, C}
    initfun::A
    objfun::B
    nextfun::C
end

struct OuterHighDimensionalProblem{Ξ, H}
    ξ::Ξ
    problem::H
end

function (outerprob::OuterHighDimensionalProblem)(x)
    ξ = outerprob.ξ
    objfun = outerprob.problem.objfun
    ξ * objfun(x)
end

function Pigeons.sample_iid!(reference_log_potential::OuterHighDimensionalProblem, replica, shared)
    rng = replica.rng
    reference_log_potential.problem.initfun(rng)
end

function Pigeons.step!(hd::HighDimensionalProblem, replica, shared)
    state = replica.state
    rng = replica.rng
    # Note: the log_potential is an InterpolatedLogPotential between the target and reference
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    log_pr_before = log_potential(state)
    state_next = hd.nextfun(rng, state)
    log_pr_after = log_potential(state_next)
    # accept-reject step
    accept_ratio = exp(log_pr_after - log_pr_before)
    if 1 < accept_ratio
        state .= state_next
    elseif rand(rng) < accept_ratio
        state .= state_next
    else
        # Do nothing
    end
end

Pigeons.initialization(outerprob::OuterHighDimensionalProblem, rng::AbstractRNG, ::Int) =
    outerprob.problem.initfun(rng)