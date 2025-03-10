struct MCMCSolver <: AbstractPolicySolver end

struct TemperedMCMCSolver <: AbstractPolicySolver end

function get_best_policy(::MCMCSolver; initfun, objfun, nextfun, β::Real, kwargs...)
    proposal = PolicySampler(initfun, nextfun, β)
    model = PolicyObjective(objfun, β)
    num_rounds = 100
    rng = Random.default_rng()

    sample(
        rng,
        model,
        proposal,
        num_rounds,
        chain_type = MCMCChains.Chains)
end

function get_best_policy(::TemperedMCMCSolver; initfun, objfun, nextfun, β::Real, kwargs...)
    βs = range(0, β; length = 3)
    proposal = PolicySampler(initfun, nextfun, β)
    model = PolicyObjective(objfun)
    num_rounds = 100
    rng = Random.default_rng()

    sample(
        rng,
        model,
        proposal,
        num_rounds,
        chain_type = MCMCChains.Chains)
end

