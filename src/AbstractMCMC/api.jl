struct MCMCSolver <: AbstractPolicySolver end

function get_best_policy(::MCMCSolver; initfun, objfun, nextfun, β::Real, kwargs...)
    proposal = PolicySampler(initfun, nextfun)
    model = PolicyObjective(objfun, β)
    num_rounds = 1000
    rng = Random.default_rng()

    sample(
        rng,
        model,
        proposal,
        num_rounds,
        chain_type = MCMCChains.Chains)
end
