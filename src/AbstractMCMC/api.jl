struct MCMCSolver <: AbstractPolicySolver end

struct TemperedMCMCSolver <: AbstractPolicySolver end

function get_best_policy(::MCMCSolver; initfun, objfun, nextfun, β::Real, kwargs...)
    sampler = PolicySampler(initfun, nextfun)
    model = TemperedPolicyObjective(objfun, β)
    num_rounds = 1000
    rng = Random.default_rng()

    s = sample(
        rng,
        model,
        sampler,
        num_rounds,
        chain_type = MCMCChains.Chains)
    last(s).params
end

function get_best_policy(::TemperedMCMCSolver; initfun, objfun, nextfun, β::Real, kwargs...)
    sampler = PolicySampler(initfun, nextfun)
    model = TemperedPolicyObjective(objfun, β)
    num_rounds = 1000
    rng = Random.default_rng()

    inverse_temperatures = 0.9 .^ (0:20)
    sampler_tempered = TemperedSampler(sampler, inverse_temperatures)

    s = sample(
        rng,
        model,
        sampler_tempered,
        num_rounds,
        chain_type = MCMCChains.Chains)
    last(s).params
end

