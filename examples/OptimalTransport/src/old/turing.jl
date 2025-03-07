# Import the package.
using AdvancedMH
using Distributions
using MCMCChains

using LinearAlgebra


function turing()

    # Create data, 30 draws from N(0, 1)
    data = rand(Normal(0, 1), 30)

    # Define the components of a basic model.
    insupport(θ) = θ[2] >= 0

    # We guess the data is drawn from N(θ[1], θ[2])
    dist(θ) = Normal(θ[1], θ[2])

    # We examine the probability that we observed this data
    # given this parametrizations
    density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

    # Construct a DensityModel.
    # A DensityModel is just a way to wrap a self-contained
    # log-likelihood function
    model = DensityModel(density)

    # Set up our sampler with a joint multivariate Normal proposal.
    # Our prior is that the the parameters are drawn from
    # a normal distribution with mean 0 and unit variance.
    # This seems odd because we know that the second parameter
    # must be greater than zero, but I guess this shows
    # the strength of their package.
    # Random Walk Metropolis-Hastings?
    # spl?
    spl = MetropolisHastings(RandomWalkProposal(MvNormal(zeros(2), I)))

    # Sample from the posterior.
    # Not sure what any of this means.
    chain = sample(model, spl, 100000; param_names=["μ", "σ"], chain_type=Chains)
end

function turing_ols()
    N = 100
    σ = 5.0
    X = hcat(ones(N), randn(N))
    β_true = [1.4, 3.4]
    ϵ = rand(Normal(0, σ), N)
    y = (X * β_true) .+ ϵ

    σ_guess = 2.0

    # The log likelihood
    density = let X = X, y = y
        β -> begin
            ϵ̂ = (X * β .- y)
            L = logpdf.(Normal(0, σ_guess), ϵ̂)
            sum(L)
        end
    end
    model = DensityModel(density)


    proposal = RandomWalkProposal(MvNormal(zeros(2), I))

    mh_problem = MetropolisHastings(proposal)

    β̂ = (X'X) \ (X'y)
    println("β̂_1 = ", β̂[1])
    println("β̂_2 = ", β̂[2])

    chain = sample(model, mh_problem, 100000; param_names=["β1", "β2"], chain_type=Chains)
end