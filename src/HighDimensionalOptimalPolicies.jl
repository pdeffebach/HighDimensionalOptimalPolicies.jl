module HighDimensionalOptimalPolicies

using AbstractMCMC
using Random
using Distributions
using DocStringExtensions
using Pigeons
using MCMCChains

abstract type AbstractPolicySolver end

include("Simple/simple.jl")

include("AbstractMCMC/typedef.jl")
include("AbstractMCMC/interface.jl")
include("AbstractMCMC/paralleltempering.jl")
include("AbstractMCMC/api.jl")

include("Pigeons/pigeons.jl")
include("Pigeons/api.jl")

include("Testing/testing.jl")

end
