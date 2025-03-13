module HighDimensionalOptimalPolicies

using AbstractMCMC
using Random
using Distributions
using DocStringExtensions
using Pigeons
using MCMCChains
using MCMCTempering
using Tables

using AdvancedMH
using LogDensityProblems

abstract type AbstractPolicySolver end
abstract type AbstractPolicyOutput end

include("Simple/simple.jl")

include("AbstractMCMC/typedef.jl")
include("AbstractMCMC/interface.jl")
include("AbstractMCMC/api.jl")

include("Pigeons/pigeons.jl")
include("Pigeons/api.jl")

include("Testing/testing.jl")

end
