module HighDimensionalOptimalPolicies

######################################################################
# Imports ############################################################
######################################################################

using Random
using Distributions

using Tables
using StatsBase
using Statistics
using Distributed

using AbstractMCMC
using MCMCChains
using LogDensityProblems

using Pigeons

using DocStringExtensions

######################################################################
# Exports ############################################################
######################################################################

export AbstractPolicySolver
export AbstractPolicyOutput
export AbstractMultiPolicyOutput

export last_half
export make_invtemps
export GenericSolverInput

export get_invtemps
export get_best_policy

export get_policy_vec
export get_average_policy
export get_last_policy
export get_objective_vec

export test_mixing

export MCMCSolver
export SimulatedAnnealingSolver
export IndependentSimulatedAnnealingSolver
export MCMCSolverOutput
export MultiMCMCSolverOutput

export PTMCMCSolver
export PTMCMCSolverOutput
export MultiPTMCMCSolverOutput

export PigeonsSolver
export PigeonsMPISolver
export PigeonsSolverOutput
export MultiPigeonsSolverOutput


######################################################################
# Includes ###########################################################
######################################################################

include("shared.jl")

include("SimpleMCMC/api.jl")

include("AbstractMCMC/typedef.jl")
include("AbstractMCMC/interface.jl")
include("AbstractMCMC/api_no_pt.jl")
include("AbstractMCMC/api_pt.jl")

include("Pigeons/interface.jl")
include("Pigeons/api.jl")

include("Testing/testing.jl")

end # module
