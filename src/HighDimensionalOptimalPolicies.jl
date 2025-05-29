module HighDimensionalOptimalPolicies

######################################################################
# Imports ############################################################
######################################################################

using Random
using Distributions
using Statistics
using Distributed

using Tables
using CSV
using OrderedCollections
using StatsBase
using SplitApplyCombine

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

export save_policy_output_csv
export CSVPolicyOutput
export MultiCSVPolicyOutput
export Tables

######################################################################
# Includes ###########################################################
######################################################################

include("shared/policy_api.jl")
include("shared/saving.jl")

include("simplemcmc/api.jl")

include("abstractmcmc/typedef.jl")
include("abstractmcmc/interface.jl")
include("abstractmcmc/api_no_pt.jl")
include("abstractmcmc/api_pt.jl")

include("pigeons/interface.jl")
include("pigeons/api.jl")

end # module
