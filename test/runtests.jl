using HighDimensionalOptimalPolicies
using Distributions
using Test

function test_objfun(x)
    x1 = x[1]
    x2 = x[2]

    #res = (1 - x1)^2 + 500 * (x2 - x1^2)^2
    res = (x1 - 1)^2 + (x2 - 1)^2
    -res
end

function test_initfun(rng)
    σ = 0.1
    [rand(rng, Normal(1, 1)) for i in 1:2]
end

function test_nextfun(rng, x)
    σ = 0.1
    [rand(rng, Normal(xi, σ)) for xi in x]
end

function test_output(multioutput)
    invtemps = get_invtemps(multioutput)
    for ind in 1:length(invtemps)
        println("  Inverse temperature: ", invtemps[ind])
        objs = get_objective_vec(multioutput; ind)
        mean_obj = mean(objs)
        println("    Average objective value: ", mean_obj)
        avg_policy = get_average_policy(multioutput; ind)
        println("    Average policy: ", avg_policy)
    end
    multioutput
end

function test_SimpleMCMC()
    println("Testing SimpleMCMCSolver:")
    out = get_best_policy(
        SimpleMCMCSolver();
        initfun = test_initfun,
        nextfun = test_nextfun,
        objfun = test_objfun,
        max_invtemp = 100.0,
        invtemps_curvature = 0.8)

    test_output(out)
end

function test_MCMC()
    out = get_best_policy(
        MCMCSolver();
        initfun = test_initfun,
        nextfun = test_nextfun,
        objfun = test_objfun,
        max_invtemp = 200.0,
        invtemps_curvature = 0.8)

    test_output(out)
end

function test_SimulatedAnnealing()
    out = get_best_policy(
        SimulatedAnnealingSolver();
        initfun = test_initfun,
        nextfun = test_nextfun,
        objfun = test_objfun,
        max_invtemp = 200.0,
        n_invtemps = 10,
        invtemps_curvature = 2.0)

    test_output(out)
end

function test_IndependentSimulatedAnnealing()
    out = get_best_policy(
        IndependentSimulatedAnnealingSolver();
        initfun = test_initfun,
        nextfun = test_nextfun,
        objfun = test_objfun,
        max_invtemp = 200.0,
        n_invtemps = 10,
        invtemps_curvature = 2.0,
        n_independent_runs = 100)

    test_output(out)
end

function test_PTMCMC()
    out = get_best_policy(
        PTMCMCSolver();
        initfun = test_initfun,
        objfun = test_objfun,
        nextfun = test_nextfun,
        max_invtemp = 200.0,
        invtemps_curvature = 0.8)

    test_output(out)
end

function test_Pigeons()
    out = get_best_policy(
        PigeonsSolver();
        initfun = test_initfun,
        nextfun = test_nextfun,
        objfun = test_objfun,
        max_invtemp = 200.0)

    test_output(out)
end

function test_PigeonsMPI()
    childprocess =  ChildProcess(
        n_local_mpi_processes = 2,
        n_threads = 2,
        dependencies = [HighDimensionalOptimalPolicies]
    )

    out = get_best_policy(
        PigeonsMPISolver();
        initfun = test_initfun,
        nextfun = test_nextfun,
        objfun = test_objfun,
        max_invtemp = 200.0,
        n_local_mpi_processes = 2,
        n_threads = 2,
        dependencies = [])

    test_output(out)
end

function test_all()
    test_SimpleMCMC()
    test_MCMC()
    test_SimulatedAnnealing()
    test_PTMCMC()
    test_Pigeons()
    test_PigeonsMPI()
end

@testset "HighDimensionalOptimalPolicies.jl" begin
   # test_SimpleMCMC()
    test_MCMC()
    test_SimulatedAnnealing()
    test_PTMCMC()
    test_Pigeons()
 #   test_PigeonsMPI()
end
