using Serialization
using Pigeons
using HighDimensionalOptimalPolicies
Pigeons.mpi_active_ref[] = true

pt_arguments = 
    try
        Pigeons.deserialize_immutables!(raw"/home/peterwd/Documents/Development/HighDimensionalOptimalPolicies/examples/OptimalTransport/results/all/2025-04-01-11-03-06-GX8Rax4g/immutables.jls")
        deserialize(raw"/home/peterwd/Documents/Development/HighDimensionalOptimalPolicies/examples/OptimalTransport/results/all/2025-04-01-11-03-06-GX8Rax4g/.pt_argument.jls")
    catch e
        println("Hint: probably missing dependencies, use the dependencies argument in MPIProcesses() or ChildProcess()")
        rethrow(e)
    end

pt = PT(pt_arguments, exec_folder = raw"/home/peterwd/Documents/Development/HighDimensionalOptimalPolicies/examples/OptimalTransport/results/all/2025-04-01-11-03-06-GX8Rax4g")
pigeons(pt)
