# Tutorial

Here we set up a simple high dimensional problem and outline the algorithms this package provides to characterize optimal policies. 

## Set-up

```@setup main
```

We start by importing the HighDimensionalOptimalPolicies.jl package, along with Plots.jl, which will help us visualize results. 

```@example main
using HighDimensionalOptimalPolicies
using Plots, StatsPlots
using Random, Distributions
using LinearAlgebra, SpecialFunctions
using StatsBase
using DataFrames
```

Next we define a simple high dimensional problem. Consider a vector of random numbers of length ``L``, denoted ``\vec{r}``. Our goal is to find a policy vector ``\vec{p}`` of length ``L`` filled with zeros and exactly ``L_p`` ones. We want to choose the locations of the ones to maximize

```math
W(\vec{p}) = \vec{p} \cdot \vec{r}
```

This is a convenient set-up to analyze high dimensional optimal policies because the state space is very large, ``L \text{ Choose } L_p`` and if ``\vec{r}`` is well chosen, then there may be many policies with similar welfare values. 

## Implementing `get_best_policy`

For concreteness, we will think of ``L`` as the total number of nodes on a transportation network and ``L_p`` as the number of edges to upgrade, where we want to try and upgrade edges that have the highest value. 

```@example main
n_edges = 1000
n_edges_to_upgrade = 500
network_values = sort(rand(LogNormal(1.0), n_edges), rev = true)
network_values = network_values ./ norm(network_values)
```

Our goal is to call the function `get_best_policy` from HighDimensionalOptimalPolicies.jl. To do this, we need to define three functions ourselves

* A function for drawing initial guesses from the policy state space
* A function for drawing the *next* guess conditional on the current guess
* The objective function

HighDimensionalOptimalPolicies.jl requires passing a random number generator (`rng`) to the initial-guess and next-guess functions. We use a `let` block to capture global variables when we define these functions in order to improve performance and reduce bugs (in case a global variable gets redefined). 

```@example main
initfun = let n_edges = n_edges, n_edges_to_upgrade = n_edges_to_upgrade
    rng -> begin
        fill(false, n_edges_to_upgrade)
        inds = sample(rng, 1:n_edges, n_edges_to_upgrade; replace = false)
        p = fill(false, n_edges)
        p[inds] .= true
        p
    end
end
```

To choose the next policy, conditional on the current one, we simply choose a pair of two policies and edges and then swap whether or not they are upgraded. That is, we randomly select an edge which is non-upgraded under the current guess and choose to upgrade it. We also randomly select an edge which is upgraded under the current guess and choose not to upgrade it. 

```@example main
nextfun =  let n_edges = n_edges, n_edges_to_upgrade = n_edges_to_upgrade
    (rng, state) -> begin
        upgraded_edges = findall(state)
        not_upgraded_edges = findall(==(false), state)

        edge_to_drop = sample(rng, upgraded_edges)
        edge_to_add = sample(rng, not_upgraded_edges)

        new_edges_to_upgrade = copy(state)
        new_edges_to_upgrade[edge_to_drop] = false
        new_edges_to_upgrade[edge_to_add] = true

        new_edges_to_upgrade
    end
end
```

Finally, we define the objective function. Because the objective function is deterministic, we do not need to pass the random number generator to this function. However, we still use a `let` block to capture the transportation network values. 

```@example main
objfun = let network_values = network_values
    state -> begin
        dot(state, network_values)
    end
end
```

Now, we we run the Parallel Tempering algorithm to get a set of optimal policies. We specify the following arguments

* `max_invtemp`: The maximum inverse temperature
* `invtemps_curvature`: The way inverse temperatures "ramp up" from zero to the highest value (see below for deatils)
* `n_invtemps`: The number of inverse temperatures to use. This is synonymous with the number of "chains" to run with the Parallel Tempering algorithm
* `n_inner_rounds`: The total number of policy draws we will take
* `n_swap_rounds`: The number of swap rounds, i.e. the number of times the inverse temperatures "meet up" and randomly swap their policy states. 

```@example main
out = get_best_policy(
    PTMCMCSolver(); 
    initfun = initfun,
    nextfun = nextfun, 
    objfun = objfun, 
    max_invtemp = 50.0,
    invtemps_curvature = 2.0,
    n_invtemps = 10,
    n_inner_rounds = 10000,
    n_swap_rounds = 100)
```

### The inverse temperatures

Note that in the above example, we did not choose the vector of inverse temperatures directly, rather we chose a maximum inverse temperature (`max_invtemp`) and a curvature. 

This is controlled by the function `make_invtemps`. A value between greater than 1 of `invtemps_curvature` causes means many inverse temperatures are close to zero, with a slow ramp-up, while a value between 0 and 1 means many temperatures close to the maximum temperature. 

!!! note 
    The inverse temperatures produced by `make_invtemps`, and *all* output of `get_best_policy` are organized with the highest temperature *first*. 

```@example main
max_invtemp = 25.0
n_invtemps = 20
invtemps_g1 = make_invtemps(25.0, length = n_invtemps, invtemps_curvature = 2.0)
invtemps_1 = make_invtemps(25.0, length = n_invtemps, invtemps_curvature = 1.0)
invtemps_l1 = make_invtemps(25.0, length = n_invtemps, invtemps_curvature = 0.5)
plot(1:n_invtemps, [invtemps_g1 invtemps_1 invtemps_l1]; 
    xlab = "Inverse temperature index",
    ylab = "Inverse temperature", 
    label = ["Curvature = 2.0" "Curvature = 1.0" "Curvature = 0.5"])
```

In general, you want to use an `invtemps_curvature` greater than `1` to ensure sufficient mixing. 

## Exploring the output

We use the functions `get_objective_vec` and `get_policy_vec` and to inspect our output. To start, let's plot out objective values across time to see how the algorithm converged from drawing uniformly from the state space of policies to drawing from the distribution of optimal policies. We use the keyword argument `only_last_half` to tell `get_objective_vec` that we want the objective value from *all* iterations, including the initial burn-in period which are unlikely to be optimal draws. 

Here we see that in the last half of the iterations, the objective value resembles a random walk, indicating we have settled on a set of optimal policies. 

```@example main
function plot_objectives_time(out; only_last_half = false, ind = 1)
    objvec = get_objective_vec(out; only_last_half, ind)
    plot(
        1:length(objvec), 
        objvec, 
        xlab = "Iteration", 
        ylab = "Objective value", 
        label = false, 
        color = "black")
end

plot_objectives_time(out)
```

What can we learn about the optimal policies? Recall that our vector of network values was sorted from the highest value to the lowest. As a consequence, we should see lots of improved edges close close to the front of the vector. 

Here we see that edges close to the front of the vector are almost always improved, while those close to the back of the vector are almost never improved. 

In this example, we know that the optimal policy *in general* is to improve the first 100 edges. However we clearly are not stuck in this global optimum, because at index 100, edges are being improved at far less than 100% of the time. 

```@example main
plot_mean_policy = function(out)
    pvec = get_policy_vec(out)
    # You can also do get_average_policy(out)
    mean_policy = mean(pvec)
    plot(
        1:length(mean_policy),
        mean_policy .* 100,
        xlab = "Edge index",
        ylab = "Percent of iterations with improvement",
        label = false, 
        color = "black", 
        xticks = 0:100:n_edges, 
        ylim = (0, 100))
end
plot_mean_policy(out)
```

### Testing for optimal mixing

The function `test_mixing` returns a `t-statistic` for whether the objective values drawn using our algorithm are drawn from the set of optimal policies. To use this function, however, we need to account for the log of the size of the state space, `log_n`. For our current example, ``1000 \text{ Choose } 100`` is such a large number Julia cannot calculate it without an overflow! We use the function `logabsbinomial` from SpecialFunctions.jl instead, which gives us the log of the size of the state space for this problem. 

For more information, see [kreindler2023](@citet)

```@example main
log_n = SpecialFunctions.logabsbinomial(n_edges, n_edges_to_upgrade)[1]
test_mixing(out, log_n)
```

This low value indicates we are *highly* confident that we are drawing from the the optimal set of policies. 

### Parallelization

The algorithm `PTMCMCSolver` uses the Distributed standard library's `pmap` for parallelization, so parallalelization happens automatically depending on the number of cores Julia is running with. 

## Parallel Tempering with Pigeons.jl

In addition to our "naive" implementation of Parallel Tempering, we also provide an interface for the Parallel Tempering algorithm of [syed2022](@citet), as implemented by Pigeons.jl. To use Pigeons.jl, we use the `PigeonsSolver()`. We also omit the keyword argument `invtemps_curvature`, because the algorithm optimally chooses the annealing schedule, and the keyword argument `n_swap_rounds`, because the number of swap rounds is deterministically set by the number of rounds.

```@example main
out_pigeons = get_best_policy(
    PigeonsSolver(); 
    initfun = initfun,
    nextfun = nextfun, 
    objfun = objfun, 
    max_invtemp = 50.0,
    n_invtemps = 10,
    n_inner_rounds = 10000)
```

We can compare the optimally chosen inverse temperatures chosen by Pigeons.jl with the ones we created using `invtemps_curvaturre`

```@example main
println(get_invtemps(out))
println(get_invtemps(out_pigeons))
```

And compare the average policy with the output using Pigeons.jl

```@example main
plot_mean_policy(out_pigeons)
```

### Parallelization with Pigeons.jl

Pigeons.jl does not use the Distributed standard library for parallelization. Rather, it uses the MPI distributed computing protocol to spawn sub-processes, and then aggregates those sub-processes together manually. To use this, you must have MPI installed on your machine. See instructions [here](https://webpages.charlotte.edu/abw/coit-grid01.uncc.edu/ParallelProgSoftware/Software/OpenMPIInstall.pdf) to download MPI on Linux. On a computing cluster, you may also need to initialiate your session with MPI-related flags. See [here](https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-mpi/) for instructions on Boston University's computing cluster. 

To use Pigeons.jl's parallelization, we use the `PigeonsMPISolver()` solver, and specify the number of child processes to spawn. If you have additional dependencies needed for your `initfun`, `nextfun`, and `objfun` to work, besides your `Main` module and `HighDimensionalOptimalPolicies`, you need to pass these dependencies to each child process using the `dependencies` keyword argument. You must also have Pigeons.jl installed in your current environment. For instance, we use `StatsBase.sample` along with `LinearAlgebra.dot`, so we pass these modules as dependencies. 

```julia
out_pigeons_mpi = get_best_policy(
    PigeonsMPISolver(); 
    initfun = initfun,
    nextfun = nextfun, 
    objfun = objfun, 
    max_invtemp = 50.0,
    n_invtemps = 10,
    n_inner_rounds = 100,
    n_local_mpi_processes = 2,
    n_threads = 2, 
    dependencies = [StatsBase, LinearAlgebra])
```

## Independent Simulated Annealing runs

An alternative to Parallel Tempering is to simply run many independent runs of a Simulated Annealing algorithm. We do this with the `IndependentSimulatedAnnealingSolver` solver. Here `n_inner_rounds` refers to the number of Metropolis-Hastings steps for a given temperature

```@example main
out_sa = get_best_policy(
    IndependentSimulatedAnnealingSolver(); 
    initfun = initfun,
    nextfun = nextfun, 
    objfun = objfun, 
    max_invtemp = 50.0,
    invtemps_curvature = 2.0,
    n_invtemps = 10,
    n_inner_rounds = 1000,
    n_independent_runs = 500)
```

Note that the average policy shows a smoother decay from the start of the edge vector to the end. This is because, since simulated annealing runs are independent, the algorithm does not get stuck improving particular edges for a long amount of time, as occurs in Parallel Tempering, where policies are generated using an (ergodic) sequence of Metropolis-Hastings draws. 

```@example main
plot_mean_policy(out_sa)
```

## Comparing inverse temperatures

As discussed above (and in the mathematical appendix), there is a tension between mixing and optimality. 

```@example main
function plot_all_objectives(out)
    p = plot()
    invtemps = get_invtemps(out)

    for ind in 1:(length(invtemps))
        obj = get_objective_vec(out; ind)
        label = invtemps[ind]
        density!(p, obj; label = label, line_z = invtemps[ind], zcolor = invtemps[ind], palette = cgrad(:grays))
    end
    p
end
plot_all_objectives(out)
```

## Saving and Reading Across Multiple Independent Runs

Running many solvers in parallel across multiple compute jobs is a time-efficient way to draw a large number of optimal policies. For example, you may want to run an array job in `slurm` where you run the exact same analysis in parallel. 

We provide utilities to save the policy guesses and objective values in a consistent way and read in these values across many independent runs, even if the runs were on separate computing jobs entirely. 

### Running Multiple Independent Runs using Slurm

Imagine you have written a package called MyAnalysisPackage.jl which has HighDimensionalOptimalPolicies.jl as a dependency to analyze a specific policy. 

### Saving output

For any given run, we can save outputs with the `save_policy_output_csv` function. 

```julia
mkdir("tmp_output")
for i in 1:3
    out_temp = get_best_policy(
        PTMCMCSolver(); 
        initfun = initfun,
        nextfun = nextfun, 
        objfun = objfun, 
        max_invtemp = 50.0,
        invtemps_curvature = 2.0,
        n_invtemps = 10,
        n_inner_rounds = 10000,
        n_swap_rounds = 100)

    save_policy_output_csv(out_temp; outdir = "tmp_output", only_max_invtemp = true)
end
```

### Reading Output

The `MultiCSVPolicyOutput` type reads in all `.csv` files in a given  directory and stores them so that you can access the vector of policy guesses and objective values just like you can with other output (i.e. `out` and `out_pigeons` in this tutorial). 

!!! warning
    `save_policy_output_csv` does not validate inputs on writing, and `MultiCSVPolicyOutput` does not validate inputs on reading. It is up to the user to ensure that all inputs to the solver are *the exact same* for all `.csv` files saved.

`MultiCSVPolicyOutput` is a limited object. Unlike other policy ouputs (`MultiMCMCPolicyOutput` etc.) it does not store the underlying functions `initfun`, `nextfun`, or `objfun`. It is only useful for analyzing results, ideally in a session where the exact same `initfun`, `nextfun`, or `objfun` that created the `.csv` files you read in. 

!!! waning
    This function, and the workflow using `.csv` in general, is highly un-optimized. 

```julia
out_csv = MultiCSVPolicyOutput("tmp_output")
```

### Running Independent Jobs using Slurm

When starting an Array job on the cluster, I highly recommend you build a System Image of all the Julia packages you will use when evaluating your policy. For instructions, see documentation [here](https://julialang.github.io/PackageCompiler.jl/dev/sysimages.html). 

Below is an example of using the `-t` option to run an an array job using Slurm by starting multiple independent compute processes. Taken from Boston University's tutorials [here](https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/). This bash script will initiate 25 independent jobs on a machine with 1 core. We can alter the number of cores and tasks depending on the computing resources necessary. 

```zsh
#!/bin/bash -l

# Specify that we will be running an Array job with 25 tasks numbered 1-25
#$ -t 1-25

# Request 1 core for my job
#$ -pe omp 1

# Give a name to my job
#$ -N optimal_policies

# Join the output and error streams
#$ -j y

# Run my julia script 
julia myscript.jl
```

Your `myscript.jl` might look something like this

```julia
using HighDimensionalOptimalPolicies
using MyAnalysisPackage

out = get_best_policy(...)
save_policy_output_csv(oudir = "tmp_output")
```

Then in an additional session, you can read in the results saved by the various tasks in this array job with by calling `MultiCSVPolicyOutput("tmp_output")`. 

### Casting Policy Outputs to DataFrames for Manual Saving

 To convert to DataFrames, use the function `Tables.dictcolumntable` as an intermediate tabular representation of a policy output.  

```julia
df = DataFrame(Tables.dictcolumntable(out))
```

We use an intermediate representation because HighDimensionalOptimalPolicies.jl does not have DataFrames.jl as a dependency. 

## Running Many Parralel Jobs within the same Julia session

All solvers in HighDimensionalOptimalPolicies.jl use `pmap` internally, meaning 



