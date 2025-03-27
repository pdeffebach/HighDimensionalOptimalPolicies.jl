using Pigeons


#=
Inputs is a struct used to create parallel tempering
algorithms.

* target: The target distribution
* seed: The master random seed
* n_rounds: the number of rounds to run
* n_chains: The number of chain to use
  (but there is also a heuristic, n_chains_variational)
* n_chains_variational: Something about the variational
  step. TODO: Look into this
* reference: the prior distribution. For the most part
  this should be automatically chosen based on the target distribution.
* variational: Something about variational inference
* checkpoint: Whether a checkpoint should be written to disk at the
  end of each round.
* record: Determine what should be stored from the simulation. A
  Vector with elements of type recorder_builder
* checked_round: The round index where `run_checks()` will be performed
  TODO: Understand what `run_checks()` means.
* multitheraded: If multithreaded explorers should be allowed.
* explorer: Unclear what "explorer" means.
* extractor: Passed to `extract_sample` and `sample_names` to determine
  how samples should be extracted for `traces`. None of this means anything
  to me. One example is `LogPotentialExtractor`, whatever that means.
* show_report: Some sort of logging function?
* extended_traces: Store all the chains or just the lowest temperature one?
=#
inputs = Inputs(target = toy_mvn_target(100))

#=
This "solves" the input. This has basically no meaning to me.
=#
pt = pigeons(inputs)

#=
You can increment the rounds for the output, performing more chaining.
Except this doesn't actually do the chaining.
=#
pt = increment_n_rounds!(pt, 2)

#=
The class of problems that can be solved using pigeons. In general,
the class of problems is "computational lebesque integration"



Let $\pi(x)$ denote a probability density called the \emph{target}. In
many problems, e.g. in Bayesian Statistics, the density $\pi$ is
typically known only up to a normalization constant.

```math
\pi(x) &= \frac{\gamma(x)}{Z} ```math

Pigeons takes as input the function $\gamma$. Pigeons can be used for
two tasks.

First, to approximate the function $\mathbb{E}(f(X))$ where
$X \sim \pi$.

Second, approximating the value of the normalization constant $Z$, for
example, in Bayesian statistics, this usually corresponds to the
marginal likelihood.

Note: The package *does not* say that it is useful for sampling from
$\pi$, which is what we want to do. On the other hand, you kind of
need to do this for $\mathbb{E}(f(X))$ so maybe it will be useful.
=#

## More on Paralell Tempering

#=
Consider a Bayesian model where the likelihood is a binomial
distribution with parameter $p$. Let us consider an *over-parametrized*
model where we write $p = p_1 p_2$

When there are many observations, the posterior of unidentified
models concentrate on a sub-manifold, making sampling difficult.

I think this means that there is a set of values $\{p_1, p_2\}$ that
act as solutions. You can see this from the pair plot.
=#

using DynamicPPL
using Pigeons
using MCMCChains
using StatsPlots
gr()


# The model described above implemented in Turing
# note we are using a large observation size here
an_unidentifiable_model = Pigeons.toy_turing_unid_target(100000, 50000)

#=
traces is a function for saving all the outputs of a chain process.
=#
pt = pigeons(
    target = an_unidentifiable_model,
    n_chains = 1, # <- corresponds to single chain MCMC
    record = [traces])

# collect the statistics and convert to MCMCChains' Chains
samples = Chains(pt)

# create the trace plots
my_plot = StatsPlots.plot(samples)
#StatsPlots.savefig(my_plot, "no_pt_posterior_densities_and_traces.html");

#=
The mixing is poor, as evidenced by the Effective Sample Size
(ESS) estimates. Here is looks lke ess_bulk is 2.8, and ess_tail are
around 30. maybe these are low?
=#
samples

#=
Okay, now the authors have shown an instance where a conventional
MCMC with one chain does poorly. Now they show the same example
where they use PT.
=#

pt = pigeons(
    target = an_unidentifiable_model,
    n_chains = 10,
    record = [traces, round_trip])

# collect the statistics and convert to MCMCChains' Chains
samples = Chains(pt)

# create the trace plots
my_plot = StatsPlots.plot(samples)

#=
The authors argue this is a big difference. I agree! But it's hard
to tell exacly what they are saying. We should expect the values of
p_1 and p_2 to mutually cover a large swath of values, that is, the
set of values {p_1, p_2} where p_1 * p_2 = p. This plot shows that
indeed, the PT algorithm can hit that large set of values
(sub-manifold)
=#

## Paralellization

#=
Now the authors explore how Pigeons makes it easy to parallelize.
your estimation.
=#

using Pigeons
pigeons(
    target = toy_mvn_target(100),
    n_chains = 2,
    multithreaded = true)


## Variational PT

#=
Variational PT is described in a paper. I have no idea what it really
is, still.
=#

## Inputs overview

#=
Now the authors finally explain what this "input" thing is.

It takes as input an expectation or integration problem. As input you
can give it

* A Turing.jl model, which is a succing specification of a joint
  distribution from which a posterior and prior are extracted. That
  is, the model encodes both the posterior and prior.
* A black-box Julia function. This is probably what we want,
  but its not clear still what this means.
* A Stan model. For adapting existing Stan code.
* MCMC code implemented in another language. This sounds complicated!
  I'd be curious to see how this works but it's not relevant for
  us at the moment.
* You can use a custom explorer. This seems like what we want, but
  it's not clear what it really means still, just like the black-box
  Julia code stuff.
=#

## Turing

#=
Let's explore using a Turing.jl model, for the sake of completeness.

The model they use is the same un-identified binomial model they
used before.

The priors for p1 and p2 are uniform in (0, 1). The outcome
is distributed Binomial(n_trials, p1 * p2)
=#

DynamicPPL.@model function my_turing_model(n_trials, n_successes)
    p1 ~ Uniform(0, 1)
    p2 ~ Uniform(0, 1)
    n_successes ~ Binomial(n_trials, p1*p2)
    return n_successes
end

#=
Here we wrap our model in TuringLogPotential. The documentation
of TuringLogPotential says that this uses Turing's backend to
construct the log density.

This just means it's using an optimized way of calculating the
log likelihood (I think).
=#
my_turing_target = TuringLogPotential(my_turing_model(100, 50))

pt = pigeons(target = my_turing_target, record = [traces])

#=
You need `record = [traces]` to make a plot. It looks like the
algorithm succeeds at having p1 and p2 be very random.
=#
samples = Chains(pt)
my_plot = StatsPlots.plot(samples)

#=
You can use AD backends when yur model has a fully continuous
state space. There is a LogDensityProblemsAD.jl interface
that they opt into. This isn't relevant for us.
=#

#=
There is a way to do custom initialization, for example to
start in a feasible region.

The following code creates our familiar under-identified binomial
model.

Next it generates the target, using our TuringLogPotential that
we were introduced to above.

This is a parametrized type and the way Pigeons.jl works is that
it dispatches on the type of the target. So we overwrite the method
for it's initialization function. I have no idea what the first
two lines of that function do. Presumably they are boilerplate
that is required in all initialization functions.

Finally it solves the model
=#

using DynamicPPL, Pigeons, Distributions, Random

DynamicPPL.@model function toy_beta_binom_model(n_trials, n_successes)
    p ~ Uniform(0, 1)
    n_successes ~ Binomial(n_trials, p)
    return n_successes
end

function toy_beta_binom_target(n_trials = 10, n_successes = 2)
    return Pigeons.TuringLogPotential(toy_beta_binom_model(n_trials, n_successes))
end

const ToyBetaBinomType = typeof(toy_beta_binom_target())

function Pigeons.initialization(target::ToyBetaBinomType, rng::AbstractRNG, ::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)

    # custom init goes here: for example here setting the variable p to 0.5
    Pigeons.update_state!(result, :p, 1, 0.5)

    return result
end

pt = pigeons(target = toy_beta_binom_target(), n_rounds = 0)
@assert Pigeons.variable(pt.replicas[1].state, :p) == [0.5]

# Julia code as input to pigeons

#=
This might be close to what we want! Let's re-implement the
un-identified black-box model manually.

Start by defining the struct containing our "data"
=#

using Pigeons
using Random
using Distributions

struct MyLogPotential
    n_trials::Int
    n_successes::Int
end

#=
Next, we define our log_potential variable. This is the log-pdf
of the outcome.
=#
function (log_potential::MyLogPotential)(x)
    p1, p2 = x
    if !(0 < p1 < 1) || !(0 < p2 < 1)
        return -Inf64
    end
    p = p1 * p2
    return logpdf(Binomial(log_potential.n_trials, p), log_potential.n_successes)
end

my_log_potential = MyLogPotential(100, 50)

#=
The last Int here, and above, is for the "replica index", which
probably has something to do with different temperatures for
different chains.
=#
Pigeons.initialization(::MyLogPotential, ::AbstractRNG, ::Int) = [0.5, 0.5]

#=
Now we solve it. Notice that we have to provide a reference (prior)
distribution. In this case, its the model with no observations.

The default "explorer" (I still don't know what this is yet), is
something called a `SliceSampler`. Looking at the reference
for a slice sampling, it has something to do with using the
currently-available information on the posterior to choose
the next guess.
=#
pt = pigeons(
        target = MyLogPotential(100, 50),
        reference = MyLogPotential(0, 0)
    )

# Sapmling from the reference distirbution

#=
Yeah! This is important, and exactly what we want. Here they implement
a way to sample from the reference distribution using a black-box
function.
=#

#=
Thought: Why is this mutating? I think this is just mutating the
input state.

*Which is exactly what we want!* I think this is key here. Currently,
it's just implementing the Uniform(0, 1) kind of drawing,
but we could change this to, say, be Binomial.

Maybe this is when it *doesn't* want any dependence on the existing
posteriors (hence the i.i.d)? I wish this worked.
=#
function Pigeons.sample_iid!(::MyLogPotential, replica, shared)
    state = replica.state
    rng = replica.rng
    # This doesn't seem to restrict p1 and p2 at all.
    p1 = rand(rng, Uniform(0.8, 1))
    p2 = rand(rng, Uniform(0.8, 1))
    state .= [p1, p2]
end


pt = pigeons(
        target = MyLogPotential(100, 50),
        reference = MyLogPotential(0, 0),
        record = [traces]
    )

samples = Chains(pt)
my_plot = StatsPlots.plot(samples)

#=
Maybe what we really want to do is change the explorer? So far,
it's been unclear what "explorer" means.

Actually, it's pretty unclear what they need the gradient for.
=#

#=
The previous section on explorer was about giving a custom gradient
to the explorer, or changing the backend AD.

What we really want to do is write our own function entirely. Maybe
this section tells us how.

> However when the state space is neither the reals nor the integers,
  or for performance reasons, it may be necessary to create custom
  exploration MCMC kernels.

Let's start by defining a custom struct for the sampler.

I have no idea what `which_parameter_index` might mean here.
I guess it means which element of the state vector we are
thinking about updating?
=#
struct MyIndependenceSampler
    which_parameter_index::Int
end

#=
Now let's define the step function.

It looks like `log_potential` is hard-coded here. I guess we are
also looking at a log potential though.

This looks like the right move. Modifying the `step` function
is the way to go.

This is the key function!
=#
function Pigeons.step!(explorer::MyIndependenceSampler, replica, shared)
    state = replica.state
    rng = replica.rng
    # Note: the log_potential is an InterpolatedLogPotential between the target and reference
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    log_pr_before = log_potential(state)
    # propose
    state_before = copy(state)
    for i in 1:length(state)
        state[i] = rand(rng, Uniform(0.8, 1))
    end
    log_pr_after = log_potential(state)
    # accept-reject step
    accept_ratio = exp(log_pr_after - log_pr_before)
    if accept_ratio < 1 && rand(rng) > accept_ratio
        # reject: revert the move we just proposed
        state .= state_before
    end # (nothing to do if accept, we work in-place)
end


pt = pigeons(
        target = MyLogPotential(100, 50),
        reference = MyLogPotential(0, 0),
        explorer = Compose(MyIndependenceSampler(1), MyIndependenceSampler(2)),
        record = [traces]
    )

samples = Chains(pt)
my_plot = StatsPlots.plot(samples)

## Manipulating the output of Pigeons

#=
Automated reports is a julia package to generate a web page or pdf
from pigeons output.

It makes a lot of graphs I don't understand.
=#

#=
Here is some information for how to interpret the tables that print
when you call `pigeons`.

* scans: Not explained.
* Λ: The global communication barrier. As described in the
  linked paper.
* time and allc, the time and allocation used in each round
* log(Z₁/Z₀): The stepping_stone estimator for the log normalization
  constant.
* min(α): The minimum swap acceptance rates over the PT chains
* mean(α): The average swap acceptance rate over the PT chains
=#

## Saving traces

#=
The `traces` refer t the list of samples X_1, X_2, ... X_n from
which we can approximate expectations of the form $E(f(x))$.

To indicate that the traces should be saved, use
=#

using DynamicPPL
using Pigeons

target = Pigeons.toy_turing_unid_target(100, 50)

pt = pigeons(;
    target,
    n_rounds = 3,
    # make sure to record the trace:
    record = [traces; round_trip; record_default()])

#=
You can access the samples used in the chain with the function
`get_sample`. This returns a vector of vectors corresponding to
the parameters (p1, p2) and the log density.
=#
get_sample(pt)

#=
You can also return it as a big matrix.

Actually, it's not a matrix, but rather a 3-dimensional array.
This probably corresponds to what chain you are focusing on.
=#
sample_array(pt)

#=
You can customize the elements you save in a trace with various
extractor types, for instance OnlyFirstExtractor
=#

## Plotting

#=
As we've seen so far, there are a number of default plots you
can make with the samples derived by `pt`. Here we explore
further plotting features.
=#
using DynamicPPL
using Pigeons
using MCMCChains
using StatsPlots
gr()

# example target: Binomial likelihood with parameter p = p1 * p2
an_unidentifiable_model = Pigeons.toy_turing_unid_target(100, 50)

pt = pigeons(
    target = an_unidentifiable_model,
    n_rounds = 12,
    # make sure to record the trace:
    record = [traces; round_trip; record_default()])

# collect the statistics and convert to MCMCChains' Chains
# to have axes labels matching variable names in Turing and Stan
samples = Chains(sample_array(pt), sample_names(pt))

# since the above line is frequently needed, Pigeons includes
# an MCMCChains extension allowinging you to use the shorter form:
samples = Chains(pt)

# create the trace plots
my_plot = StatsPlots.plot(samples)

#=
What about the log density?. We use the MCCChains API to get
more information
=#
params, internals = MCMCChains.get_sections(samples)

my_plot = StatsPlots.plot(internals)

## Approximation of the normalization constant

#=
In Bayesian statistics, we are often interested in the denominator
of the expression

```math
\pi(x) = \frac{\gamma(x)}{Z}
```

In Bayesian statistics, this corresponds to the marginal likelihood.

Fortunately, this likelihood is automatically calculated in the
process of the parallel tempering algorithm.
=#
using DynamicPPL
using Pigeons

# example target: Binomial likelihood with parameter p = p1 * p2
an_unidentifiable_model = Pigeons.toy_turing_unid_target(100, 50)

pt = pigeons(target = an_unidentifiable_model,
    record = [traces; round_trip; record_default()])

#=
The function `stepping_stone` returns log(Z)
=#

## Numerical outputs and diagnostics

#=
We can use the function `samples` to get a variety of metrics about
our MCMC.
=#

#=
Given samples from MCMC, we can also take the `mean` and
index into that
=#
samples = Chains(pt)

## Online statistics

#=
When the dimensionality of a model is large, or the number of MCMC
samples is large, the samples may not fit in memory. We can write
these samples to disk and process them one at a time. This might
seem burdensome, and it is.

Fortunately, many statistics can be calculated by updated
themselves iteratively (like mean)

This process uses OnlineStats.jl
=#

## Off-memory statistics

#=
You can write the record to a temp file. Basically, the `pt` saves
a reference to the file that stores everything. Then you can loop
through parts of the file and extract the needed elements.
=#

## Diagnostics of the PT algorithm

#=
This section isn't very useful if I haven't read the Syed paper.
Stuff like the "global communication barrier" (used to set the number
of chains).

And a "tempered restart" which "happens when a sample from the
reference percolates to the target." I have no idea what
that means.
=#

## Custom types

#=
This might be close to the meat of what we want. It discusses how
to post-process samples when the states are not real or integer
vectors.

As a concrete example, consider an implementation of an Ising model
where a state contains a matrix of binary variables as well as
some other caches.

This is useful for us in the sense that we want to make nice
plots. The solution is to overload the function
`Pigeons.extract_sample`. Here they just do copy(vec(...))
=#

## Extended outputs

#=
So far, we have just been interested in the "target distribution",
that is, the one with the lowest temperature. But we can also
show the output from other chains.

You can see that we are using the `extended_traces = true` option,
below, which controls whether we return all chains or not.
=#

# example target: Binomial likelihood with parameter p = p1 * p2
an_unidentifiable_model = Pigeons.toy_turing_unid_target(100, 50)

pt = pigeons(
    target = an_unidentifiable_model,
    n_rounds = 12,
    extended_traces = true,
    # make sure to record the trace:
    record = [traces; round_trip; record_default()])

# collect the statistics and convert to MCMCChains' Chains
# to have axes labels matching variable names in Turing and Stan
samples = Chains(pt)

#=
You can also use off-memory processing for all the chains. Here
you modify the for-loop (which I haven't expressed here or above)
to also look through the chains.
=#

#=
Importantly, you can access the annealing parameters.

This is going to be very important for us. In particular, don't
we want to select our temperature ourselves? It's not clear
how to set this.
=#
pt.shared.tempering.schedule

## Post-processing for MPI runs

#=
This section will be relevant in the later stages of
implementation, but not now.
=#

## Checkpoints

#=
Pigeons can write a "checkpoint" periodically to ensure that
not more than half of the work is lost in the event of, e.g.
a server failure.

The function that calls this is write_checkpoint()

A checkpoint is a *full* accounting of the entire chain in
a way that can be reproduced.
=#

## Correctness

#=
This describes some ways to check correctness of your algorithm
when it is run in multi-vs-single threaded contexts.
=#

## More on Parallel Tempering

#=
This is an overview of Non-Reversible Parallel Tempering (PT) in
the Syed paper, which I will read shortly.


Let $X_n$ denote a Markov chain on state space $\mathcal{X}$ with
stationary distribution $\pi$.

PT is a Markov chain defined on the augmented state space
$\mathcal{X}^N$. hence a state has the form
```math
\mathbf{X} = (X^(1), X^(2), ... X^(N))
```

Each component of $\mathbf{X}$ is stored in a struct called a
`Replica`. The storage of the vector of replicas $\mathbf{X}$ is
done with the informal interface `replicas`.

Internally, PT operates on a discrete set of distributions
$\pi_1, \pi_2, ..., \pi_N$.

Typically, $\pi_N$ coincides with the distribution of interest, $\pi$
called the "target". While $\pi_1$ is a tractable apprximation that
will help PT efficiently explore the state space (makes large jumps).

The stationary distribution of $\textbf{X}$ is
```math
\textbf{\pi} = \pi_1 \times \pi_2 \times ... \pi_N
```
As a result, subsetting each sample to its component corresponding to
$\pi_N = \pi$ and applying an integrable function $f$ to each will,
under weak assumptions, appropriately measure $\mathbb{E}(f(X))$.

I think the paragraph above means that each chain on its own can
be used to approximate $\pi$

The PT alternates between two stages: The local exploration
phase and the communication phase.

Informally, the first phase attempts to achieve mixing for
the univariate statistics $\pi_i(X^(i))$ while the second
phase attempts to translate the well-mixing of these univariate
stateistics into the global mixing of $X^(i)$.

Let's talk about the first stage, local exploration. In this stage,
each `Replica`'s state is modified using a $\pi_i$-invariant kernel.

I'm not sure what $\pi_i$-invariant means. In think it means that
no matter what $\pi_i$ looks like, the draw for the next candidate
state is the same.

Now let's discuss the communication phase. This phase proposes
swaps between pairs of replicas. These swaps allow each replica's
state to periodically visit reference chains. During these reference
visits, the state can move around the state space quickly.

There are equivalent ways to do a swap. You can swap the state field
or they could exchain their chain fields. The latter is what
this package uses, since the data exchange can be very small.

Local explorers:

Typical target distributions are expected to take care of building
their own explorers. So most users are not expected to have to
write their own.

But for a non-standard target (like ours), it is useful.

Suppose you are planning to use a non-standard target of type
`MyTargetType` (this is what we are planning to do).

Create a `MyExplorerStruct` that may contain adaption information
such as step sizes for HMC or proposal bandwith (is this the variance
of the normal distribution you draw from in most sampling procedures?)

Implement all the methods in the section "contract" of `explorer`,
making sure to type the explorer arguments as
`explorer::MyExplorerStruct`. There are examples.

Define a method `default_explorer(target::MyTargetType)` which
should return a fresh `MyExplorerStruct` instance.

One explorer struct will be shared by all threads, so it should be
read-only during execution of `run_one_round!()`.

Tempering:

This feels very important for our purposes. Tempering is done through
the step `communicate!`, and adapting it follows the same general
proces as custom explorers. You make a struct, then implement
all methods of `tempering`, and set a default construction.
=#

## Next, the authors discuss details for distrbuted-ness



