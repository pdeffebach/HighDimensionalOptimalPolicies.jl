# Mathematical Appendix

## The Standard Optimal Policy Problem

The researcher starts with model economy which is parametrized according to estimated parameters ``\hat{\theta}``. Denote ``N \in \mathcal{N}`` to be a "policy" within the the model. For any given pair of parameter estimates ``(\hat{\theta}, N)`` we can generate a welfare level ``W(N; \hat{\theta}) \in \mathbb{R}``. In this way, we can compare policies ``N, N' \in \mathcal{N}``. 

For example, the consider a model of the transportation throughout the city, where ``\hat{\theta}`` represents, among other things, the preferences of households for taking the car vs. the bus, or the speed of various transportation modes. In this example, ``\mathbb{N}`` represents the set of public transportation routes throughout the city. 

The researcher wishes to advise the policy-maker on the optimal policy. That is, they want to find

```math
N^* \in \arg \max_{N} W(N; \hat{\theta})
```

Consider the case where ``N`` is high-dimensional or otherwise difficult to characterize and where ``\mathcal{N}`` either a very large discrete set or an otherwise large continuous space. In this scenario, we face two main constraints. 
1. he high dimensionality of ``N`` can make conventional descent-based optimization methods prohibitively computationally expensive. 
2. The large state space of ``\mathcal{N}`` makes it difficult to ensure the we have correctly identified the global best policy ``N^*`` as opposed to one of many local optima. 

## The Relaxed Optimal Policy Problem

To solve these issues, we re-characterize the optimal policy problem such that our optimal policy ``N^*`` is now defined as
```math
N^* \in \arg \max W(N; \hat{\theta}) + \epsilon_N
```
where ``\epsilon_N`` is an i.i.d extreme value type-I distribution with dispersion parameter ``\beta``. That is, a ``\text{Gumble}(\beta^{-1})``.

In this relaxed problem, ``\epsilon_N`` is unobserved by the researcher. This might represent, for example, social welfare factors that are not in the model, or idiosyncratic policy effects that are not observed by the researcher. In the relaxed problem, ``\beta``, the dispersion of errors ``\epsilon_N`` is also unobserved by the researcher and the researcher must make assumptions about its value. 

This re-characterization of the optimal problem now implies that, from the researcher's perspective, *any* policy ``N \in \mathcal{N}`` could be the optimal policy ``N^*``, given a high enough unobserved value of the idiosyncratic shock ``\epsilon_N``. As a consequence, the researcher is no longer in searching for the *single* optimal policy, now seeks to characterize policies by the *probability* that a given policy is optimal. 

Given standard results about multinomial logit probabilities, we define the probability that a given policy ``N`` is optimal as

```math
\mathbb{P}(N \text{optimal}) = \pi_\beta(N) = \frac{\exp{\beta W(N; \hat{\theta})}}{\sum_{N' \in \mathcal{N}} \exp{\beta W(N'; \hat\theta)}}
```

Our goal, then, is to estimate ``\pi_\beta(N)``. 


## The Metropolis-Hastings Algorithm for Calculating Optimal Policies

Given our constraints, listed above, that ``N`` is both high dimensional and ``\mathcal{N}`` is a large set, it is intractable to compute or estimate the probabilities ``\pi_\beta(N)`` explicitly. To compensate for this, instead of analyzing features of ``\pi_\beta(N)``, we generate a set of "likely optimal policies"

Instead, our goal is to sample optimal policies from the distribution ``\pi_\beta(N)`` without fully characterizing ``\pi_\beta(N)``. We accomplish this through the Metropolitan-Hastings algorithm. 

At a high level, the Metropolitan-Hastings algorithm is a Markov Chain Monte-Carlo algorithm for sampling from a probability distribution which is difficult to characterize. It is an iterative procedure which takes as an input an initial "guess" of an initial Markov Chain and over time will characterize a Markov Chain whose stationary distribution corresponds to the distribution of interest. The exposition of this section borrows from Levin and Peres (2017).

Fixing ideas, consider an initial Markov chain ``\Psi`` which is both aperiodic and irreducible, and satisfies ``\Psi(N, N') > 0 \iff \Psi(N', N) > 0``. Begin with network ``N_1``. Given network ``N_s`` and step ``s``, draw a candidate network ``N'`` from the initial distribution ``\Psi(N' \mid N_s)``. This candidate network ``N'`` becomes ``N_{s+1}`` with probability given
```math
\mathbb{P}(N_{s+1} = N' \mid N_s) = \min \left(1, \frac{\exp(\beta W(N'))\Psi(N_s \mid N')}{\exp(\beta W(N_s))\Psi(N' \mid N_s)}\right)
```

To understand this expression, examine the case where ``\Psi(N, N') = 1 \text{ for all } N, N' \in \mathcal{N}``. If ``W(N') > W(N)``, then ``N'`` is accepted. If ``W(N') < W(N)``, then it is accepted with a probability that is increasing in ``W(N')``. 

As ``s \rightarrow \infty``, then ``\mathbb{P}(N_s = N) \rightarrow \pi_\beta(N)``.  

## Simulated Annealing and Parallel Tempering for Metropolis-Hastings

A downside of Metropolis-Hastings algorithms a lack of "mixing". That is, we the algorithm may repeatedly draw the same sample over and over again. This problem is particularly severe in our context because of the functional form of our modified objective function, ``e^{\beta W(N)}``. For high values of ``\beta``, even a marginall smaller of value of ``W(N)`` will have an almost zero chance of being accepted, and the Metropolis-Hastings algorithm will be stuck in a local optimal. 

Lower values of ``\beta`` will lead the Metropolis-Hastings algorithm to be less likely to stuck in a local optima. However, since marginally smaller valus of ``W(N)`` are more likely to be accepted with a small ``\beta``, meaning the algorithm will be less likely to find optimal policies at all. The trade-off is then

* High values of ``\beta`` are more likely to find optimal policies, but also more likely to get stuck in a local optima and not accurately characterize the state space of optimal policies. 
* Low values of ``\beta`` traverse the state space of optimal policies, but the policies it draws are less likely to be optimal in general. 

The solution to this problem is a procedure that combines low-``\beta`` and high-``\beta`` runs of the Metropolis-Hastings algorithm. Low-``\beta`` runs of the algorithm traverse the state space and search for optimal policy candidates. Later on, the high-``\beta`` runs of the algorithm take up these optimal policy guesses and draw from the state space near these candidates. 

We accomplish this general goal in two ways, with a Simulated Annealing Algorithm and with the Parallel Tempering algorithm. 

### Simulated Annealing

Simulated Annealing takes as an input a schedule of inverse temperatures ``\beta_1 < \beta_2 < \mathellipsis < \beta_K``. Starting with ``\beta_1``, we run ``S`` iterations of the Metropolis-Hastings algorithm with inverse temperature ``\beta_1``. Because ``\beta_1`` is relatively low, this algorithm quickly traverses the state space and after ``S`` iterations returns a candidate network ``N_1``. 

Next, we use this candidate network ``N_1`` as the initial state for another ``S`` runs of the Metropolis-Hastings algorithm, this time using inverse temperature ``\beta_2``. This run will traverse the state space of policies less quickly, and will spend more time on optimal policies. This run returns candidate policy ``N_2``, and so on, until the we return policy ``N_k``. We are confident ``N_k`` is a globally optimal policy because previous iterations of the algorithm have thoroughly explore the state space. 

To get a set of optimal policies, we run many independent runs of the Simulated Annealing algorithm. 

### Parallel Tempering

Parallel Tempering might be thought of as running simultaneous Simulated Annealing algorithms in parallel. We implement a textbook (naive) version of Parallel Tempering, as well as provide bindings to a more sophisticated version of the Algorithm, implemented by [Pigeons.jl](https://pigeons.run/stable/), which implements the algorithm described by [syed2022](@citet).

Like Simulated Annealing, Parallel Tempering takes as an input a schedule of inverse temperatures ``\beta_1 < \beta_2 < \mathellipsis < \beta_K``. It then runs separate Metropolis-Hastings algorithms for each temperature in parallel. 

After ``S`` runs, each algorithm returns their latest policy candidate ``N_k`` and a "swapping" stage occurs. Inverse temperatures are "paired up" such that ``N_1`` is compared to ``N_2``, ``N_3`` is compared to ``N_4``, etc. For each comparison ``k`` to ``k+1`` decide whether or not the two inverse temperatures should swap policies according to 

```math
\mathbb{P}(N_{k+1}' = N_{k}, N_{k}' = N_{k+1}) = \min\left\{1, \exp\left(\left(\beta_{k+1} - \beta_k\right) \times \left(W(N_{k}) - W(N_{k+1})\right)\right)\right\}
```

Recall that ``\beta_{k_1} - \beta_k > 0``. So when, by chance, the lower temperature returns a more optimal policy ``N_k``, it will be accepted. Otherwise, the lower temperature policy will be accepted according to a probability determined by the ``\beta_{k+1} - \beta_k``. 

Including a Metropolis-Hastings swap stage after every ``S`` iterations in should result in a high degree of mixing, such that the Parallel Tempering algorithm can run indefinitely and draws from the time series of optimal policies produced by this algorithm mirror the distribution of optimal policies more generally. 

The above describes our "naive" implementation Parallel Tempering. Please see [syed2022](@citet) for a discussion of their more sophisticated algorithm. In particular, the authors do not take the inverse temperature schedule as given. Instead, we only assign a maximum inverse temperature, ``N_K`` and the schedule is determined optimally to ensure mixing. 

## Testing for optimal mixing

Given a set of policies ``\{N_l\}_{l =1}^{L}``, how can we be sure that this set of policies is drawn from the the distribution of optimal policies ``\pi_\beta``? We propose the following test statistic

```math
\hat{T} = \hat{W}_{\max} - \hat{W}_{\text{mean}} - \frac{\log n}{\beta}
```

Where ``\hat{W}_{\max}`` and ``\hat{W}_{\text{mean}}`` are the maximum and mean values of the the welfare values from the observed set of policies, ``\{W(N_l)\}_{l =1}^{L}``, respectively, ``n`` represents the size of the state space (i.e. the number of different policies which can be selected), and ``\beta`` is the inverse temperature associated with this set of draws. 

Asymptotically, ``\hat{T}`` is bounded above by the normal distribution of ``\mathcal{N}\left(0, \frac{\sigma_\pi}{L}\right)``. Therefore, to test whether ``\{N_l\}_{l =1}^{L}`` are drawn independently from ``\pi_\beta``, we use the one-sided t-test based on

```math
\frac{\hat{T}}{\hat{\sigma}_\pi / \sqrt{L - 1}}
```

where ``\hat{\sigma}_\pi`` is the estimated standard deviation of welfare values.  


