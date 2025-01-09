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

Fixing ideas, consider an initial Markov chain ``\Psi`` which is both aperiodic and irreducible, and satisfies ``\Psi(N, N') > 0 \iff \Psi(N', N) > 0``. Begin with network ``N_1``. Given network ``N_k`` and step ``k``, draw a candidate network ``N'`` from the initial distribution ``\Psi(N' \mid N_k)``. This candidate network ``N'`` becomes ``N_{k+1}`` with probability given
```math
\mathbb{P}(N_{k+1} = N' \mid N_k) = \min \left(1, \frac{\exp(\beta W(N'))\Psi(N_k \mid N')}{\exp(\beta W(N_k))\Psi(N' \mid N_k)}\right)
```

To understand this expression, examine the case where ``\Psi(N, N') = 1 \text{ for all } N, N' \in \mathcal{N}``. If ``W(N') > W(N)``, then ``N'`` is accepted. If ``W(N') < W(N)``, then it is accepted with a probability that is increasing in ``W(N')``. 

As ``k \rightarrow \infty``, then ``\mathbb{P}(N_k = N) \rightarrow \pi_\beta(N)``.  







