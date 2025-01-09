# Mathematical Appendix

## The Standard Optimal Policy Problem

The researcher starts with model economy which is parametrized according to estimated parameters $\hat{\theta}$. Denote $N \in \mathcal{N}$ to be a "policy" within the the model. For any given pair of parameter estimates $(\hat{\theta}, N)$ we can generate a welfare level $W(N; \hat{\theta}) \in \mathbb{R}$. In this way, we can compare policies $N, N' \in \mathcal{N}$. 

For example, the consider a model of the transportation throughout the city, where $\hat{\theta}$ represents, among other things, the preferences of households for taking the car vs. the bus, or the speed of various transportation modes. In this example, $\mathbb{N}$ represents the set of public transportation routes throughout the city. 

The researcher wishes to advise the policy-maker on the optimal policy. That is, they want to find

```math
N^* \in \arg \max_{N} W(N; \hat{\theta})
```

Consider the case where $N$ is high-dimensional or otherwise difficult to characterize and where $\mathcal{N}$ either a very large discrete set or an otherwise large continuous space. In this scenario, we face two main constraints. 
1. he high dimensionality of $N$ can make conventional descent-based optimization methods prohibitively computationally expensive. 
2. The large state space of $\mathcal{N}$ makes it difficult to ensure the we have correctly identified the global best policy $N^*$ as opposed to one of many local optima. 

## The Relaxed Optimal Policy Problem

To solve these issues, we re-characterize the optimal policy problem such that our optimal policy $N^*$ is now defined as
```math
N^* \in \arg \max W(N; \hat{\theta}) + \epsilon_N
```
where $\epsilon_N$ is an i.i.d extreme value type-I distribution with dispersion parameter $\beta$. That is, a $\text{Gumble}(\beta^{-1})$.

In this relaxed problem, $\epsilon_N$ is unobserved by the researcher. This might represent, for example, social welfare factors that are not in the model, or idiosyncratic policy effects that are not observed by the researcher. In the relaxed problem, $\beta$, the dispersion of errors $\epsilon_N$ is also unobserved by the researcher and the researcher must make assumptions about its value. 

This re-characterization of the optimal problem now implies that, from the researcher's perspective, *any* policy $N \in \mathcal{N}$ could be the optimal policy $N^*$, given a high enough unobserved value of the idiosyncratic shock $\epsilon_N$. As a consequence, the researcher is no longer in searching for the *single* optimal policy, now seeks to characterize policies by the *probability* that a given policy is optimal. 

Given standard results about multinomial logit probabilities, we define the probability that a given policy $N$ is optimal as

```math
\mathbb{P}(N \text{optimal}) = \pi_\beta(N) = \frac{\exp{\beta W(N; \hat{\theta})}}{\sum_{N' \in \mathcal{N}} \exp{\beta W(N'; \hat\theta)}}
```

Given our constraints, listed above, that $N$ is both high dimensional and $\mathcal{N}$ is a large set, it is intractable to compute or estimate these probabilities explicitly. 





