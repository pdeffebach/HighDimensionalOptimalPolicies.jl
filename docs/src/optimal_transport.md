## Example: Optimal Transportation Policy

This document outlines the set-up for the optimal transport problem 
implemented in the `examples` folder. It shows a more serious 
mathematical problem that is a good candidate for this package. 

### Set-up

Consider a city defined as a weighted graph with ``N`` nodes and a road
network defined as ``E`` edges (roads) between nodes. The weight of the
graph represents travel time between nodes. If nodes ``i`` and ``j`` have
a road connecting them, the travel time for the direct route between
the two nodes is given by ``t_{ij}``. If nodes ``i`` and ``j`` do *not*
have a road connecting them, then there is no direct route and a
traveler is forced to take an indirect route (however an indirect
route may still be faster than a direct route).

An individual lives in node ``i`` and works in node ``j``. All home and
workplace decisions are exogenous. Denote the fraction of individuals
commuting from location ``i`` to location ``j`` as ``p_{ij}``.

However an individual still
chooses the *route* they take from home to work. That is, they choose
what node-to-node-to-node route ``\mathfrak{R}_{ij}`` they take between
their home ``i`` and work ``j``. 

We borrow the decision problem and solution for a worker's optimal
route from [allen2022a](@citet). In brief, workers seek the route
with the shortest travel time, but also receive i.i.d. Frechet
preference shocks over potential routes.

[allen2022a](@citet) show that for a given ``i``-``j`` home-workplace
pair, the effective expected travel cost faced by the worker,
accounting for both travel time and the idiosyncratic preference
shock, can be expressed as 

```math
\tau_{ij} = \left(\sum_{r \in \mathfrak{R}_{ij}} \left(\prod_{l = 1}^{L} t^{-\theta}_{r_{l-1}, r_l}\right)\right)^{-\frac{1}{\theta}}
```
Where ``t_{r_{l-1}, rl}`` corresponds to the node-to-node travel time between leg ``l-1`` and step ``l`` on the route. 

Moreover Allen and Arkolakis show that there is a convenient matrix expression for this travel time as well. Let ``\mathbb{A} = [t_{ij}^{-\theta}]``, then define ``\mathbf{B}`` as the Leontief inverse of ``A``

```math
\mathbf{B} = (I - A)^{-1}
``` 
and we thus have

```math
\tau_{ij} = b_{ij}^{-\frac{1}{\theta}}
```

The planner cares about the total cost of travel faced by all
individuals in the city. Thus their welfare function is

```math
W\left(\{\tau_{ij}\}_{i,j\in N}\right) = -\left(\sum_{i,j \in N}\tau_{ij}\pi_{ij}\right)
```

Let ``\mathbf{T}`` be an ``N \times N`` matrix representing the travel
time between any two nodes, and let ``\alpha`` be the shape parameter of
the idiosyncratic Frechet preference shock for any route.

### The Policy Space

The planner has the budget to improve ``K < E`` roads between nodes by
reducing travel speed on a given road. They need to choose *which*
roads they want to upgrade. 

We find the best combination of roads to upgrade through the 
Metropolitan-Hastings algorithm. Mapping the problem described above
to the discussion previously about the Metropolitan-Hastings algorithm
we have

* ``\mathcal{N}``: The policy space. This is a list of ``K`` roads to
  improve. Note that this policy space is very large, exactly
  ``N \text{ Choose } K``. If you have 40 roads and have the budget to
  improve 10 of them, you have approximately 848 million ways to do
  that. 
* ``\Psi(N, N')``: The initial guess for a Markov Chain. Because the 
  policy space is so large, we don't characterize $\Psi$ directly, 
  and instead think of a method for drawing $N'$ given $N$ in a way
  that is equivalent to an aperiodic and irreducible Markov Chain. 

  Given an initial set of improvements $N$, we simply remove an
  improvement on one of the improved edges and add an improvement on
  one of the non-improved edges. Note that with this update procedure,
  we have ``\Psi(N, N') = \Psi(N', N)`` for all ``N`` and ``N'``. So
  we can focus only on the exponential terms. 

That's really all we need to solve for the optimal policy using the
Metropolitan-Hastings algorithm. 



