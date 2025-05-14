# High Dimensional Optimal Policies

This package implements a suite of algorithms for identifying optimal policies using the methodology of [kreindler2023](@citet). 

## Motivation

Policymakers are often tasked with identifying "optimal" policies. That is, the choices that policymakers can make which result in the highest level of welfare (however defined) for a population. Sometimes there are only a few levers available to the policymaker, such that their choice involves manipulating only a small number of key policies. For instance, "what should the interest rate paid on bank deposits in the Federal Reserve?" involves finding the optimal value of one key variable. 

Often, however, policymakers have a multitude of levers at their disposal, and finding the "optimal" policy involves an optimization problem of hundreds, thousands, or even tens of thousands of different variables. For example, [kreindler2023](@citet) asks what the optimal transportation network looks like in Jakarta, Indonesia. This task involves 

* Where should the buses go? Along a network with 1000s of nodes and an order of magnitude more edges. 
* How many buses should go on each route? 
* Which routes should have separate Bus Rapid Transit lanes? Which should have normal lanes? 

In this instance, the state space of policy choices is so large that it is impossible to characterize a single "best" policy. Moreover, policymakers might have other considerations not fully captured in an economist's simplified model of the economy, and would prefer a menu of candidate policies rather than one answer by itself. Finally, it may also be useful to learn what "qualities" are associated with a *set* of optimal policies rather than analyze one policy on its own. 

While optimal transportation policy is the main motivation of this package and will serve as it's guiding example, the same reasoning can be applied to any economic problem where the state space of policy levers is large, for instance place-based policies where resources are distributed across many different locations, or a tax system where taxes must be levied on a wide variety of goods. 


