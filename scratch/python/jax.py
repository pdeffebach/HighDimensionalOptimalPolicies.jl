import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import jax.numpy as jnp

import jax

from functools import partial

def joint_log_prob(x, y, tau, lamb, beta):
"""
Evaluate the joint probability of observing that
particular combination of parameters and outcomes
given inputs
"""
    lp = tfd.Gamma(0.5, 0.5).log_prob(tau)
    lp += tfd.Gamma(0.5, 0.5).log_prob(lamb).sum()
    lp += tfd.Normal(0., 1.).log_prob(beta).sum()
    logits = x @ (tau * lamb * beta)
    lp += tfd.Bern



def unconstrained_joint_log_prob(x, y, z):
"""
x is an
"""
    ndims = x.shape[-1]
    unc_tau, unc_lamb, beta = jnp.split(z, [1, 1 + ndims])
    unc_tau = unc_tau.reshape([])
    tau = jnp.exp(unc_tau)
    ldj = unc_tau # Make unc_tau a scalar
    lamb = jnp.exp(unc_lamb)
    ldj += unc_lamb.sum()
    return joint_log_prob(x, y, tau, lamb, beta) + ldj

target_log_prob = partial(unconstrained_joint_log_prob, x_data, y_data)

target_log_prob_and_grad = jax.value_and_grad(target_log_prob)
tlp, tlp_grad = target_log_prob_and_grad(z)