"""Metropolis-Hastings MCMC for Bayesian inference."""

from __future__ import annotations

import numpy as np


def log_beta_posterior(
    theta: float,
    successes: int,
    trials: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> float:
    """Log-posterior for Beta-Bernoulli: Beta(alpha + s, beta + n - s)."""
    if theta <= 0 or theta >= 1:
        return -np.inf
    return (
        (alpha_prior + successes - 1) * np.log(theta)
        + (beta_prior + trials - successes - 1) * np.log(1 - theta)
    )


def metropolis_hastings(
    log_target,
    initial: float,
    n_samples: int = 5000,
    proposal_std: float = 0.05,
    burn_in: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Generic random-walk Metropolis-Hastings sampler."""
    rng = np.random.default_rng(seed)
    samples = np.zeros(n_samples + burn_in)
    current = initial
    current_logp = log_target(current)

    for i in range(n_samples + burn_in):
        proposal = np.clip(rng.normal(current, proposal_std), 1e-6, 1 - 1e-6)
        proposal_logp = log_target(proposal)
        log_alpha = min(0.0, proposal_logp - current_logp)
        if np.log(rng.random()) < log_alpha:
            current, current_logp = proposal, proposal_logp
        samples[i] = current

    return samples[burn_in:]
