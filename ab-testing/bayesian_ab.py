"""Bayesian A/B testing with Beta-Bernoulli conjugate priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ArmResult:
    name: str
    successes: int
    trials: int
    posterior_mean: float
    ci_lower: float
    ci_upper: float


def update_beta(successes: int, trials: int, alpha: float = 1.0, beta: float = 1.0) -> stats.beta:
    """Posterior Beta(alpha + s, beta + n - s)."""
    return stats.beta(alpha + successes, beta + trials - successes)


def compare_arms(
    arm_a: tuple[int, int],
    arm_b: tuple[int, int],
    n_samples: int = 100_000,
    seed: int = 42,
) -> dict:
    """P(B > A) via Monte Carlo from Beta posteriors."""
    rng = np.random.default_rng(seed)
    post_a = update_beta(*arm_a)
    post_b = update_beta(*arm_b)

    samples_a = post_a.rvs(n_samples, random_state=rng)
    samples_b = post_b.rvs(n_samples, random_state=rng)
    prob_b_better = float(np.mean(samples_b > samples_a))

    def arm_summary(name: str, s: int, n: int, post: stats.beta) -> ArmResult:
        mean = post.mean()
        lo, hi = post.ppf([0.025, 0.975])
        return ArmResult(name, s, n, float(mean), float(lo), float(hi))

    return {
        "arm_a": arm_summary("A", *arm_a, post_a),
        "arm_b": arm_summary("B", *arm_b, post_b),
        "prob_b_better_than_a": prob_b_better,
    }


def simulate_ab_test(
    true_rate_a: float,
    true_rate_b: float,
    n_per_arm: int = 1000,
    seed: int = 42,
) -> dict:
    """Simulate experiment then run Bayesian analysis."""
    rng = np.random.default_rng(seed)
    s_a = rng.binomial(n_per_arm, true_rate_a)
    s_b = rng.binomial(n_per_arm, true_rate_b)
    return compare_arms((s_a, n_per_arm), (s_b, n_per_arm))
