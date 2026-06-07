"""Monte Carlo uncertainty quantification for AI operational metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricDistribution:
    name: str
    mean: float
    p50: float
    p95: float
    p99: float
    std: float


def simulate_latency_ms(
    base: float = 120,
    jitter: float = 30,
    spike_prob: float = 0.02,
    spike_mult: float = 5.0,
    n: int = 10_000,
    seed: int = 42,
) -> MetricDistribution:
    """Simulate API latency with occasional spikes."""
    rng = np.random.default_rng(seed)
    lat = rng.normal(base, jitter, n)
    spikes = rng.random(n) < spike_prob
    lat[spikes] *= spike_mult
    lat = np.clip(lat, 0, None)
    return _summarize("latency_ms", lat)


def simulate_cost_per_request(
    tokens_mean: float = 800,
    tokens_std: float = 200,
    price_per_1k: float = 0.002,
    n: int = 10_000,
    seed: int = 42,
) -> MetricDistribution:
    """Simulate LLM cost per request from token distribution."""
    rng = np.random.default_rng(seed)
    tokens = np.clip(rng.normal(tokens_mean, tokens_std, n), 50, None)
    cost = tokens / 1000 * price_per_1k
    return _summarize("cost_usd", cost)


def simulate_failure_rate(
    base_rate: float = 0.001,
    stress_factor: float = 1.0,
    n_days: int = 365,
    requests_per_day: int = 50_000,
    seed: int = 42,
) -> MetricDistribution:
    """Simulate daily failure counts over a year."""
    rng = np.random.default_rng(seed)
    rate = base_rate * stress_factor
    daily_failures = rng.binomial(requests_per_day, rate, n_days)
    return _summarize("daily_failures", daily_failures.astype(float))


def _summarize(name: str, samples: np.ndarray) -> MetricDistribution:
    return MetricDistribution(
        name=name,
        mean=float(np.mean(samples)),
        p50=float(np.percentile(samples, 50)),
        p95=float(np.percentile(samples, 95)),
        p99=float(np.percentile(samples, 99)),
        std=float(np.std(samples)),
    )
