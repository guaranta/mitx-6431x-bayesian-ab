"""Generate README figures — run: python docs/generate_figures.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# --- MCMC posterior ---
successes, trials = 45, 100
alpha_p, beta_p = 1 + successes, 1 + trials - successes
theta = np.linspace(0.001, 0.999, 500)
prior = stats.beta(1, 1).pdf(theta)
posterior = stats.beta(alpha_p, beta_p).pdf(theta)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(theta, prior, "--", color="#94a3b8", label="Prior Beta(1,1)")
ax.fill_between(theta, posterior, alpha=0.35, color="#3b82f6")
ax.plot(theta, posterior, color="#1d4ed8", lw=2, label=f"Posterior Beta({alpha_p},{beta_p})")
ax.axvline(alpha_p / (alpha_p + beta_p), color="#ef4444", ls=":", label=f"MAP = {alpha_p/(alpha_p+beta_p):.3f}")
ax.set_xlabel("θ (conversion rate)")
ax.set_ylabel("Density")
ax.set_title("Bayesian updating: 45 successes in 100 trials")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "mcmc_posterior.png", dpi=150)
plt.close()

# --- A/B posteriors ---
rng = np.random.default_rng(42)
post_a = stats.beta(121, 881)
post_b = stats.beta(146, 856)
x = np.linspace(0, 0.3, 300)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, post_a.pdf(x), label="Arm A: 120/1000", color="#6366f1", lw=2)
ax.plot(x, post_b.pdf(x), label="Arm B: 145/1000", color="#10b981", lw=2)
samples_a = post_a.rvs(100_000, random_state=rng)
samples_b = post_b.rvs(100_000, random_state=rng)
p_better = (samples_b > samples_a).mean()
ax.set_title(f"Bayesian A/B — P(B > A) = {p_better:.3f}")
ax.set_xlabel("Conversion rate")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "ab_posteriors.png", dpi=150)
plt.close()

# --- Monte Carlo latency ---
rng = np.random.default_rng(7)
lat = rng.normal(120, 30, 10_000)
lat[rng.random(10_000) < 0.02] *= 5
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lat, bins=60, color="#8b5cf6", alpha=0.8, edgecolor="white")
ax.axvline(np.percentile(lat, 95), color="#ef4444", ls="--", label=f"p95 = {np.percentile(lat,95):.0f} ms")
ax.axvline(np.percentile(lat, 99), color="#f97316", ls="--", label=f"p99 = {np.percentile(lat,99):.0f} ms")
ax.set_xlabel("Latency (ms)")
ax.set_title("Monte Carlo: LLM inference latency with 2% spike events")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "latency_distribution.png", dpi=150)
plt.close()

print(f"Saved 3 figures to {OUT}")
