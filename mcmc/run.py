"""Demo: Metropolis-Hastings on Beta-Bernoulli posterior."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metropolis_hastings import log_beta_posterior, metropolis_hastings

if __name__ == "__main__":
    successes, trials = 45, 100
    log_post = lambda t: log_beta_posterior(t, successes, trials)
    samples = metropolis_hastings(log_post, initial=0.5, n_samples=8000)

    print(f"Observed: {successes}/{trials}")
    print(f"MCMC mean: {samples.mean():.4f}")
    print(f"MCMC 95% CI: [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")
    print(f"Analytic Beta({1+successes},{1+trials-successes}) mean: {(1+successes)/(2+trials):.4f}")
