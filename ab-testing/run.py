"""Demo: Bayesian A/B test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bayesian_ab import compare_arms, simulate_ab_test

if __name__ == "__main__":
    result = compare_arms((120, 1000), (145, 1000))
    print("=== Bayesian A/B (observed data) ===")
    for arm in [result["arm_a"], result["arm_b"]]:
        print(f"  {arm.name}: {arm.successes}/{arm.trials} → mean={arm.posterior_mean:.4f} CI=[{arm.ci_lower:.4f},{arm.ci_upper:.4f}]")
    print(f"  P(B > A) = {result['prob_b_better_than_a']:.4f}")

    sim = simulate_ab_test(0.12, 0.15, n_per_arm=2000)
    print("\n=== Simulated A/B ===")
    print(f"  P(B > A) = {sim['prob_b_better_than_a']:.4f}")
