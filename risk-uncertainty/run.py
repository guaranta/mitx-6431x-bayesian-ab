"""Demo: Monte Carlo operational metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from monte_carlo_metrics import simulate_cost_per_request, simulate_failure_rate, simulate_latency_ms

if __name__ == "__main__":
    for metric in [
        simulate_latency_ms(),
        simulate_cost_per_request(),
        simulate_failure_rate(stress_factor=2.0),
    ]:
        print(f"\n=== {metric.name} ===")
        print(f"  mean={metric.mean:.4f}  p95={metric.p95:.4f}  p99={metric.p99:.4f}")
