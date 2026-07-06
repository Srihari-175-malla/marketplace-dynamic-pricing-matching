import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.marketplace_opt.simulation import MarketplaceSimulator
from src.marketplace_opt.matching import BipartiteMatcher
from src.marketplace_opt.pricing import SurgePricer
from src.marketplace_opt.joint_optimizer import JointPricingMatchingOptimizer
from src.marketplace_opt.fairness import (
    gini_coefficient, price_dispersion_index, generate_pareto_frontier
)
from src.marketplace_opt.stress_testing import StressTestScenarioRunner

NUM_ZONES = 9
NUM_STEPS = 10
SEEDS = [42, 43, 44]

def run_strategy(strategy_fn, seeds=SEEDS, num_steps=NUM_STEPS):
    revenues, matched_counts, request_counts, ginis = [], [], [], []

    for seed in seeds:
        sim = MarketplaceSimulator(num_zones=NUM_ZONES, drivers_per_zone=15, seed=seed)
        tot_rev = 0.0
        tot_matched = 0
        tot_req = 0

        for step in range(1, num_steps + 1):
            sim.reset_driver_availability()
            flat = {z: 1.0 for z in range(NUM_ZONES)}
            reqs = sim.step(step, flat)
            tot_req += len(reqs)

            if reqs:
                prices, pairs = strategy_fn(sim.drivers, reqs)
                sim.record_matched_trips(pairs, prices)
                tot_rev += sum(t["fare"] for t in sim.completed_trips[-len(pairs):])
                tot_matched += len(pairs)

        revenues.append(tot_rev)
        matched_counts.append(tot_matched)
        request_counts.append(tot_req)
        earnings = [d.total_earnings for d in sim.drivers]
        ginis.append(gini_coefficient(earnings))

    match_rates = [m / max(1, r) * 100.0 for m, r in zip(matched_counts, request_counts)]
    return {
        "revenue": float(np.mean(revenues)),
        "matched": float(np.mean(matched_counts)),
        "match_rate": float(np.mean(match_rates)),
        "gini": float(np.mean(ginis))
    }

def run_marketplace_benchmark():
    print("=" * 90)
    print("   MARKETPLACE DYNAMIC PRICING & MATCHING OPTIMIZATION BENCHMARK")
    print(f"   {NUM_ZONES} Geographic Zones | 135 Drivers | {NUM_STEPS}-Step Episodes | {len(SEEDS)} Random Seeds")
    print("=" * 90)

    joint_opt = JointPricingMatchingOptimizer(num_zones=NUM_ZONES, max_surge=3.0)

    strategies = [
        (
            "1. Fixed Pricing + Greedy Dispatch (Baseline)",
            lambda d, r: (SurgePricer.fixed_pricing(NUM_ZONES), BipartiteMatcher.greedy_match(d, r))
        ),
        (
            "2. Rule-Based Surge + Hungarian Bipartite",
            lambda d, r: (SurgePricer.rule_based_surge(NUM_ZONES, d, r), BipartiteMatcher.hungarian_optimal_match(d, r))
        ),
        (
            "3. Rule-Based Surge + KD-Tree Spatial Match",
            lambda d, r: (SurgePricer.rule_based_surge(NUM_ZONES, d, r), BipartiteMatcher.kdtree_spatial_match(d, r))
        ),
        (
            "4. Elasticity LP + Revenue-Weighted Joint Opt",
            lambda d, r: joint_opt.optimize_step(d, r)
        ),
        (
            "5. Joint Optimization + Regulatory Cap (<=2.0x)",
            lambda d, r: joint_opt.fairness_constrained_step(d, r, surge_cap=2.0)
        )
    ]

    header = f"{'Dispatch & Pricing Strategy':<45} | {'Revenue ($)':>12} | {'Matched':>8} | {'Match Rate':>11} | {'Driver Gini':>11}"
    print(f"\n{header}")
    print("-" * len(header))

    baseline_rev = None
    results = []
    for name, fn in strategies:
        t0 = time.perf_counter()
        res = run_strategy(fn)
        elapsed = time.perf_counter() - t0
        results.append((name, res, elapsed))
        if baseline_rev is None:
            baseline_rev = res["revenue"]

    for name, res, elapsed in results:
        rev_delta = f"+{(res['revenue'] - baseline_rev) / baseline_rev * 100:.1f}%" if baseline_rev else "baseline"
        print(
            f"{name:<45} | {res['revenue']:>12,.2f} | {res['matched']:>8.1f} | "
            f"{res['match_rate']:>10.1f}% | {res['gini']:>11.3f}  ({rev_delta})"
        )

    # Pareto Frontier
    print(f"\n{'-' * 90}")
    print("PARETO FRONTIER: Platform Revenue vs Price Fairness vs Driver Income Gini")
    print(f"{'-' * 90}")
    caps = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    frontier = generate_pareto_frontier(surge_caps=caps, num_zones=NUM_ZONES, num_steps=NUM_STEPS, seeds=SEEDS)

    p_header = f"{'Regulatory Cap':>14} | {'Platform Revenue':>18} | {'Price Fairness':>16} | {'Driver Income Gini':>20}"
    print(p_header)
    print("-" * len(p_header))
    for cap, (rev, pfair, gini) in zip(caps, frontier):
        print(f"{cap:>13.1f}x | {rev:>17,.2f} | {pfair:>16.4f} | {gini:>20.4f}")

    # Stress Testing
    print(f"\n{'-' * 90}")
    print("STRESS TEST: Concentrated 4x Event Surge Shock in Zone 4")
    print(f"{'-' * 90}")
    runner = StressTestScenarioRunner(num_zones=NUM_ZONES, seed=42)
    stress_res = runner.run_event_surge_stress(surge_zone=4, surge_multiplier=4.0, num_steps=5)
    print(f"Total Surge Requests: {stress_res['total_requests']}")
    print(f"Greedy Baseline   -> Matched: {stress_res['greedy_matched']} ({stress_res['greedy_match_rate']:.1f}%) | Revenue: ${stress_res['greedy_revenue']:,.2f}")
    print(f"Joint Optimizer   -> Matched: {stress_res['joint_matched']} ({stress_res['joint_match_rate']:.1f}%) | Revenue: ${stress_res['joint_revenue']:,.2f} (+{stress_res['revenue_lift']:.1f}%)")
    print("=" * 90)

if __name__ == "__main__":
    run_marketplace_benchmark()
