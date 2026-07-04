import pytest
import numpy as np
from src.marketplace_opt.simulation import (
    MarketplaceSimulator,
    Driver,
    RideRequest,
    generate_synthetic_city_market
)
from src.marketplace_opt.matching import BipartiteMatcher
from src.marketplace_opt.pricing import SurgePricer
from src.marketplace_opt.joint_optimizer import JointPricingMatchingOptimizer
from src.marketplace_opt.fairness import (
    gini_coefficient,
    price_dispersion_index,
    generate_pareto_frontier
)
from src.marketplace_opt.stress_testing import StressTestScenarioRunner

def test_simulation_step():
    sim = MarketplaceSimulator(num_zones=4, drivers_per_zone=5, seed=42)
    assert len(sim.drivers) == 20
    assert len(sim.zone_centers) == 4

    reqs = sim.step(1)
    assert isinstance(reqs, list)
    for r in reqs:
        assert r.origin_zone in range(4)
        assert r.dest_zone in range(4)
        assert r.trip_distance >= 1.0
        assert r.base_fare > 0

def test_bipartite_matching_algorithms():
    sim = MarketplaceSimulator(num_zones=4, drivers_per_zone=5, seed=42)
    reqs = sim.step(1)

    # 1. Greedy
    greedy_pairs = BipartiteMatcher.greedy_match(sim.drivers, reqs)
    assert isinstance(greedy_pairs, list)

    # 2. Hungarian
    hungarian_pairs = BipartiteMatcher.hungarian_optimal_match(sim.drivers, reqs)
    assert isinstance(hungarian_pairs, list)
    assert len(hungarian_pairs) >= len(greedy_pairs) or len(hungarian_pairs) > 0

    # 3. KD-Tree
    kdtree_pairs = BipartiteMatcher.kdtree_spatial_match(sim.drivers, reqs)
    assert isinstance(kdtree_pairs, list)

def test_surge_pricing():
    sim = MarketplaceSimulator(num_zones=4, drivers_per_zone=5, seed=42)
    reqs = sim.step(1)

    fixed = SurgePricer.fixed_pricing(4)
    assert all(v == 1.0 for v in fixed.values())

    rule_surge = SurgePricer.rule_based_surge(4, sim.drivers, reqs, max_surge=3.0)
    assert all(1.0 <= v <= 3.0 for v in rule_surge.values())

    elastic_surge = SurgePricer.elasticity_optimized_pricing(4, sim.drivers, reqs, max_surge=3.0)
    assert all(1.0 <= v <= 3.0 for v in elastic_surge.values())

def test_joint_optimizer():
    sim = MarketplaceSimulator(num_zones=4, drivers_per_zone=5, seed=42)
    reqs = sim.step(1)

    joint_opt = JointPricingMatchingOptimizer(num_zones=4, max_surge=2.5)
    prices, pairs = joint_opt.optimize_step(sim.drivers, reqs)

    assert len(prices) == 4
    assert all(1.0 <= p <= 2.5 for p in prices.values())
    assert isinstance(pairs, list)

def test_fairness_and_gini():
    equal_incomes = [100.0, 100.0, 100.0, 100.0]
    assert pytest.approx(gini_coefficient(equal_incomes), abs=1e-3) == 0.0

    unequal_incomes = [0.0, 0.0, 0.0, 400.0]
    assert gini_coefficient(unequal_incomes) > 0.50

    disp = price_dispersion_index({0: 1.0, 1: 1.0, 2: 1.0})
    assert disp == 0.0

def test_pareto_frontier():
    frontier = generate_pareto_frontier(
        surge_caps=[1.0, 2.0],
        num_zones=4,
        drivers_per_zone=4,
        num_steps=2,
        seeds=[42]
    )
    assert len(frontier) == 2
    for rev, pfair, gini in frontier:
        assert rev >= 0
        assert pfair > 0
        assert 0.0 <= gini <= 1.0

def test_stress_testing():
    runner = StressTestScenarioRunner(num_zones=4, seed=42)
    res = runner.run_event_surge_stress(surge_zone=0, surge_multiplier=3.0, num_steps=2)
    assert res["total_requests"] >= 0
    assert "revenue_lift" in res
