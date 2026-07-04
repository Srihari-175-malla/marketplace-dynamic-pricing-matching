import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .simulation import MarketplaceSimulator
from .joint_optimizer import JointPricingMatchingOptimizer

def gini_coefficient(incomes: List[float]) -> float:
    """
    Computes the Gini inequality coefficient of driver income distributions.
    0.0 represents perfect equality; 1.0 represents complete inequality.
    """
    arr = np.array(incomes, dtype=float)
    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def price_dispersion_index(surge_multipliers: Dict[int, float]) -> float:
    """
    Computes standard deviation of surge multipliers across zones as a proxy for price disparity.
    """
    values = list(surge_multipliers.values())
    if not values:
        return 0.0
    return float(np.std(values))


def generate_pareto_frontier(
    surge_caps: Optional[List[float]] = None,
    num_zones: int = 9,
    drivers_per_zone: int = 15,
    num_steps: int = 10,
    seeds: Optional[List[int]] = None
) -> List[Tuple[float, float, float]]:
    """
    Constructs the Revenue vs. Price-Fairness vs. Driver Income-Equality Pareto Frontier.
    
    Sweeps regulatory surge caps to trace the multi-objective trade-off curve.
    Returns: List of (Total Revenue, Price Fairness Metric, Driver Gini Index) per cap.
    """
    if surge_caps is None:
        surge_caps = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    if seeds is None:
        seeds = [42, 43, 44]

    frontier = []
    optimizer = JointPricingMatchingOptimizer(num_zones=num_zones, max_surge=3.0)

    for cap in surge_caps:
        cap_revenues = []
        cap_price_disps = []
        cap_ginis = []

        for seed in seeds:
            sim = MarketplaceSimulator(
                num_zones=num_zones,
                drivers_per_zone=drivers_per_zone,
                seed=seed
            )
            tot_rev = 0.0
            price_disps = []

            for step in range(1, num_steps + 1):
                sim.reset_driver_availability()
                flat = {z: 1.0 for z in range(num_zones)}
                reqs = sim.step(step, flat)

                if reqs:
                    prices, pairs = optimizer.fairness_constrained_step(sim.drivers, reqs, surge_cap=cap)
                    sim.record_matched_trips(pairs, prices)
                    tot_rev += sum(t["fare"] for t in sim.completed_trips[-len(pairs):])
                    price_disps.append(price_dispersion_index(prices))

            cap_revenues.append(tot_rev)
            cap_price_disps.append(float(np.mean(price_disps)) if price_disps else 0.0)
            earnings = [d.total_earnings for d in sim.drivers]
            cap_ginis.append(gini_coefficient(earnings))

        mean_rev = float(np.mean(cap_revenues))
        # Price fairness metric: higher is fairer (1 / (1 + mean price dispersion))
        mean_pfair = float(1.0 / (1.0 + np.mean(cap_price_disps)))
        mean_gini = float(np.mean(cap_ginis))

        frontier.append((mean_rev, mean_pfair, mean_gini))

    return frontier
