import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any, Optional
from .simulation import Driver, RideRequest
from .pricing import SurgePricer

class JointPricingMatchingOptimizer:
    """
    Coupled Joint Pricing and Matching Optimizer.

    Avoids sequential suboptimality by co-optimizing:
      1. Zone-level surge pricing multipliers based on anticipated matching opportunities.
      2. Revenue-weighted bipartite matching dispatch prioritizing high-value long-distance
         and market-clearing trips while respecting spatial pickup constraints.
    """
    def __init__(
        self,
        num_zones: int = 9,
        max_surge: float = 3.0,
        max_pickup_dist: float = 14.0,
        price_elasticity: float = 0.75
    ):
        self.num_zones = num_zones
        self.max_surge = max_surge
        self.max_pickup_dist = max_pickup_dist
        self.price_elasticity = price_elasticity

    def optimize_step(
        self,
        drivers: List[Driver],
        requests: List[RideRequest]
    ) -> Tuple[Dict[int, float], List[Tuple[Driver, RideRequest]]]:
        """
        Computes joint pricing and matching dispatch for the current simulation step.
        """
        # 1. Compute Elasticity-Optimized Prices
        prices = SurgePricer.elasticity_optimized_pricing(
            num_zones=self.num_zones,
            drivers=drivers,
            requests=requests,
            price_elasticity=self.price_elasticity,
            max_surge=self.max_surge
        )

        # 2. Revenue-Weighted Minimum-Cost Bipartite Matching
        available_drivers = [d for d in drivers if d.is_available]
        if not available_drivers or not requests:
            return prices, []

        num_d = len(available_drivers)
        num_r = len(requests)

        # Objective in Hungarian is to minimize cost matrix.
        # We define: Cost(i, j) = Pickup_Distance(i, j) - Revenue_Weight * Fare(j)
        cost_matrix = np.full((num_d, num_r), fill_value=1e5, dtype=float)

        for i, d in enumerate(available_drivers):
            for j, r in enumerate(requests):
                dist = np.hypot(d.x - r.origin_coords[0], d.y - r.origin_coords[1])
                if dist <= self.max_pickup_dist:
                    surge = prices.get(r.origin_zone, 1.0)
                    fare = r.base_fare * surge
                    # Balance pickup travel time with trip economic value
                    cost_matrix[i, j] = dist - 0.40 * fare

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[Tuple[Driver, RideRequest]] = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e4:
                matched_pairs.append((available_drivers[r], requests[c]))

        return prices, matched_pairs

    def fairness_constrained_step(
        self,
        drivers: List[Driver],
        requests: List[RideRequest],
        surge_cap: float = 2.0
    ) -> Tuple[Dict[int, float], List[Tuple[Driver, RideRequest]]]:
        """
        Joint optimization subject to an explicit regulatory fairness surge cap.
        """
        capped_max_surge = min(self.max_surge, surge_cap)
        prices = SurgePricer.rule_based_surge(
            num_zones=self.num_zones,
            drivers=drivers,
            requests=requests,
            max_surge=capped_max_surge
        )

        available_drivers = [d for d in drivers if d.is_available]
        if not available_drivers or not requests:
            return prices, []

        num_d = len(available_drivers)
        num_r = len(requests)
        cost_matrix = np.full((num_d, num_r), fill_value=1e5, dtype=float)

        for i, d in enumerate(available_drivers):
            for j, r in enumerate(requests):
                dist = np.hypot(d.x - r.origin_coords[0], d.y - r.origin_coords[1])
                if dist <= self.max_pickup_dist:
                    # In fairness mode, weight ETA minimization higher
                    cost_matrix[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[Tuple[Driver, RideRequest]] = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e4:
                matched_pairs.append((available_drivers[r], requests[c]))

        return prices, matched_pairs
