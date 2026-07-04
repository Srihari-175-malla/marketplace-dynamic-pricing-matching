import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .simulation import MarketplaceSimulator, Driver, RideRequest
from .matching import BipartiteMatcher
from .pricing import SurgePricer
from .joint_optimizer import JointPricingMatchingOptimizer

class StressTestScenarioRunner:
    """
    Stress-Testing Harness for Marketplace Optimization Strategies.
    
    Evaluates system robustness under extreme market disruptions:
      1. Concentrated Event Surge: 4x demand surge in a single zone (e.g. stadium event).
      2. Driver Supply Bottleneck: Severe weather reducing available driver fleet by 40%.
      3. Geographic Zone Outage: Partial network connectivity failure.
    """
    def __init__(self, num_zones: int = 9, seed: int = 42):
        self.num_zones = num_zones
        self.seed = seed

    def run_event_surge_stress(
        self,
        surge_zone: int = 4,
        surge_multiplier: float = 4.0,
        num_steps: int = 5
    ) -> Dict[str, Any]:
        """Runs a concentrated event demand surge test."""
        sim = MarketplaceSimulator(num_zones=self.num_zones, drivers_per_zone=12, seed=self.seed)
        # Artificially inflate the target zone popularity weight
        sim.zone_weights[surge_zone] *= surge_multiplier
        sim.zone_weights /= np.sum(sim.zone_weights)

        joint_opt = JointPricingMatchingOptimizer(num_zones=self.num_zones, max_surge=3.5)

        total_reqs = 0
        matched_greedy = 0
        matched_joint = 0
        rev_greedy = 0.0
        rev_joint = 0.0

        for step in range(1, num_steps + 1):
            sim.reset_driver_availability()
            reqs = sim.step(step, {z: 1.0 for z in range(self.num_zones)})
            total_reqs += len(reqs)

            # Strategy A: Baseline Greedy
            pairs_g = BipartiteMatcher.greedy_match(sim.drivers, reqs)
            matched_greedy += len(pairs_g)
            rev_greedy += sum(r.base_fare for _, r in pairs_g)

            # Strategy B: Joint Optimizer
            prices_j, pairs_j = joint_opt.optimize_step(sim.drivers, reqs)
            matched_joint += len(pairs_j)
            rev_joint += sum(r.base_fare * prices_j.get(r.origin_zone, 1.0) for _, r in pairs_j)

        return {
            "total_requests": total_reqs,
            "greedy_matched": matched_greedy,
            "greedy_match_rate": (matched_greedy / max(1, total_reqs)) * 100.0,
            "greedy_revenue": rev_greedy,
            "joint_matched": matched_joint,
            "joint_match_rate": (matched_joint / max(1, total_reqs)) * 100.0,
            "joint_revenue": rev_joint,
            "revenue_lift": ((rev_joint - rev_greedy) / max(1.0, rev_greedy)) * 100.0
        }
