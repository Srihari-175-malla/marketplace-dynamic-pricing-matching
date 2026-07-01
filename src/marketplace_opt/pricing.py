import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .simulation import Driver, RideRequest

class SurgePricer:
    """
    Dynamic Surge Pricing Engine for Multi-Zone Marketplaces.
    """
    @staticmethod
    def fixed_pricing(num_zones: int) -> Dict[int, float]:
        """Flat baseline pricing without surge."""
        return {z: 1.0 for z in range(num_zones)}

    @staticmethod
    def rule_based_surge(
        num_zones: int,
        drivers: List[Driver],
        requests: List[RideRequest],
        alpha: float = 0.50,
        max_surge: float = 3.0,
        min_surge: float = 1.0
    ) -> Dict[int, float]:
        """
        Calculates surge multiplier per zone based on local supply-demand deficit:
          Surge_z = 1.0 + alpha * max(0, Requests_z / max(1, AvailableDrivers_z) - 1.0)
        """
        # Count available drivers per zone
        zone_supply = {z: 0 for z in range(num_zones)}
        for d in drivers:
            if d.is_available:
                zone_supply[d.zone_id] = zone_supply.get(d.zone_id, 0) + 1

        # Count incoming requests per zone
        zone_demand = {z: 0 for z in range(num_zones)}
        for r in requests:
            zone_demand[r.origin_zone] = zone_demand.get(r.origin_zone, 0) + 1

        surge_multipliers = {}
        for z in range(num_zones):
            supply = max(1, zone_supply[z])
            demand = zone_demand[z]
            ratio = demand / supply

            if ratio > 1.0:
                surge = 1.0 + alpha * (ratio - 1.0)
            else:
                surge = 1.0

            surge_multipliers[z] = float(np.clip(surge, min_surge, max_surge))

        return surge_multipliers

    @staticmethod
    def elasticity_optimized_pricing(
        num_zones: int,
        drivers: List[Driver],
        requests: List[RideRequest],
        price_elasticity: float = 0.75,
        max_surge: float = 3.0
    ) -> Dict[int, float]:
        """
        Optimizes surge multiplier per zone to maximize expected revenue
        given downward-sloping demand curve and driver supply capacity:
          Revenue(s) = s * min(Supply_z, Demand_z * exp(-elasticity * (s - 1)))
        """
        zone_supply = {z: 0 for z in range(num_zones)}
        for d in drivers:
            if d.is_available:
                zone_supply[d.zone_id] = zone_supply.get(d.zone_id, 0) + 1

        zone_demand = {z: 0 for z in range(num_zones)}
        for r in requests:
            zone_demand[r.origin_zone] = zone_demand.get(r.origin_zone, 0) + 1

        surge_multipliers = {}
        candidate_surges = np.linspace(1.0, max_surge, 41)

        for z in range(num_zones):
            supply = zone_supply[z]
            nominal_demand = zone_demand[z]

            if nominal_demand == 0 or supply == 0:
                surge_multipliers[z] = 1.0
                continue

            best_rev = -1.0
            best_surge = 1.0

            for s in candidate_surges:
                realized_demand = nominal_demand * np.exp(-price_elasticity * (s - 1.0))
                matched_volume = min(float(supply), realized_demand)
                rev = s * matched_volume

                if rev > best_rev:
                    best_rev = rev
                    best_surge = s

            surge_multipliers[z] = float(best_surge)

        return surge_multipliers
