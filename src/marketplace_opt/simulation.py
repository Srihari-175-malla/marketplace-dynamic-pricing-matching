import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Driver:
    driver_id: int
    zone_id: int
    x: float
    y: float
    is_available: bool = True
    total_earnings: float = 0.0
    trips_completed: int = 0
    time_busy_remaining: int = 0

@dataclass
class RideRequest:
    request_id: int
    origin_zone: int
    dest_zone: int
    origin_coords: Tuple[float, float]
    dest_coords: Tuple[float, float]
    trip_distance: float
    base_fare: float
    created_step: int
    max_wait_steps: int = 3
    is_matched: bool = False

class MarketplaceSimulator:
    """
    Two-Sided Spatial Marketplace Simulator for Urban Ride-Hailing.
    
    Models:
      - 2D Spatial Grid partitioned into distinct geographic zones.
      - Stochastic Poisson arrival processes for riders and driver fleets.
      - Dynamic demand elasticity responding to surge pricing multipliers.
      - Fleet tracking, driver dispatch, trip fulfillment, and driver income accumulation.
    """
    def __init__(
        self,
        num_zones: int = 9,
        grid_width: float = 30.0,
        drivers_per_zone: int = 12,
        base_fare_rate: float = 2.50,
        per_km_rate: float = 1.80,
        price_elasticity: float = 0.75,
        seed: int = 42
    ):
        self.num_zones = num_zones
        self.grid_width = grid_width
        self.base_fare_rate = base_fare_rate
        self.per_km_rate = per_km_rate
        self.price_elasticity = price_elasticity
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Generate Zone Centers on a grid
        self.zone_centers = self._init_zone_centers()
        self.zone_weights = self._init_zone_popularity_weights()

        # Initialize Driver Fleet
        self.drivers: List[Driver] = []
        self._init_driver_fleet(drivers_per_zone)

        # Simulation telemetry
        self.current_step = 0
        self.active_requests: List[RideRequest] = []
        self.completed_trips: List[Dict[str, Any]] = []
        self.request_counter = 0

    def _init_zone_centers(self) -> Dict[int, Tuple[float, float]]:
        zones = {}
        grid_dim = int(np.ceil(np.sqrt(self.num_zones)))
        step_size = self.grid_width / (grid_dim + 1)
        z_id = 0
        for i in range(1, grid_dim + 1):
            for j in range(1, grid_dim + 1):
                if z_id < self.num_zones:
                    zones[z_id] = (float(i * step_size), float(j * step_size))
                    z_id += 1
        return zones

    def _init_zone_popularity_weights(self) -> np.ndarray:
        # Central zones have higher demand density
        weights = np.zeros(self.num_zones)
        center_x = self.grid_width / 2.0
        center_y = self.grid_width / 2.0
        for z_id, (zx, zy) in self.zone_centers.items():
            dist_to_center = np.hypot(zx - center_x, zy - center_y)
            weights[z_id] = np.exp(-0.08 * dist_to_center)
        return weights / np.sum(weights)

    def _init_driver_fleet(self, drivers_per_zone: int) -> None:
        d_id = 0
        for z_id in range(self.num_zones):
            zx, zy = self.zone_centers[z_id]
            for _ in range(drivers_per_zone):
                # Small dispersion around zone center
                dx = float(np.clip(zx + self.rng.normal(0, 1.5), 0.5, self.grid_width - 0.5))
                dy = float(np.clip(zy + self.rng.normal(0, 1.5), 0.5, self.grid_width - 0.5))
                self.drivers.append(Driver(driver_id=d_id, zone_id=z_id, x=dx, y=dy))
                d_id += 1

    def reset_driver_availability(self) -> None:
        """Updates driver busy timers and resets available drivers at step boundary."""
        for d in self.drivers:
            if not d.is_available:
                d.time_busy_remaining -= 1
                if d.time_busy_remaining <= 0:
                    d.is_available = True
                    d.time_busy_remaining = 0

    def step(self, step_idx: int, surge_multipliers: Optional[Dict[int, float]] = None) -> List[RideRequest]:
        """
        Advances the simulation by one discrete time step:
          1. Generates stochastic ride requests according to price-elastic Poisson arrivals.
          2. Returns active requests for pricing/matching evaluation.
        """
        self.current_step = step_idx
        if surge_multipliers is None:
            surge_multipliers = {z: 1.0 for z in range(self.num_zones)}

        new_requests: List[RideRequest] = []
        base_lambda = 8.0  # nominal requests per step across system

        for z_id in range(self.num_zones):
            surge = surge_multipliers.get(z_id, 1.0)
            # Demand elasticity curve: D(p) = lambda_0 * exp(-elasticity * (surge - 1))
            elastic_factor = np.exp(-self.price_elasticity * max(0.0, surge - 1.0))
            expected_arrivals = base_lambda * self.zone_weights[z_id] * elastic_factor
            num_arrivals = self.rng.poisson(expected_arrivals)

            zx, zy = self.zone_centers[z_id]
            for _ in range(num_arrivals):
                ox = float(np.clip(zx + self.rng.normal(0, 1.2), 0.5, self.grid_width - 0.5))
                oy = float(np.clip(zy + self.rng.normal(0, 1.2), 0.5, self.grid_width - 0.5))

                # Destination zone chosen proportional to weights
                dest_z = int(self.rng.choice(self.num_zones, p=self.zone_weights))
                dx, dy = self.zone_centers[dest_z]
                dest_x = float(np.clip(dx + self.rng.normal(0, 1.5), 0.5, self.grid_width - 0.5))
                dest_y = float(np.clip(dy + self.rng.normal(0, 1.5), 0.5, self.grid_width - 0.5))

                dist = float(max(1.0, np.hypot(ox - dest_x, oy - dest_y)))
                base_fare = float(self.base_fare_rate + self.per_km_rate * dist)

                req = RideRequest(
                    request_id=self.request_counter,
                    origin_zone=z_id,
                    dest_zone=dest_z,
                    origin_coords=(ox, oy),
                    dest_coords=(dest_x, dest_y),
                    trip_distance=dist,
                    base_fare=base_fare,
                    created_step=step_idx
                )
                self.request_counter += 1
                new_requests.append(req)

        self.active_requests = new_requests
        return new_requests

    def record_matched_trips(
        self,
        matched_pairs: List[Tuple[Driver, RideRequest]],
        surge_multipliers: Dict[int, float]
    ) -> None:
        """
        Executes matched dispatches:
          - Marks driver busy and relocates to destination coordinates upon trip completion.
          - Credits driver earnings (80% platform split).
          - Records transaction telemetry.
        """
        for driver, request in matched_pairs:
            surge = surge_multipliers.get(request.origin_zone, 1.0)
            final_fare = float(request.base_fare * surge)
            driver_payout = float(0.80 * final_fare)

            driver.is_available = False
            # Approximate trip duration in steps proportional to distance
            driver.time_busy_remaining = int(max(1, np.round(request.trip_distance / 6.0)))
            driver.x, driver.y = request.dest_coords
            driver.zone_id = request.dest_zone
            driver.total_earnings += driver_payout
            driver.trips_completed += 1

            request.is_matched = True

            self.completed_trips.append({
                "step": self.current_step,
                "driver_id": driver.driver_id,
                "request_id": request.request_id,
                "origin_zone": request.origin_zone,
                "dest_zone": request.dest_zone,
                "fare": final_fare,
                "driver_payout": driver_payout,
                "surge": surge,
                "distance": request.trip_distance
            })


def generate_synthetic_city_market(
    num_zones: int = 9,
    drivers_per_zone: int = 15,
    seed: int = 42
) -> MarketplaceSimulator:
    return MarketplaceSimulator(
        num_zones=num_zones,
        drivers_per_zone=drivers_per_zone,
        seed=seed
    )
