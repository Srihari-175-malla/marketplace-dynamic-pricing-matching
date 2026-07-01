import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, Any, Optional
from .simulation import Driver, RideRequest

class BipartiteMatcher:
    """
    Bipartite Matching Engine for Driver-Rider Dispatch.
    
    Provides:
      1. Greedy Nearest-Neighbor Dispatch (First-Come First-Served)
      2. Globally Optimal Bipartite Matching via Hungarian Algorithm (O(V^3))
      3. Fast Spatial Indexing Matching via SciPy KD-Tree (O(N log M))
    """
    @staticmethod
    def greedy_match(
        drivers: List[Driver],
        requests: List[RideRequest],
        max_pickup_dist: float = 12.0
    ) -> List[Tuple[Driver, RideRequest]]:
        """Greedily assigns each ride request to the nearest available driver."""
        available_drivers = [d for d in drivers if d.is_available]
        if not available_drivers or not requests:
            return []

        matched_pairs: List[Tuple[Driver, RideRequest]] = []
        claimed_drivers = set()

        for req in requests:
            best_dist = float("inf")
            best_driver: Optional[Driver] = None

            for d in available_drivers:
                if d.driver_id in claimed_drivers:
                    continue
                dist = np.hypot(d.x - req.origin_coords[0], d.y - req.origin_coords[1])
                if dist < best_dist and dist <= max_pickup_dist:
                    best_dist = dist
                    best_driver = d

            if best_driver is not None:
                claimed_drivers.add(best_driver.driver_id)
                matched_pairs.append((best_driver, req))

        return matched_pairs

    @staticmethod
    def hungarian_optimal_match(
        drivers: List[Driver],
        requests: List[RideRequest],
        max_pickup_dist: float = 15.0
    ) -> List[Tuple[Driver, RideRequest]]:
        """
        Solves minimum-cost maximum-cardinality bipartite matching
        using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
        """
        available_drivers = [d for d in drivers if d.is_available]
        if not available_drivers or not requests:
            return []

        num_d = len(available_drivers)
        num_r = len(requests)

        # Build Cost Matrix (Pickup Distance)
        cost_matrix = np.full((num_d, num_r), fill_value=1e5, dtype=float)
        for i, d in enumerate(available_drivers):
            for j, r in enumerate(requests):
                dist = np.hypot(d.x - r.origin_coords[0], d.y - r.origin_coords[1])
                if dist <= max_pickup_dist:
                    cost_matrix[i, j] = dist

        # Solve Linear Sum Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[Tuple[Driver, RideRequest]] = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e4:  # valid finite distance match
                matched_pairs.append((available_drivers[r], requests[c]))

        return matched_pairs

    @staticmethod
    def kdtree_spatial_match(
        drivers: List[Driver],
        requests: List[RideRequest],
        max_pickup_dist: float = 12.0
    ) -> List[Tuple[Driver, RideRequest]]:
        """
        Scalable spatial batch matching using SciPy cKDTree for O(N log M) querying.
        """
        available_drivers = [d for d in drivers if d.is_available]
        if not available_drivers or not requests:
            return []

        driver_coords = np.array([[d.x, d.y] for d in available_drivers], dtype=float)
        tree = cKDTree(driver_coords)

        req_coords = np.array([[r.origin_coords[0], r.origin_coords[1]] for r in requests], dtype=float)
        distances, indices = tree.query(req_coords, k=1, distance_upper_bound=max_pickup_dist)

        matched_pairs: List[Tuple[Driver, RideRequest]] = []
        claimed_drivers = set()

        for j, (dist, d_idx) in enumerate(zip(distances, indices)):
            if d_idx < len(available_drivers):
                driver = available_drivers[d_idx]
                if driver.driver_id not in claimed_drivers:
                    claimed_drivers.add(driver.driver_id)
                    matched_pairs.append((driver, requests[j]))

        return matched_pairs
