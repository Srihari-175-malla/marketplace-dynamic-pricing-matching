"""
Marketplace Dynamic Pricing & Matching Optimization Library
"""

from .simulation import (
    MarketplaceSimulator,
    Driver,
    RideRequest,
    generate_synthetic_city_market
)
from .matching import BipartiteMatcher
from .pricing import SurgePricer
from .joint_optimizer import JointPricingMatchingOptimizer
from .fairness import (
    gini_coefficient,
    price_dispersion_index,
    generate_pareto_frontier
)
from .stress_testing import StressTestScenarioRunner

__version__ = "1.0.0"
__all__ = [
    "MarketplaceSimulator",
    "Driver",
    "RideRequest",
    "generate_synthetic_city_market",
    "BipartiteMatcher",
    "SurgePricer",
    "JointPricingMatchingOptimizer",
    "gini_coefficient",
    "price_dispersion_index",
    "generate_pareto_frontier",
    "StressTestScenarioRunner"
]
