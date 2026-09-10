"""Threshold-focused classification trees for interpretable policy targeting."""

from __future__ import annotations

from .cart import CART
from .utils import find_elements


def targetree(
    depth: int,
    minimum_portion: float,
    lbd: float | None = None,
    cut: float = 0.5,
    *,
    method: str = "pfs",
    feature_name: list[str] | None = None,
    calibrated: bool = False,
    categorical_features: list[int] | None = None,
) -> CART:
    """Create a targetree model using CART, PFS, or MDFS splitting.

    This is the recommended public constructor. The returned model provides
    ``fit()``, ``predict()``, ``get_risk()``, ``honest_approach()``, and
    ``print_tree()`` methods. ``CART`` remains available as a backward-
    compatible class name for existing code.
    """
    return CART(
        depth=depth,
        minimum_portion=minimum_portion,
        lbd=lbd,
        cut=cut,
        method=method,
        feature_name=feature_name,
        calibrated=calibrated,
        categorical_features=categorical_features,
    )


__all__ = ["targetree", "CART", "find_elements"]
__version__ = "0.1.0"
