"""targetree — CART with PFS and MDFS splitting for threshold-focused classification."""

from .cart import CART
from .utils import find_elements

__all__ = ["CART", "find_elements"]
__version__ = "0.1.0"
