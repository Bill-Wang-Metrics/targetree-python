"""Utility functions for targetree."""

from __future__ import annotations


def find_elements(v: list, a: float, b: float) -> tuple[int, int | None]:
    """Find boundary indices in a sorted list for sample-size constraints.

    Returns the index of the largest element <= a, and the index of the
    smallest element >= b.  Used internally to enforce the 10%–90% split
    window when choosing the best split threshold.

    Parameters
    ----------
    v:
        A monotone-increasing list of numbers (e.g. cumulative counts).
    a:
        Lower boundary value.
    b:
        Upper boundary value.

    Returns
    -------
    tuple[int, int | None]
        ``(largest_index_le_a, smallest_index_ge_b)``
    """
    if not v:
        return 0, None

    largest_le_a = 0
    smallest_ge_b: int | None = None

    for i, value in enumerate(v):
        if value <= a:
            largest_le_a = i
        elif value >= b and smallest_ge_b is None:
            smallest_ge_b = i

    return largest_le_a, smallest_ge_b
