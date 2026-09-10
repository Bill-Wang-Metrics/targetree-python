"""CART with PFS and MDFS splitting methods."""

from __future__ import annotations

import numpy as np


class CART:
    """Classification and Regression Tree with optional PFS/MDFS splitting.

    Parameters
    ----------
    depth:
        Maximum depth of the tree.
    minimum_portion:
        Minimum fraction of the total training sample required in every
        terminal node.
    lbd:
        Weight of the L1-style penalty term relative to impurity.
        Defaults to ``0`` for ``'cart'``, ``0`` for ``'pfs'``, ``1`` for
        ``'mdfs'``.  Overriding the default is only allowed for ``'pfs'``.
    cut:
        Probability threshold used for classification and the PFS criterion.
    method:
        One of ``'cart'``, ``'pfs'``, or ``'mdfs'``.
    feature_name:
        Optional list of feature names for pretty-printing the tree.
    calibrated:
        Whether to use calibrated probability estimates (reserved for future
        use).
    categorical_features:
        List of column indices that should be treated as categorical.
    """

    def __init__(
        self,
        depth: int,
        minimum_portion: float,
        lbd: float | None = None,
        cut: float = 0.5,
        *,
        method: str = "pfs",
        feature_name: list[str] | None = None,
        calibrated: bool = False,
        categorical_features: list[int] | None = None,
    ) -> None:
        self.depth = depth
        self.minimum_portion = minimum_portion
        self.cut = cut
        self.tree: dict | tuple | None = None
        self.method = method
        self.feature_name = feature_name
        self.calibrated = calibrated
        self.categorical_features: list[int] = categorical_features or []
        self.is_categorical: np.ndarray | None = None

        _lbd_defaults = {"cart": 0, "mdfs": 1, "pfs": 0}
        if lbd is None:
            self.lbd = _lbd_defaults.get(self.method, 0)
        else:
            self.lbd = lbd

        if self.method == "cart":
            if self.lbd != 0:
                raise ValueError("lbd must be 0 when method='cart'.")
        elif self.method == "pfs":
            pass  # user-specified lbd is allowed
        elif self.method == "mdfs":
            if self.lbd != 1:
                raise ValueError("lbd must be 1 when method='mdfs'.")
        else:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                "Choose from ['cart', 'pfs', 'mdfs']."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_mask(
        self, X: np.ndarray, feature_idx: int, threshold
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(threshold, (set, tuple, list)):
            left = np.isin(X[:, feature_idx], list(threshold))
        else:
            left = X[:, feature_idx] <= threshold
        return left, ~left

    @staticmethod
    def _best_candidate_index(
        left_counts: np.ndarray,
        scores: np.ndarray,
        total: int,
        min_leaf_samples: int,
    ) -> int | None:
        """Return the best feasible split index under the leaf-size rule."""
        right_counts = total - left_counts
        feasible = np.flatnonzero(
            (left_counts >= min_leaf_samples)
            & (right_counts >= min_leaf_samples)
        )
        if feasible.size == 0:
            return None

        # Preserve the existing preference for 10%--90% splits when that
        # window contains at least one candidate satisfying the user's
        # minimum terminal-node size.
        if len(scores) > 10:
            balanced = feasible[
                (left_counts[feasible] >= int(0.1 * total))
                & (left_counts[feasible] <= int(0.9 * total))
            ]
            if balanced.size:
                feasible = balanced

        return int(feasible[np.argmin(scores[feasible])])

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, min_leaf_samples: int
    ):
        """Find the best impurity-reducing split across all features."""
        best_feature, best_threshold, best_impurity = None, None, float("inf")
        sum_y = y.sum()
        sum_y2 = (y ** 2).sum()
        nn = len(y)

        for feature_idx in range(X.shape[1]):
            if len(np.unique(X[:, feature_idx])) == 1:
                continue

            if self.is_categorical is not None and self.is_categorical[feature_idx]:
                result = self._best_split_categorical(
                    X[:, feature_idx], y, sum_y, sum_y2, nn,
                    min_leaf_samples,
                )
            else:
                result = self._best_split_numerical(
                    X[:, feature_idx], y, sum_y, sum_y2, nn,
                    min_leaf_samples,
                )

            if result is None:
                continue
            threshold, impurity = result
            if impurity < best_impurity:
                best_feature, best_threshold, best_impurity = (
                    feature_idx,
                    threshold,
                    impurity,
                )

        return best_feature, best_threshold

    def _best_split_categorical(
        self, x, y, sum_y, sum_y2, nn, min_leaf_samples
    ):
        categories, inverse = np.unique(x, return_inverse=True)
        if len(categories) == 1:
            return None

        x_count = np.bincount(inverse)
        y_count = np.bincount(inverse, weights=y)
        y2_count = np.bincount(inverse, weights=y ** 2)
        cat_means = y_count / x_count

        sorted_cat = np.argsort(cat_means)
        x_count_s = x_count[sorted_cat]
        y_count_s = y_count[sorted_cat]
        y2_count_s = y2_count[sorted_cat]

        lc = np.cumsum(x_count_s)[:-1]
        ls = np.cumsum(y_count_s)[:-1]
        lsq = np.cumsum(y2_count_s)[:-1]
        rc = nn - lc

        left_var = lsq / lc - (ls / lc) ** 2
        right_var = (sum_y2 - lsq) / rc - ((sum_y - ls) / rc) ** 2
        weighted_impurity = (left_var * lc + right_var * rc) * 2

        index = self._best_candidate_index(
            lc, weighted_impurity, nn, min_leaf_samples
        )
        if index is None:
            return None

        threshold = set(categories[sorted_cat[: index + 1]])
        return threshold, weighted_impurity[index]

    def _best_split_numerical(
        self, x, y, sum_y, sum_y2, nn, min_leaf_samples
    ):
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        left_count, left_sum, left_sq = [0], [0.0], [0.0]
        loc = 0

        for i in range(1, len(y_sorted)):
            if x_sorted[i] == x_sorted[i - 1]:
                continue
            chunk = y_sorted[loc:i]
            left_sum.append(left_sum[-1] + chunk.sum())
            left_sq.append(left_sq[-1] + (chunk ** 2).sum())
            left_count.append(left_count[-1] + (i - loc))
            loc = i

        lc = np.array(left_count[1:])
        ls = np.array(left_sum[1:])
        lsq = np.array(left_sq[1:])
        rc = nn - lc

        if len(lc) == 0:
            return None

        left_var = lsq / lc - (ls / lc) ** 2
        right_var = (sum_y2 - lsq) / rc - ((sum_y - ls) / rc) ** 2
        weighted_impurity = (left_var * lc + right_var * rc) * 2

        index = self._best_candidate_index(
            lc, weighted_impurity, nn, min_leaf_samples
        )
        if index is None:
            return None

        thre_idx = int(lc[index]) - 1
        threshold = (x_sorted[thre_idx] + x_sorted[thre_idx + 1]) / 2
        return threshold, weighted_impurity[index]

    def _best_pfs_split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        is_categorical: bool,
        min_leaf_samples: int,
    ):
        """Leaf-level split using the PFS/MDFS criterion."""
        if is_categorical:
            categories, inverse = np.unique(x, return_inverse=True)
            if len(categories) == 1:
                return None

            x_count = np.bincount(inverse)
            y_count = np.bincount(inverse, weights=y)
            y2_count = np.bincount(inverse, weights=y ** 2)
            cat_means = y_count / x_count

            sorted_cat = np.argsort(cat_means)
            x_count_s = x_count[sorted_cat]
            y_count_s = y_count[sorted_cat]
            y2_count_s = y2_count[sorted_cat]

            lc = np.cumsum(x_count_s)[:-1]
            ls = np.cumsum(y_count_s)[:-1]
            lsq = np.cumsum(y2_count_s)[:-1]
            nn = len(y)
            rc = nn - lc
            lp = lsq / lc
            rp = (y2_count.sum() - lsq) / rc

            weighted_impurity = (
                (lp - (ls / lc) ** 2) * lc
                + (rp - ((y_count.sum() - ls) / rc) ** 2) * rc
            ) * (1 - self.lbd) + (
                -np.abs(self.cut - lp) * lc
                - np.abs(self.cut - rp) * rc
            ) * self.lbd

            index = self._best_candidate_index(
                lc, weighted_impurity, nn, min_leaf_samples
            )
            if index is None:
                return None
            return set(categories[sorted_cat[: index + 1]])

        sum_y = y.sum()
        sum_y2 = (y ** 2).sum()
        nn = len(y)

        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        left_count, left_sum, left_sq = [0], [0.0], [0.0]
        loc = 0

        for i in range(1, len(y_sorted)):
            if x_sorted[i] == x_sorted[i - 1]:
                continue
            chunk = y_sorted[loc:i]
            left_sum.append(left_sum[-1] + chunk.sum())
            left_sq.append(left_sq[-1] + (chunk ** 2).sum())
            left_count.append(left_count[-1] + (i - loc))
            loc = i

        lc = np.array(left_count[1:])
        ls = np.array(left_sum[1:])
        lsq = np.array(left_sq[1:])
        if len(lc) == 0:
            return None

        rc = nn - lc
        lp = lsq / lc
        rp = (sum_y2 - lsq) / rc

        weighted_impurity = (
            (lp - (ls / lc) ** 2) * lc + (rp - ((sum_y - ls) / rc) ** 2) * rc
        ) * (1 - self.lbd) + (
            -np.abs(self.cut - lp) * lc - np.abs(self.cut - rp) * rc
        ) * self.lbd

        index = self._best_candidate_index(
            lc, weighted_impurity, nn, min_leaf_samples
        )
        if index is None:
            return None

        thre_idx = int(lc[index]) - 1
        return (x_sorted[thre_idx] + x_sorted[thre_idx + 1]) / 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        prob: np.ndarray | None = None,
    ) -> "CART":
        """Fit the CART model.

        Parameters
        ----------
        features:
            ``(n, p)`` array of input features.
        target:
            1-D binary outcome array.
        prob:
            Optional 1-D array of continuous probabilities in ``[0, 1]``.
            When provided the tree is grown against ``prob`` rather than
            ``target`` at ordinary splits, and terminal-node estimates use
            ``prob``. For PFS and MDFS, the final split continues to use the
            observed ``target``. This enables probability-assisted fitting,
            including KD-CART and KD-MDFS with probabilities from a teacher
            model.

        Returns
        -------
        self
        """
        total = len(target)
        min_leaf_samples = max(
            1, int(np.ceil(self.minimum_portion * total))
        )

        n_features = features.shape[1]
        self.is_categorical = np.zeros(n_features, dtype=bool)
        if self.categorical_features:
            self.is_categorical[self.categorical_features] = True

        leaf_method = self._best_pfs_split  # used at final depth level

        def grow(X, y, depth):
            if (
                depth == self.depth
                or len(np.unique(y)) == 1
                or len(y) < 2 * min_leaf_samples
            ):
                return (np.mean(y), len(y))

            feature, threshold = self._best_split(
                X, y, min_leaf_samples
            )
            if feature is None:
                return (np.mean(y), len(y))

            left_mask, right_mask = self._split_mask(X, feature, threshold)
            if min(left_mask.sum(), right_mask.sum()) < min_leaf_samples:
                return (np.mean(y), len(y))

            at_leaf = depth == self.depth - 1
            if at_leaf and self.method != "cart":
                threshold = leaf_method(
                    X[:, feature], y, bool(self.is_categorical[feature]),
                    min_leaf_samples,
                )
                if threshold is None:
                    return (np.mean(y), len(y))
                left_mask, right_mask = self._split_mask(X, feature, threshold)
                if min(left_mask.sum(), right_mask.sum()) < min_leaf_samples:
                    return (np.mean(y), len(y))
                return {
                    "feature": feature,
                    "threshold": threshold,
                    "left": (np.mean(y[left_mask]), left_mask.sum()),
                    "right": (np.mean(y[right_mask]), right_mask.sum()),
                }

            return {
                "feature": feature,
                "threshold": threshold,
                "left": grow(X[left_mask], y[left_mask], depth + 1),
                "right": grow(X[right_mask], y[right_mask], depth + 1),
            }

        def grow_with_prob(X, y, p, depth):
            if (
                depth == self.depth
                or (p.min() > self.cut or p.max() < self.cut)
                or len(y) < 2 * min_leaf_samples
            ):
                return (np.mean(p), len(p))

            feature, threshold = self._best_split(
                X, p, min_leaf_samples
            )
            if feature is None:
                return (np.mean(p), len(p))

            left_mask, right_mask = self._split_mask(X, feature, threshold)
            if min(left_mask.sum(), right_mask.sum()) < min_leaf_samples:
                return (np.mean(p), len(p))

            at_leaf = depth == self.depth - 1
            if at_leaf and self.method != "cart":
                threshold = leaf_method(
                    X[:, feature], y, bool(self.is_categorical[feature]),
                    min_leaf_samples,
                )
                if threshold is None:
                    return (np.mean(p), len(p))
                left_mask, right_mask = self._split_mask(X, feature, threshold)
                if min(left_mask.sum(), right_mask.sum()) < min_leaf_samples:
                    return (np.mean(p), len(p))
                return {
                    "feature": feature,
                    "threshold": threshold,
                    "left": (np.mean(p[left_mask]), left_mask.sum()),
                    "right": (np.mean(p[right_mask]), right_mask.sum()),
                }

            return {
                "feature": feature,
                "threshold": threshold,
                "left": grow_with_prob(
                    X[left_mask], y[left_mask], p[left_mask], depth + 1
                ),
                "right": grow_with_prob(
                    X[right_mask], y[right_mask], p[right_mask], depth + 1
                ),
            }

        if prob is None:
            self.tree = grow(features, target, 0)
        else:
            self.tree = grow_with_prob(features, target, prob, 0)

        return self

    def predict(
        self,
        features: np.ndarray,
        *,
        honest: bool = False,
        probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict probability estimates for each sample.

        Parameters
        ----------
        features:
            ``(n, p)`` array of input features.
        honest:
            If ``True``, return the honest-estimation leaf value (requires
            :meth:`honest_approach` to have been called first).
        probs:
            Calibrated probability inputs (reserved for calibrated variant).

        Returns
        -------
        np.ndarray
            1-D array of predicted probabilities.
        """

        def predict_single(x, node):
            if isinstance(node, tuple):
                return node[2] if honest else node[0]
            if isinstance(node["threshold"], float):
                branch = "left" if x[node["feature"]] <= node["threshold"] else "right"
            else:
                branch = "left" if x[node["feature"]] in node["threshold"] else "right"
            return predict_single(x, node[branch])

        return np.array([predict_single(x, self.tree) for x in features])

    def get_risk(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        honest: bool = False,
        probs: np.ndarray | None = None,
    ) -> tuple[int, int, int, int]:
        """Compute confusion-matrix counts at the classification threshold.

        Returns
        -------
        tuple[int, int, int, int]
            ``(TP, FN, FP, TN)``
        """
        estimate = self.predict(features, honest=honest, probs=probs)
        above = target > self.cut
        pred_above = estimate > self.cut

        TP = int((pred_above & above).sum())
        FN = int((~pred_above & above).sum())
        FP = int((pred_above & ~above).sum())
        TN = int((~pred_above & ~above).sum())
        return TP, FN, FP, TN

    def honest_approach(
        self, features: np.ndarray, target: np.ndarray
    ) -> "CART":
        """Attach honest-estimation leaf values from a held-out dataset.

        Traverses the existing tree structure and appends a third element to
        each leaf tuple containing the mean of ``target`` for samples that
        fall into that leaf.

        Parameters
        ----------
        features:
            ``(n, p)`` feature array (held-out / honest split).
        target:
            1-D binary outcome array for the held-out samples.

        Returns
        -------
        self
        """

        def traverse(node, X, y):
            if isinstance(node, tuple):
                honest_est = np.mean(y) if len(y) > 0 else 0.0
                return node + (honest_est,)

            feature_idx = node["feature"]
            threshold = node["threshold"]
            if isinstance(threshold, (set, tuple, list)):
                left_mask = np.isin(X[:, feature_idx], list(threshold))
            else:
                left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            node["left"] = traverse(node["left"], X[left_mask], y[left_mask])
            node["right"] = traverse(node["right"], X[right_mask], y[right_mask])
            return node

        self.tree = traverse(self.tree, features, target)
        return self

    def print_tree(self, node=None, depth: int = 0) -> None:
        """Print a text representation of the fitted tree."""
        if node is None:
            node = self.tree

        indent = "  " * depth
        if isinstance(node, tuple):
            print(f"{indent}Leaf: mean={node[0]:.3f}, n={node[1]}")
            return

        feature_idx = node["feature"]
        threshold = node["threshold"]
        name = (
            self.feature_name[feature_idx]
            if self.feature_name is not None
            else f"feature_{feature_idx}"
        )
        if isinstance(threshold, set):
            print(f"{indent}[{name} in {sorted(threshold)}]")
        else:
            print(f"{indent}[{name} <= {threshold:.4f}]")

        self.print_tree(node["left"], depth + 1)
        self.print_tree(node["right"], depth + 1)
