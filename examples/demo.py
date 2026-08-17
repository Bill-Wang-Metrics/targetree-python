"""Minimal demo — reproduce the original __main__ block."""

import numpy as np
from scipy.special import expit

from targetree import CART
from targetree.tree_vis import plot_cart_tree


def generate_data(n: int = 5000, seed: int = 42):
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    p = expit(X.sum(axis=1))
    y = np.random.binomial(1, p)
    return X, y, p


def main():
    X, y, p = generate_data()

    model = CART(depth=3, minimum_portion=0.02, lbd=1, cut=0.3, method="mdfs")
    model.fit(X, y)

    model.print_tree()
    tp, fn, fp, tn = model.get_risk(X, p)
    print(f"\nTP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"Accuracy: {(tp + tn) / (tp + fn + fp + tn):.3f}")

    plot_cart_tree(model.tree, feature_name=model.feature_name, cut=model.cut)


if __name__ == "__main__":
    main()
