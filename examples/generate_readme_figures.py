"""Generate the reproducible tree diagrams displayed in the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from targetree import targetree
from targetree.tree_vis import plot_cart_tree


EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR / "data"
FIGURE_DIR = EXAMPLES_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

METHODS = (
    ("cart", None),
    ("mdfs", None),
    ("pfs", 0.5),
)


def fit_and_save(frame, outcome, predictors, dataset, slug, cut):
    """Fit CART, MDFS, and PFS and save one diagram for each method."""
    X = frame[predictors].to_numpy(dtype=float)
    y = outcome.to_numpy(dtype=float)
    results = {}

    for method, lbd in METHODS:
        model = targetree(
            depth=3,
            minimum_portion=0.02,
            method=method,
            lbd=lbd,
            cut=cut,
            feature_name=predictors,
        )
        model.fit(X, y)
        results[method] = model.get_risk(X, y)

        figure, _ = plot_cart_tree(
            model.tree,
            feature_name=model.feature_name,
            cut=model.cut,
            title=f"{dataset}: {method.upper()}",
            save_path=FIGURE_DIR / f"{slug}-{method}.png",
            split_rule_lines=2,
            title_font_size=18,
        )
        plt.close(figure)

    return results


def fit_kd_and_save(frame, outcome, predictors, dataset, slug, cut):
    """Fit KD-CART and KD-MDFS using a random-forest teacher."""
    X = frame[predictors].to_numpy(dtype=float)
    y = outcome.to_numpy(dtype=float)

    teacher = RandomForestClassifier(n_estimators=100, random_state=6)
    teacher.fit(X, y)
    teacher_probability = teacher.predict_proba(X)[:, 1]

    results = {}
    for method in ("cart", "mdfs"):
        model = targetree(
            depth=3,
            minimum_portion=30 / len(y),
            method=method,
            cut=cut,
            feature_name=predictors,
        )
        model.fit(X, y, prob=teacher_probability)
        results[f"kd-{method}"] = model.get_risk(X, y)

        figure, _ = plot_cart_tree(
            model.tree,
            feature_name=model.feature_name,
            cut=model.cut,
            title=f"{dataset}: KD-{method.upper()}",
            save_path=FIGURE_DIR / f"{slug}-kd-{method}.png",
            split_rule_lines=2,
            title_font_size=18,
        )
        plt.close(figure)

    return results


def main():
    diabetes = pd.read_csv(DATA_DIR / "diabetes.csv")
    diabetes_predictors = [
        column for column in diabetes.columns if column != "Outcome"
    ]
    diabetes_results = fit_and_save(
        diabetes,
        diabetes["Outcome"],
        diabetes_predictors,
        "Diabetes",
        "diabetes",
        0.60,
    )

    forestfires = pd.read_csv(DATA_DIR / "forestfires.csv")
    forestfires_predictors = [
        "X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"
    ]
    forestfires_results = fit_and_save(
        forestfires,
        (forestfires["area"] > 5).astype(float),
        forestfires_predictors,
        "Forest fires",
        "forestfires",
        1 / 3,
    )
    forestfires_kd_results = fit_kd_and_save(
        forestfires,
        (forestfires["area"] > 5).astype(float),
        forestfires_predictors,
        "Forest fires",
        "forestfires",
        1 / 3,
    )

    for dataset, results in (
        ("Diabetes", diabetes_results),
        ("Forestfires", forestfires_results),
        ("Forestfires", forestfires_kd_results),
    ):
        for method, counts in results.items():
            print(f"{dataset:11s} {method.upper():4s}: TP, FN, FP, TN = {counts}")


if __name__ == "__main__":
    main()
