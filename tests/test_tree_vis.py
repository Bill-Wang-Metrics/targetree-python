"""Tests for publication-ready targetree diagrams."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from targetree.tree_vis import plot_cart_tree


@pytest.fixture
def simple_tree():
    return {
        "feature": 0,
        "threshold": 1.25,
        "left": (0.2, 60),
        "right": (0.8, 40),
    }


def test_plot_saves_and_returns_figure(simple_tree, tmp_path):
    output = tmp_path / "tree.pdf"
    figure, axes = plot_cart_tree(
        simple_tree,
        feature_name=["Risk score"],
        cut=0.5,
        title="MDFS tree",
        save_path=output,
        split_rule_lines=2,
        title_font_size=18,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert axes.title.get_fontsize() == 18
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        r"$\hat{\mu} > 0.5$  (targeted)",
        r"$\hat{\mu} \leq 0.5$  (not targeted)",
    ]
    plt.close(figure)


@pytest.mark.parametrize("title_font_size", [0, -1])
def test_invalid_title_font_size_raises(simple_tree, title_font_size):
    with pytest.raises(ValueError, match="title_font_size must be positive"):
        plot_cart_tree(simple_tree, title_font_size=title_font_size)


def test_nonnumeric_title_font_size_raises(simple_tree):
    with pytest.raises(TypeError, match="title_font_size must be a positive number"):
        plot_cart_tree(simple_tree, title_font_size="large")


def test_auto_font_uses_final_box_width(tmp_path):
    tree = {
        "feature": 0,
        "threshold": 127.5,
        "left": {
            "feature": 1,
            "threshold": 28.5,
            "left": {
                "feature": 2,
                "threshold": 22.25,
                "left": (0.0000, 35),
                "right": (0.0975, 236),
            },
            "right": {
                "feature": 2,
                "threshold": 28.25,
                "left": (0.2222, 63),
                "right": (0.3775, 151),
            },
        },
        "right": {
            "feature": 2,
            "threshold": 29.95,
            "left": {
                "feature": 0,
                "threshold": 161.5,
                "left": (0.2333, 60),
                "right": (0.6250, 16),
            },
            "right": {
                "feature": 0,
                "threshold": 143.5,
                "left": (0.6176, 68),
                "right": (0.7770, 139),
            },
        },
    }
    figure, axes = plot_cart_tree(
        tree,
        feature_name=["Glucose", "Age", "BMI"],
        cut=0.60,
        title="Diabetes: MDFS",
        split_rule_lines=2,
        save_path=tmp_path / "auto-font.pdf",
    )

    node_font_sizes = {text.get_fontsize() for text in axes.texts}
    assert len(node_font_sizes) == 1
    assert node_font_sizes.pop() >= 14.5
    plt.close(figure)


def test_wide_legend_does_not_overlap_asymmetric_tree(tmp_path):
    tree = {
        "feature": 0,
        "threshold": 85.85,
        "left": (0.4427, 51),
        "right": {
            "feature": 1,
            "threshold": 26.0,
            "left": {
                "feature": 2,
                "threshold": 769.35,
                "left": (0.2494, 374),
                "right": (0.3739, 42),
            },
            "right": (0.4183, 50),
        },
    }
    figure, axes = plot_cart_tree(
        tree,
        feature_name=["FFMC", "temp", "DC"],
        cut=1 / 3,
        title="Forest fires: KD-MDFS",
        split_rule_lines=2,
        save_path=tmp_path / "no-overlap.pdf",
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_bounds = axes.get_legend().get_window_extent(renderer=renderer)
    assert not any(
        legend_bounds.overlaps(box.get_window_extent(renderer=renderer))
        for box in axes.patches
    )
    plt.close(figure)
