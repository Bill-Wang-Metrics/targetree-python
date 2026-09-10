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
