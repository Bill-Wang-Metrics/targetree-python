"""
tree_vis.py
-----------
Visualize a CART decision tree in the style of sklearn's plot_tree.

  - Internal nodes : light-grey boxes with the split condition only.
  - Leaf nodes     : blue  when mu-hat > cut  (predicted positive),
                     white when mu-hat ≤ cut  (predicted negative),
                     showing "N = n" and the estimated leaf mean, mu-hat.
  - Edges          : plain grey lines (no labels / no arrows).
  - Legend         : shows the blue / white colour meaning and the cut value.

Usage
-----
    from targetree.tree_vis import plot_cart_tree
    plot_cart_tree(model.tree,
                   feature_name=model.feature_name,
                   cut=model.cut,
                   title="My Tree")
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

_BLUE  = '#5B9BD5'   # positive leaf colour  (mu-hat > cut)
_WHITE = '#FFFFFF'   # negative leaf colour  (mu-hat ≤ cut)


def plot_cart_tree(tree, feature_name=None, cut=0.5,
                   figsize=None, title=None, save_path=None,
                   font_size=None, split_rule_lines=1,
                   title_font_size=None):
    """
    Draw a CART decision tree as a matplotlib figure.

    Parameters
    ----------
    tree         : root node of the CART tree (dict or tuple).
                   Internal nodes are dicts with keys
                   "feature", "threshold", "left", "right".
                   Leaf nodes are tuples (mean_prob, n_samples).
    feature_name : list/array of feature names.  None → "X0", "X1", …
    cut          : decision probability threshold drawn on the colorbar.
    figsize      : (width, height) in inches.  Auto-computed if None.
    title        : optional figure title string.
    save_path    : file path to save the figure. If it has no extension,
                   ".pdf" is appended; for example, "tree" saves
                   "tree.pdf". An explicit extension can still be supplied.
                   If None, calls plt.show() instead.
    font_size    : font size used for every node label and the legend. If
                   None, uses the largest size for which every node label
                   fits inside its box.
    split_rule_lines : number of lines used for an internal-node split rule.
                   Use 1 for "X ≤ a" or 2 for "X" on the first line and
                   "≤ a" on the second line.
    title_font_size : title font size in points. If None, the title is made
                   slightly larger than the resolved node font size.

    Returns
    -------
    fig, ax : the Matplotlib figure and axes containing the tree.
    """

    if split_rule_lines not in (1, 2):
        raise ValueError("split_rule_lines must be either 1 or 2")
    if title_font_size is not None:
        if (isinstance(title_font_size, bool) or
                not isinstance(title_font_size, (int, float))):
            raise TypeError("title_font_size must be a positive number or None")
        if title_font_size <= 0:
            raise ValueError("title_font_size must be positive")

    # ── 1. Assign (x, y) positions ──────────────────────────────────────────
    #
    #  Strategy  (standard in-order layout):
    #    • Leaves get consecutive integer x-slots: 0, 1, 2, …
    #    • Internal nodes sit at the midpoint of their two children.
    #    • y = −depth  (root at 0, children below).
    #
    positions = {}    # nid → (x, y)  in grid units
    node_info = {}    # nid → tree node object
    _col      = [0]   # mutable leaf-column counter
    _maxdepth = [0]

    def _assign(node, depth, nid):
        node_info[nid] = node
        if depth > _maxdepth[0]:
            _maxdepth[0] = depth
        if isinstance(node, tuple):                     # leaf
            positions[nid] = (_col[0], -depth)
            _col[0] += 1
        else:                                            # internal
            _assign(node["left"],  depth + 1, nid * 2 + 1)
            _assign(node["right"], depth + 1, nid * 2 + 2)
            lx = positions[nid * 2 + 1][0]
            rx = positions[nid * 2 + 2][0]
            positions[nid] = ((lx + rx) / 2.0, -depth)

    _assign(tree, 0, 0)
    n_leaves  = _col[0]
    max_depth = _maxdepth[0]

    # ── 2. Figure size and axis limits ───────────────────────────────────────
    #
    #  We allocate ~1.5 in per leaf column (horizontal) and
    #  ~1.8 in per tree level (vertical), giving node boxes of roughly
    #  1.2 in × 1.0 in — similar in proportion to sklearn's plot_tree.
    #
    col_in  = 1.5          # inches per leaf column
    row_in  = 1.8          # inches per tree level
    node_w  = 0.82         # node box width  in grid units
    node_h  = 0.55         # node box height in grid units
    hw, hh  = node_w / 2, node_h / 2

    if figsize is None:
        fig_w = max(n_leaves * col_in + 1.8, 7.0)   # +1.8 for colorbar
        fig_h = max((max_depth + 1) * row_in + 0.7, 3.5)
    else:
        fig_w, fig_h = figsize

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, n_leaves - 0.5)
    ax.set_ylim(-max_depth - 0.65, 0.65)
    ax.axis('off')

    # ── Helper: feature name and split label ─────────────────────────────────
    def _fname(f):
        return feature_name[f] if feature_name is not None else f"X{f}"

    def _split_label(feature, threshold):
        fn = _fname(feature)
        if isinstance(threshold, (set, frozenset, list, tuple)):
            cats = sorted(str(c) for c in threshold)
            condition = f"∈ {{{', '.join(cats)}}}"
        else:
            condition = f"≤ {threshold:.4f}"
        separator = " " if split_rule_lines == 1 else "\n"
        return f"{fn}{separator}{condition}"

    # Use one common font size throughout the tree. In automatic mode, measure
    # the rendered labels (including math notation) instead of estimating their
    # width from character counts.
    split_labels = [
        _split_label(node["feature"], node["threshold"])
        for node in node_info.values() if not isinstance(node, tuple)
    ]
    leaf_labels = [
        (rf"$\hat{{\mu}}$ = {node[0]:.4f}", f"N = {node[1]}")
        for node in node_info.values() if isinstance(node, tuple)
    ]

    if font_size is not None:
        if isinstance(font_size, bool) or not isinstance(font_size, (int, float)):
            raise TypeError("font_size must be a positive number or None")
        if font_size <= 0:
            raise ValueError("font_size must be positive")
        resolved_font_size = float(font_size)
    else:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        origin = ax.transData.transform((0, 0))
        corner = ax.transData.transform((node_w, node_h))
        box_width_px = abs(corner[0] - origin[0])
        box_height_px = abs(corner[1] - origin[1])

        def _fits(candidate):
            measurements = [(label, "normal", 0.82)
                            for label in split_labels]
            for mean_label, count_label in leaf_labels:
                measurements.append((mean_label, "bold", 0.38))
                measurements.append((count_label, "normal", 0.38))

            for label, weight, height_fraction in measurements:
                probe = ax.text(0, 0, label, fontsize=candidate,
                                fontweight=weight, alpha=0)
                bounds = probe.get_window_extent(renderer=renderer)
                probe.remove()
                if (bounds.width > box_width_px * 0.88 or
                        bounds.height > box_height_px * height_fraction):
                    return False
            return True

        # Search in quarter-point steps. The upper bound prevents a very small
        # tree from receiving disproportionately large labels.
        lower, upper = 4.0, 24.0
        while upper - lower > 0.25:
            candidate = (lower + upper) / 2
            if _fits(candidate):
                lower = candidate
            else:
                upper = candidate
        resolved_font_size = round(lower * 4) / 4

    resolved_title_font_size = (
        max(14.0, resolved_font_size + 2.0)
        if title_font_size is None else float(title_font_size)
    )
    if title:
        ax.set_title(title, fontsize=resolved_title_font_size,
                     fontweight='bold', pad=10)

    # ── 3. Draw edges (plain lines, top of child ↔ bottom of parent) ────────
    for nid, node in node_info.items():
        if isinstance(node, tuple):
            continue
        px, py = positions[nid]
        lx, ly = positions[nid * 2 + 1]
        rx, ry = positions[nid * 2 + 2]
        ax.plot([px, lx], [py - hh, ly + hh],
                color='#777777', lw=0.9, zorder=1)
        ax.plot([px, rx], [py - hh, ry + hh],
                color='#777777', lw=0.9, zorder=1)

    # ── 4. Draw nodes ─────────────────────────────────────────────────────────
    for nid, node in node_info.items():
        x, y = positions[nid]

        if isinstance(node, tuple):                     # ── leaf node ──
            prob   = node[0]
            n      = node[1]
            fcolor = _BLUE if prob > cut else _WHITE
            tc     = 'white' if prob > cut else '#111111'

            box = FancyBboxPatch(
                (x - hw, y - hh), node_w, node_h,
                boxstyle="round,pad=0.03",
                linewidth=1.2, edgecolor='#444444',
                facecolor=fcolor, zorder=2)
            ax.add_patch(box)

            ax.text(x, y + hh * 0.30,
                    rf"$\hat{{\mu}}$ = {prob:.4f}",
                    ha='center', va='center',
                    fontsize=resolved_font_size, fontweight='bold',
                    color=tc, zorder=3)
            ax.text(x, y - hh * 0.38,
                    f"N = {n}",
                    ha='center', va='center',
                    fontsize=resolved_font_size, color=tc, zorder=3)

        else:                                           # ── internal node ──
            label = _split_label(node["feature"], node["threshold"])

            box = FancyBboxPatch(
                (x - hw, y - hh), node_w, node_h,
                boxstyle="round,pad=0.03",
                linewidth=1.0, edgecolor='#999999',
                facecolor='#e6e6e6', zorder=2)
            ax.add_patch(box)

            ax.text(x, y, label,
                    ha='center', va='center',
                    fontsize=resolved_font_size, color='#111111', zorder=3)

    # ── 5. Legend ─────────────────────────────────────────────────────────────
    cut_label = f"{cut:g}"
    pos_patch = mpatches.Patch(facecolor=_BLUE,  edgecolor='#444444',
                               label=rf'$\hat{{\mu}} > {cut_label}$  (targeted)')
    neg_patch = mpatches.Patch(facecolor=_WHITE, edgecolor='#444444',
                               label=rf'$\hat{{\mu}} \leq {cut_label}$  (not targeted)')
    ax.legend(handles=[pos_patch, neg_patch],
              loc='upper right', fontsize=resolved_font_size, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".pdf")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return fig, ax
