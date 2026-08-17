"""
tree_vis.py
-----------
Visualize a CART decision tree in the style of sklearn's plot_tree.

  - Internal nodes : light-grey boxes with the split condition only.
  - Leaf nodes     : blue  when P(Y=1|X) > cut  (predicted positive),
                     white when P(Y=1|X) ≤ cut  (predicted negative),
                     showing  "samples = N"  and  "P(Y=1|X) = x.xxxx".
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

_BLUE  = '#5B9BD5'   # positive leaf colour  (P > cut)
_WHITE = '#FFFFFF'   # negative leaf colour  (P ≤ cut)


def plot_cart_tree(tree, feature_name=None, cut=0.5,
                   figsize=None, title=None, save_path=None):
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
    save_path    : file path to save the figure (e.g. "tree.png").
                   If None, calls plt.show() instead.
    """

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
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # ── Helper: feature name and split label ─────────────────────────────────
    def _fname(f):
        return feature_name[f] if feature_name is not None else f"X{f}"

    def _split_label(feature, threshold):
        fn = _fname(feature)
        if isinstance(threshold, (set, frozenset, list, tuple)):
            cats = sorted(str(c) for c in threshold)
            return f"{fn} ∈ {{{', '.join(cats)}}}"
        return f"{fn} ≤ {threshold:.4f}"

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
                    f"P(Y=1|X) = {prob:.4f}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', color=tc, zorder=3)
            ax.text(x, y - hh * 0.38,
                    f"samples = {n}",
                    ha='center', va='center',
                    fontsize=8, color=tc, zorder=3)

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
                    fontsize=8, color='#111111', zorder=3)

    # ── 5. Legend ─────────────────────────────────────────────────────────────
    pos_patch = mpatches.Patch(facecolor=_BLUE,  edgecolor='#444444',
                               label=f'P(Y=1|X) > {cut}  (positive)')
    neg_patch = mpatches.Patch(facecolor=_WHITE, edgecolor='#444444',
                               label=f'P(Y=1|X) ≤ {cut}  (negative)')
    ax.legend(handles=[pos_patch, neg_patch],
              loc='upper right', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
