# targetree

CART decision trees with **PFS** (Penalized Final Split) and **MDFS** (Maximum Distance Final Split) — designed for threshold-focused binary classification.

## Installation

```bash
pip install targetree
```

Tree visualization (`plot_cart_tree`) is included automatically — no extras needed.

## Quick start

```python
import numpy as np
from targetree import CART

model = CART(depth=5, minimum_portion=0.02, method="mdfs", cut=0.3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)   # probabilities in [0, 1]
tp, fn, fp, tn = model.get_risk(X_test, y_test.astype(float))
```

## Methods

| `method` | `lbd` default | Description |
|----------|--------------|-------------|
| `'cart'` | 0 | Standard CART (Gini impurity) |
| `'pfs'`  | 0 | PFS criterion (user-specified `lbd`) |
| `'mdfs'` | 1 | MDFS — maximise distance to the decision frontier |

## Key parameters

- **`depth`** — maximum tree depth
- **`minimum_portion`** — minimum fraction of samples required to split
- **`cut`** — classification threshold (default `0.5`)
- **`lbd`** — penalty weight blending impurity and frontier distance (PFS only)
- **`categorical_features`** — list of column indices treated as categorical
- **`feature_name`** — list of feature names for `print_tree()`

## Visualization

```python
from targetree.tree_vis import plot_cart_tree

plot_cart_tree(model.tree,
               feature_name=model.feature_name,
               cut=model.cut,
               title="Hospital Closure Tree")
```

Leaf nodes are colored **blue** when `P(Y=1|X) > cut` (predicted positive) and **white** otherwise.

## Honest estimation

```python
model.fit(X_build, y_build)
model.honest_approach(X_honest, y_honest)
honest_preds = model.predict(X_test, honest=True)
```

## Development

```bash
git clone https://github.com/YOUR_USERNAME/targetree
cd targetree
pip install -e ".[dev]"
pytest
```
