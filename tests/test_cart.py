"""Basic tests for targetree."""

import numpy as np
import pytest
from scipy.special import expit

from targetree import CART, targetree


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.random.randn(500, 2)
    p = expit(X.sum(axis=1))
    y = np.random.binomial(1, p)
    return X, y, p


def test_fit_predict_cart(simple_data):
    X, y, _ = simple_data
    model = CART(depth=4, minimum_portion=0.05, method="cart")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_targetree_is_recommended_constructor(simple_data):
    X, y, _ = simple_data
    model = targetree(
        depth=3,
        minimum_portion=0.05,
        method="mdfs",
        cut=0.3,
    )
    assert isinstance(model, CART)
    assert model.fit(X, y) is model
    assert model.predict(X).shape == (len(y),)


def test_fit_predict_pfs(simple_data):
    X, y, p = simple_data
    model = CART(depth=4, minimum_portion=0.05, method="pfs", lbd=0.5, cut=0.5)
    model.fit(X, y, prob=p)
    preds = model.predict(X)
    assert preds.shape == (len(y),)


def test_fit_predict_mdfs(simple_data):
    X, y, _ = simple_data
    model = CART(depth=4, minimum_portion=0.05, method="mdfs", cut=0.3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)


@pytest.mark.parametrize("method", ["cart", "mdfs"])
def test_fit_predict_knowledge_distillation(simple_data, method):
    X, y, p = simple_data
    model = CART(depth=4, minimum_portion=0.05, method=method, cut=0.3)
    model.fit(X, y, prob=p)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0
    assert sum(model.get_risk(X, y)) == len(y)


def test_get_risk_returns_four_ints(simple_data):
    X, y, _ = simple_data
    model = CART(depth=3, minimum_portion=0.05, method="cart")
    model.fit(X, y)
    tp, fn, fp, tn = model.get_risk(X, y.astype(float))
    assert tp + fn + fp + tn == len(y)
    assert all(isinstance(v, int) for v in (tp, fn, fp, tn))


def test_honest_approach(simple_data):
    X, y, _ = simple_data
    X_train, X_honest = X[:300], X[300:]
    y_train, y_honest = y[:300], y[300:]

    model = CART(depth=3, minimum_portion=0.05, method="cart")
    model.fit(X_train, y_train)
    model.honest_approach(X_honest, y_honest)

    preds = model.predict(X, honest=True)
    assert preds.shape == (len(y),)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Invalid method"):
        CART(depth=3, minimum_portion=0.05, method="bogus")


def test_cart_nonzero_lbd_raises():
    with pytest.raises(ValueError, match="lbd must be 0"):
        CART(depth=3, minimum_portion=0.05, method="cart", lbd=0.5)


def test_categorical_features():
    np.random.seed(0)
    n = 300
    x_num = np.random.randn(n)
    cats = np.random.choice(["A", "B", "C"], size=n)
    X = np.empty((n, 2), dtype=object)
    X[:, 0] = x_num
    X[:, 1] = cats
    p = expit(x_num + np.where(cats == "A", 1.0, -0.5))
    y = np.random.binomial(1, p)

    model = CART(
        depth=4,
        minimum_portion=0.05,
        method="mdfs",
        categorical_features=[1],
        feature_name=["x_num", "x_cat"],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (n,)
