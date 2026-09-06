"""Unit tests for trajectory dynamics audit math helpers."""
from pathlib import Path

import numpy as np

from trajectory_dynamics_audit import (
    fit_linear,
    fit_quadratic,
    natural_frame_key,
    r_squared,
    residuals,
    rmse,
)


def test_natural_frame_order():
    paths = [Path("frame_10.png"), Path("frame_2.png"), Path("frame_1.png")]
    ordered = sorted(paths, key=natural_frame_key)
    assert [p.name for p in ordered] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_fit_linear_perfect_line():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 + 3.0 * t
    fit = fit_linear(t, y)
    assert abs(fit["y0"] - 2.0) < 1e-9
    assert abs(fit["v"] - 3.0) < 1e-9
    assert abs(fit["r2"] - 1.0) < 1e-9
    assert fit["rmse"] < 1e-9


def test_fit_quadratic_perfect_parabola():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # y = 1 + 2t + 0.5*0.5*t^2 = 1 + 2t + 0.25 t^2
    a_true = 0.5
    y = 1.0 + 2.0 * t + 0.5 * a_true * t**2
    fit = fit_quadratic(t, y)
    assert abs(fit["y0"] - 1.0) < 1e-8
    assert abs(fit["v"] - 2.0) < 1e-8
    assert abs(fit["a"] - a_true) < 1e-8
    assert abs(fit["r2"] - 1.0) < 1e-8
    assert fit["rmse"] < 1e-8


def test_rmse_and_r2():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0
    assert abs(r_squared(y_true, y_pred) - 1.0) < 1e-12
    y_pred2 = np.array([2.0, 2.0, 2.0])
    assert rmse(y_true, y_pred2) > 0
    assert r_squared(y_true, y_pred2) < 1.0


def test_residuals():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.5, 2.0, 3.5])
    res = residuals(y_true, y_pred)
    assert list(res) == [0.5, 0.0, -0.5]


def test_fit_handles_short_series():
    t = np.array([0.0])
    y = np.array([1.0])
    lin = fit_linear(t, y)
    assert np.isnan(lin["r2"])
    quad = fit_quadratic(t, y)
    assert np.isnan(quad["a"])
