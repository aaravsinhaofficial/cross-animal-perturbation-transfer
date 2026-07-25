"""Tests for the scoring machinery, including a numerical validation of the
split-half noise-ceiling estimator against a ground-truth simulation."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import metrics as M


def _simulate(n_units=40, T=30, n_cond=6, n_per_cond=40, n_unp=200,
              signal=1.0, noise=1.0, seed=0):
    """Trials whose condition means differ by a known signal Delta."""
    rng = np.random.default_rng(seed)
    base_mean = rng.normal(0, 1.0, (T, n_units))
    deltas = {c: signal * rng.normal(0, 1.0, (T, n_units)) for c in range(1, n_cond + 1)}
    ys, conds, pert = [], [], []
    ys.append(base_mean + noise * rng.normal(0, 1, (n_unp, T, n_units)))
    conds.append(np.zeros(n_unp, int))
    pert.append(np.zeros(n_unp, bool))
    for c, d in deltas.items():
        ys.append(base_mean + d + noise * rng.normal(0, 1, (n_per_cond, T, n_units)))
        conds.append(np.full(n_per_cond, c))
        pert.append(np.ones(n_per_cond, bool))
    return (
        np.concatenate(ys),
        np.concatenate(conds),
        np.concatenate(pert),
        deltas,
    )


def test_delta_r2_perfect_and_null():
    y, cond, pert, deltas = _simulate(noise=0.0)
    d_true, _ = M.measured_delta(y, cond, pert)
    assert M.delta_r2(
        np.stack([d_true[c] for c in sorted(d_true)]),
        np.stack([deltas[c] for c in sorted(deltas)]),
    ) == pytest.approx(1.0, abs=1e-8)
    zeros = {c: np.zeros_like(v) for c, v in deltas.items()}
    out = M.evaluate_delta(y, cond, pert, zeros)
    assert out["delta_r2"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    "noise,n_per_cond",
    [(0.5, 40), (1.0, 40), (2.0, 40), (1.0, 15), (1.0, 120), (3.0, 200)],
)
def test_noise_ceiling_matches_achievable_r2(noise, n_per_cond):
    """The estimated ceiling must match the Delta-R^2 that an *oracle* model --
    one that knows the true Delta exactly -- actually attains."""
    achieved, estimated = [], []
    for seed in range(12):
        y, cond, pert, deltas = _simulate(
            noise=noise, n_per_cond=n_per_cond, n_unp=4 * n_per_cond, seed=seed
        )
        out = M.evaluate_delta(y, cond, pert, deltas)
        achieved.append(out["delta_r2"])
        ceil = M.noise_ceiling(y, cond, pert, n_splits=200, seed=seed)
        estimated.append(ceil["delta_r2_ceiling"])
    a, e = float(np.mean(achieved)), float(np.mean(estimated))
    # the estimator should track the attainable value closely
    assert abs(a - e) < 0.06, f"achieved {a:.3f} vs estimated ceiling {e:.3f}"


def test_bootstrap_ci_covers_mean():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    m, lo, hi = M.bootstrap_ci(v, n_boot=4000, seed=0)
    assert m == pytest.approx(0.3)
    assert lo < 0.3 < hi


def test_paired_permutation_directional():
    a = [0.5, 0.6, 0.55, 0.7, 0.62, 0.58]
    b = [0.1, 0.2, 0.15, 0.05, 0.12, 0.18]
    diff, p = M.paired_permutation_test(a, b, n_perm=20000, seed=0)
    assert diff > 0
    assert p == pytest.approx(2.0 ** -(len(a) - 1), rel=1e-6)


def test_corr_handles_constant():
    assert np.isnan(M.corr(np.zeros(10), np.arange(10)))
