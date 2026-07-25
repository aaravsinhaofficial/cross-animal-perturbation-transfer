"""Checks on the split into shared and individual parts.

The claims these protect are the ones that would quietly invalidate the result: that
the control trials the model reads never overlap the ones defining its target, that a
stereotype scores exactly zero on the individual part, and that the ceiling is the
number it says it is.
"""

from __future__ import annotations

import numpy as np

from cadence import individuality as I
from cadence.data.containers import AnimalTrials


def make_set(key="a/1", n_trials=200, T=30, n_obs=12, t0=10, effect=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.poisson(3.0, size=(n_trials, T, n_obs)).astype(np.float32)
    perturbed = np.zeros(n_trials, bool)
    perturbed[: n_trials // 2] = True
    cond = np.where(perturbed, 1, 0).astype(np.int64)
    # a per-neuron effect that is not the same for every neuron
    kick = effect * np.linspace(-1, 1, n_obs)[None, :]
    y[perturbed, t0:] += kick
    return AnimalTrials(
        key=key, animal=key.split("/")[0], y=y, u=None,
        interv_raw=np.zeros((n_trials, T, 1), np.float32),
        interv_on=np.zeros((n_trials, T), np.float32),
        behavior=None, perturbed=perturbed, t0=t0, bin_s=0.05, cond=cond,
        meta={"cond_amp": {1: 5.0}, "cond_depth_um": {1: 0.0},
              "unit_y_um": list(np.linspace(0, 800, n_obs))},
    )


def test_control_split_is_disjoint_and_stable():
    s = make_set()
    a, b = I.control_split(s)
    assert len(np.intersect1d(a, b)) == 0
    assert len(a) + len(b) == int((~s.perturbed).sum())
    # the split is a property of the recording, not of when it is asked for
    a2, b2 = I.control_split(s)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    # and it uses only control trials
    assert not s.perturbed[np.concatenate([a, b])].any()


def test_centre_removes_the_shared_part():
    x = np.random.default_rng(0).normal(size=(4, 20, 9))
    d = I.centre(x)
    assert np.allclose(d.mean(axis=-1), 0.0, atol=1e-10)


def test_a_stereotype_scores_exactly_zero_on_the_individual_part():
    """A prediction that is the same for every neuron has no individual part."""
    rng = np.random.default_rng(1)
    truth = I.centre(rng.normal(size=(3, 20, 9)))
    stereotype = np.tile(rng.normal(size=(3, 20, 1)), (1, 1, 9))
    assert I.score(truth, I.centre(stereotype)) == 0.0


def test_ceiling_is_high_for_a_large_effect_and_low_for_none():
    strong = I.delta_ceiling(make_set(effect=4.0, seed=2))
    none = I.delta_ceiling(make_set(effect=0.0, seed=2))
    assert strong > 0.8
    assert none < 0.2


def test_operator_finds_nothing_when_there_is_nothing():
    """With the perturbed trials replaced by control trials, the answer is zero."""
    op = I.SharedOperator()
    for i in range(4):
        op.add(make_set(key=f"a{i}/1", seed=i), null=True)
    res = op.loao()
    vals = np.array([v["delta_r2"] for v in res.values()])
    assert np.all(np.abs(vals) < 0.25), vals


def test_operator_recovers_a_shared_depth_rule():
    """When the effect is the same smooth function of depth in every animal, a shared
    operator fitted on the others should find it in the one held out."""
    op = I.SharedOperator()
    for i in range(5):
        op.add(make_set(key=f"a{i}/1", n_trials=400, effect=3.0, seed=100 + i))
    res = op.loao()
    vals = np.array([v["delta_r2"] for v in res.values()])
    assert vals.mean() > 0.3, vals
