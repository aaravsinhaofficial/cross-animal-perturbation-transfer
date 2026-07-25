"""Protocol tests: the properties the central claim depends on.

These are the checks that would catch a leak or a broken holdout, independent of
whether the real dataset is present.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadence.data.containers import AnimalTrials
from cadence.data.features import N_FEATURES, unit_features
from cadence.data.icms import stable_seed
from cadence.holdout import (
    InterventionHoldout,
    eval_conditions,
    is_held_out,
    strip_training_conditions,
)
from cadence.linear_response import (
    LinearResponseConfig,
    fit_propagator,
    fit_shared,
    predict_delta,
)


def make_set(key="a/1", animal="a", n_obs=6, T=20, t0=5, n_unp=40, n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    conds = [1, 2]
    n = n_unp + n_per * len(conds)
    y = rng.poisson(1.0, (n, T, n_obs)).astype(np.float32)
    raw = np.zeros((n, T, 4), dtype=np.float32)
    on = np.zeros((n, T), dtype=np.float32)
    pert = np.zeros(n, dtype=bool)
    cond = np.zeros(n, dtype=np.int64)
    k = n_unp
    depths = np.linspace(100, 1800, n_obs)
    for ci, c in enumerate(conds):
        amp = 3.0 + 2.0 * ci
        for _ in range(n_per):
            raw[k, t0 : t0 + 8, 0] = amp / 12.0
            raw[k, t0 : t0 + 8, 1] = 0.5
            on[k, t0 : t0 + 8] = 1.0
            pert[k] = True
            cond[k] = c
            # an actual effect so the model has something to fit
            y[k, t0 : t0 + 8, :] += rng.poisson(amp / 3.0, (8, n_obs))
            k += 1
    return AnimalTrials(
        key=key, animal=animal, y=y, u=None, interv_raw=raw, interv_on=on,
        behavior=None, perturbed=pert, t0=t0, bin_s=0.025, cond=cond,
        unit_features=unit_features(y[~pert], depth_um=depths, cell_type=["pyramidal"] * n_obs),
        meta={
            "cond_amp": {1: 3.0, 2: 5.0},
            "cond_depth_um": {1: 900.0, 2: 900.0},
            "cond_info": {1: (3.0, 5), 2: (5.0, 5)},
            "unit_y_um": depths.tolist(),
            "cell_type": ["pyramidal"] * n_obs,
        },
    )


def test_stable_seed_is_process_independent():
    """Reproducible across interpreters, unlike ``hash`` which is salted per process."""
    import subprocess
    import sys

    expected = stable_seed("sub-ICMS93", "2023-09-14", 0)
    out = subprocess.run(
        [sys.executable, "-c",
         "from cadence.data.icms import stable_seed;"
         "print(stable_seed('sub-ICMS93','2023-09-14',0))"],
        capture_output=True, text=True, check=True,
    )
    assert int(out.stdout.strip()) == expected
    assert stable_seed("a", 1) != stable_seed("a", 2)


def test_unit_features_use_only_unperturbed_and_are_finite():
    s = make_set()
    f = s.unit_features
    assert f.shape == (s.n_obs, N_FEATURES)
    assert np.all(np.isfinite(f))
    # recomputing from the unperturbed trials must reproduce them exactly
    f2 = unit_features(
        s.y[~s.perturbed],
        depth_um=np.asarray(s.meta["unit_y_um"], float),
        cell_type=s.meta["cell_type"],
    )
    assert np.allclose(f, f2)


def test_no_intervention_before_alignment_or_on_unperturbed_trials():
    s = make_set()
    assert s.interv_on[:, : s.t0].sum() == 0
    assert s.interv_on[~s.perturbed].sum() == 0
    assert np.abs(s.interv_raw[~s.perturbed]).sum() == 0


def test_holdout_removes_condition_from_training_and_restricts_evaluation():
    s = make_set()
    spec = InterventionHoldout(kind="amplitude", amplitudes=(5.0,))
    assert is_held_out(s, 2, spec)
    assert not is_held_out(s, 1, spec)
    assert eval_conditions(s, spec) == [2]
    stripped = strip_training_conditions(s, spec)
    assert stripped is not None
    remaining = {int(c) for c in stripped.cond[stripped.perturbed]}
    assert remaining == {1}, "the held-out amplitude must be gone from training"
    # unperturbed trials are kept: they are what calibration uses
    assert int((~stripped.perturbed).sum()) == int((~s.perturbed).sum())


def test_holdout_none_is_identity():
    s = make_set()
    spec = InterventionHoldout(kind="none")
    assert strip_training_conditions(s, spec) is s
    assert eval_conditions(s, spec) == [1, 2]


def test_propagator_uses_only_unperturbed_trials():
    s = make_set()
    A, mu = fit_propagator(s, LinearResponseConfig())
    # corrupting only the perturbed trials must not change the propagator
    s2 = make_set()
    s2.y = s2.y.copy()
    s2.y[s2.perturbed] += 50.0
    A2, mu2 = fit_propagator(s2, LinearResponseConfig())
    assert np.allclose(A, A2)
    assert np.allclose(mu, mu2)


def test_propagator_is_stable():
    s = make_set()
    cfg = LinearResponseConfig()
    A, _ = fit_propagator(s, cfg)
    assert np.max(np.abs(np.linalg.eigvals(A))) <= cfg.spectral_clip + 1e-8


def test_shared_operator_transfers_to_a_second_animal():
    """Two animals generated the same way: the drive fitted on one must produce a
    positive-Delta prediction on the other."""
    cfg = LinearResponseConfig(n_time_basis=6, sigmas_um=(200.0, 800.0))
    a = make_set(key="a/1", animal="a", seed=0)
    b = make_set(key="b/1", animal="b", seed=1)
    props = {s.key: fit_propagator(s, cfg) for s in (a, b)}
    theta = fit_shared([a], cfg, props)
    pred = predict_delta(b, cfg, props[b.key][0], theta)
    from cadence import metrics as M

    out = M.evaluate_delta(b.y[:, b.t0 :], b.cond, b.perturbed, pred)
    assert out["delta_corr"] > 0.3, out


def test_subset_preserves_unit_features_and_metadata():
    s = make_set()
    sub = s.subset(np.arange(10))
    assert sub.unit_features is s.unit_features
    assert sub.meta is s.meta
    assert sub.t0 == s.t0


@pytest.mark.parametrize("kind", ["amplitude", "depth"])
def test_holdout_kinds_are_recognised(kind):
    s = make_set()
    spec = InterventionHoldout(
        kind=kind, amplitudes=(3.0,), depth_band_um=(800.0, 1000.0)
    )
    held = eval_conditions(s, spec)
    assert len(held) >= 1
