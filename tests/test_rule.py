"""Checks on the analysis that says what the shared rule is.

That analysis is the paper's centrepiece and it fits nothing, which makes it easy to
read but also easy to fool: any procedure that correlates a property of a neuron with
its measured response could pick up a relationship that the measurement itself creates.
So the checks here build recordings where the answer is known, and confirm the analysis
finds it when it is there and does not when it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cadence import individuality as I
from cadence.data.containers import AnimalTrials

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse_rule import properties, response  # noqa: E402


def make_set(mode, key="a/1", n_trials=400, T=30, n_obs=40, t0=10, seed=0):
    """A recording where the light's effect is, or is not, tied to firing rate.

    Rates start well above the size of the flat effect so that subtracting a constant
    never clips at zero. Clipping would make even a uniform effect fall harder on the
    slowest cells, which is a property of the construction and not of the analysis.
    """
    rng = np.random.default_rng(seed)
    rate = np.linspace(6.0, 14.0, n_obs)                   # neurons differ in rate
    y = rng.poisson(rate[None, None, :], size=(n_trials, T, n_obs)).astype(np.float32)
    perturbed = np.zeros(n_trials, bool)
    perturbed[: n_trials // 2] = True
    if mode == "proportional":
        # suppression proportional to what the neuron was already doing
        y[perturbed, t0:] *= 0.6
    elif mode == "flat":
        # every neuron loses the same amount, so nothing is individual
        y[perturbed, t0:] -= 1.0
    elif mode == "none":
        pass
    return AnimalTrials(
        key=key, animal=key.split("/")[0], y=y, u=None,
        interv_raw=np.zeros((n_trials, T, 1), np.float32),
        interv_on=np.zeros((n_trials, T), np.float32),
        behavior=None, perturbed=perturbed, t0=t0, bin_s=0.05,
        cond=np.where(perturbed, 1, 0).astype(np.int64),
        meta={"cond_amp": {1: 5.0}, "cond_depth_um": {1: 0.0},
              "unit_y_um": list(np.linspace(0, 800, n_obs))},
    )


def measure(s):
    feat_idx, base_idx = I.control_split(s)
    props = properties(s, feat_idx)
    out = []
    for d in response(s, feat_idx, base_idx):
        out.append(float(np.corrcoef(props["firing rate"], d)[0, 1]))
    return float(np.mean(out))


def test_finds_a_rate_dependent_effect():
    r = [measure(make_set("proportional", key=f"a{i}/1", seed=i)) for i in range(5)]
    assert np.mean(r) < -0.8, r
    assert all(x < 0 for x in r), r


# A correlation between two unrelated vectors of length n has standard deviation
# about 1/sqrt(n - 2), so with 40 neurons a single recording scatters by about 0.16
# around zero. Averaging over eight recordings leaves about 0.06, and the bounds below
# are three times that.
TOL = 0.18


def test_reports_nothing_when_every_neuron_loses_the_same_amount():
    """The population still moves; no neuron moves differently for its own reasons."""
    r = [measure(make_set("flat", key=f"b{i}/1", seed=100 + i)) for i in range(8)]
    assert abs(np.mean(r)) < TOL, r


def test_reports_nothing_when_there_is_no_effect():
    r = [measure(make_set("none", key=f"c{i}/1", seed=200 + i)) for i in range(8)]
    assert abs(np.mean(r)) < TOL, r


def test_the_null_protocol_finds_nothing_in_a_real_effect():
    """Replacing the perturbed trials with control trials must remove the signal."""
    real, null = [], []
    for i in range(8):
        s = make_set("proportional", key=f"d{i}/1", seed=7 + i)
        feat_idx, base_idx = I.control_split(s)
        props = properties(s, feat_idx)
        real += [float(np.corrcoef(props["firing rate"], d)[0, 1])
                 for d in response(s, feat_idx, base_idx)]
        null += [float(np.corrcoef(props["firing rate"], d)[0, 1])
                 for d in response(s, feat_idx, base_idx, null=True)]
    assert np.mean(real) < -0.8, np.mean(real)
    assert abs(np.mean(null)) < TOL, np.mean(null)
