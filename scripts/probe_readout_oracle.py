"""Is the bottleneck the causal operator, or the per-unit readout?

The shared drive is fitted on *other* animals and then held fixed. On the held-out
animal we grant, one at a time, increasingly generous oracles that touch only the
*readout* and never the operator:

  none          the honest zero-shot prediction
  gain          one scalar per unit                         (n_obs parameters)
  gain+offset   a scalar and an offset per unit             (2 n_obs parameters)
  timecourse    one scalar per unit, plus one shared        (n_obs + T parameters)
                temporal rescaling common to all units

Each oracle is fitted on the held-out animal's intervention trials, which the honest
protocol forbids -- that is the point. If a handful of per-unit numbers recovers most
of the accuracy, then the shared causal operator was already right and what is
missing is the map between the conserved response and individual neurons.

A per-unit gain has no way to invent structure in *time*: the temporal profile of
each unit's predicted response is fixed by the shared operator. So an improvement
here cannot be explained by the oracle fitting the response itself.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.linear_response import (
    LinearResponseConfig,
    fit_propagator,
    fit_shared_from_blocks,
    precompute_blocks,
    predict_delta,
)


def stack(d: dict[int, np.ndarray], conds) -> np.ndarray:
    return np.stack([d[c] for c in conds])


def apply_oracle(A: np.ndarray, B: np.ndarray, mode: str) -> np.ndarray:
    """A, B: (n_cond, T, n_obs) measured and predicted effects."""
    if mode == "none":
        return B
    n = A.shape[2]
    out = np.zeros_like(B)
    if mode in ("gain", "gain+offset"):
        for k in range(n):
            x = B[:, :, k].ravel()
            y = A[:, :, k].ravel()
            if mode == "gain":
                den = float(x @ x)
                g = float(x @ y) / den if den > 1e-12 else 0.0
                out[:, :, k] = g * B[:, :, k]
            else:
                X = np.stack([x, np.ones_like(x)], 1)
                w = np.linalg.lstsq(X, y, rcond=None)[0]
                out[:, :, k] = w[0] * B[:, :, k] + w[1]
        return out
    if mode == "timecourse":
        # per-unit gain, then one temporal profile shared by all units
        g = np.zeros(n)
        for k in range(n):
            x = B[:, :, k].ravel()
            den = float(x @ x)
            g[k] = float(x @ A[:, :, k].ravel()) / den if den > 1e-12 else 0.0
        Bg = B * g[None, None, :]
        T = A.shape[1]
        h = np.ones(T)
        for t in range(T):
            x = Bg[:, t, :].ravel()
            y = A[:, t, :].ravel()
            den = float(x @ x)
            h[t] = float(x @ y) / den if den > 1e-12 else 1.0
        return Bg * h[None, :, None]
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/readout_oracle.json"))
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    cfg = LinearResponseConfig()
    props = {s.key: fit_propagator(s, cfg) for s in ds.sets}
    blocks = {s.key: precompute_blocks(s, cfg, props[s.key][0]) for s in ds.sets}

    modes = ("none", "gain", "gain+offset", "timecourse")
    res: dict[str, list[float]] = {m: [] for m in modes}
    per_animal: dict[str, dict[str, float]] = {m: {} for m in modes}
    for a in ds.animals:
        train = [s for s in ds.sets if s.animal != a]
        theta = fit_shared_from_blocks(blocks, train, cfg)
        acc: dict[str, list[float]] = {m: [] for m in modes}
        for s in ds.sets:
            if s.animal != a:
                continue
            conds = sorted({int(c) for c in s.cond[s.perturbed]})
            pred = predict_delta(s, cfg, props[s.key][0], theta, conds=conds)
            dl, _ = M.measured_delta(s.y[:, s.t0 :], s.cond, s.perturbed)
            A = stack(dl, conds)
            B = stack(pred, conds)
            for m in modes:
                r2 = M.delta_r2(A, apply_oracle(A, B, m))
                res[m].append(r2)
                acc[m].append(r2)
        for m in modes:
            per_animal[m][a] = float(np.mean(acc[m]))

    print(f"{'readout oracle':16s} {'dR2 [95% CI]':>26s} {'>0':>8s}   per-animal")
    print("-" * 96)
    out = {}
    for m in modes:
        mean, lo, hi = M.bootstrap_ci(res[m])
        out[m] = {"delta_r2": mean, "ci": [lo, hi],
                  "sessions_above_zero": int(sum(x > 0 for x in res[m])),
                  "n": len(res[m]), "per_animal": per_animal[m]}
        pa = " ".join(f"{k.replace('sub-ICMS','m')}={v:+.2f}" for k, v in per_animal[m].items())
        print(f"{m:16s} {mean:+.3f} [{lo:+.3f},{hi:+.3f}]{'':4s} "
              f"{out[m]['sessions_above_zero']:3d}/{len(res[m])}   {pa}")
    diff, p = M.paired_permutation_test(res["gain"], res["none"])
    print(f"\ngain vs zero-shot: diff={diff:+.3f}  p_perm={p:.2e}")
    out["test_gain_vs_none"] = {"mean_diff": diff, "p_perm": p}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
