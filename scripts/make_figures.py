"""Build every figure in the paper from the cached dataset and the result files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import figures as F
from cadence import metrics as M
from cadence.linear_response import (
    LinearResponseConfig,
    amp_features,
    fit_propagator,
    fit_shared_from_blocks,
    precompute_blocks,
    raised_cosine_basis,
)

MAX_D = 1900.0


# ---------------------------------------------------------------------------
def behaviour_traces(ds, lam: float = 1e-2) -> dict:
    """Leave-one-animal-out measured vs predicted detection-probability curves.

    Uses exactly the model reported in the results table (``cadence.dose``), so
    the per-animal numbers on this figure match Table 2.
    """
    from cadence.dose import cond_params, dose_design, ridge_solve

    sets = [s for s in ds.sets if s.behavior is not None]

    def rows(s):
        b = s.behavior[:, s.t0 :, 2]
        base = b[~s.perturbed].mean(0)
        X, Y, meta = [], [], []
        for c in sorted({int(x) for x in s.cond[s.perturbed]}):
            a, d = cond_params(s, c)
            X.append(dose_design(a, d, 0.0))
            Y.append(b[s.cond == c].mean(0) - base)
            meta.append((a, d, c))
        return np.array(X, float), np.array(Y, float), meta

    per = {s.key: rows(s) for s in sets}
    out = {}
    for a in ds.animals:
        tr = [s.key for s in sets if s.animal != a]
        if not tr:
            continue
        X = np.concatenate([per[k][0] for k in tr])
        Y = np.concatenate([per[k][1] for k in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        sd[-1] = 1.0; mu[-1] = 0.0
        W = ridge_solve((X - mu) / sd, Y, lam)
        # pool this animal's sessions, grouped by amplitude
        meas: dict[float, list[np.ndarray]] = {}
        pred: dict[float, list[np.ndarray]] = {}
        r2s = []
        for s in sets:
            if s.animal != a:
                continue
            Xa, Ya, meta = per[s.key]
            P = ((Xa - mu) / sd) @ W
            r2s.append(M.delta_r2(Ya[:, :, None], P[:, :, None]))
            for i, (amp, _, _) in enumerate(meta):
                meas.setdefault(amp, []).append(Ya[i])
                pred.setdefault(amp, []).append(P[i])
        amps = sorted(meas)
        # keep a readable number of amplitudes
        if len(amps) > 5:
            idx = np.linspace(0, len(amps) - 1, 5).astype(int)
            amps = [amps[i] for i in idx]
        T = len(next(iter(meas.values()))[0])
        t = (np.arange(T)) * ds.bin_s
        out[a] = {
            "t": t.tolist(),
            "amps": amps,
            "measured": [np.mean(meas[k], 0).tolist() for k in amps],
            "predicted": [np.mean(pred[k], 0).tolist() for k in amps],
            "delta_r2": float(np.mean(r2s)),
        }
    return out


def operator_kernel(ds) -> dict:
    """Visualise the fitted shared drive: its depth x time profile, its dose
    function, and how consistent it is when refitted leaving each animal out."""
    cfg = LinearResponseConfig()
    props = {s.key: fit_propagator(s, cfg) for s in ds.sets}
    blocks = {s.key: precompute_blocks(s, cfg, props[s.key][0]) for s in ds.sets}
    theta_all = fit_shared_from_blocks(blocks, ds.sets, cfg)

    n_s, n_a, n_b = len(cfg.sigmas_um), len(cfg.amp_basis), cfg.n_time_basis
    T = ds.sets[0].T - ds.sets[0].t0
    B = raised_cosine_basis(T, n_b)
    th = theta_all[: n_s * n_a * n_b].reshape(n_s, n_a, n_b)

    # depth x time drive at a reference amplitude
    dz = np.linspace(-900, 900, 121)
    psi = amp_features(5.0, cfg.amp_basis)
    K = np.zeros((len(dz), T))
    for j, sig in enumerate(cfg.sigmas_um):
        spatial = np.exp(-((dz / sig) ** 2))
        temporal = sum(psi[l] * (th[j, l] @ B) for l in range(n_a))
        K += np.outer(spatial, temporal)

    # dose function: total drive vs amplitude
    grid = np.linspace(1.0, 10.0, 40)
    gain = []
    for a in grid:
        p = amp_features(a, cfg.amp_basis)
        g = 0.0
        for j in range(n_s):
            temporal = sum(p[l] * (th[j, l] @ B) for l in range(n_a))
            g += temporal.sum()
        gain.append(g)

    # consistency: cosine between leave-one-animal-out refits
    thetas = {}
    for a in ds.animals:
        tr = [s for s in ds.sets if s.animal != a]
        thetas[a] = fit_shared_from_blocks(blocks, tr, cfg)
    labels, vals = [], []
    for a in ds.animals:
        v = thetas[a]
        cs = [
            float(abs(v @ w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-12))
            for b, w in thetas.items() if b != a
        ]
        labels.append(a.replace("sub-ICMS", "m"))
        vals.append(float(np.mean(cs)))
    return {
        "depth_time": K.tolist(),
        "extent": [0.0, float(T * ds.bin_s), float(dz[0]), float(dz[-1])],
        "amp_grid": grid.tolist(),
        "amp_gain": gain,
        "consistency_labels": labels,
        "consistency": vals,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--audit", type=Path, default=Path("results/tables/icms_audit.json"))
    ap.add_argument("--ladder", type=Path, default=Path("results/icms_ladder.json"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    want = set(args.only) if args.only else None

    def do(name):
        return want is None or name in want

    if do("concept"):
        print("fig1 concept")
        F.fig_concept(args.out)

    if do("teacher"):
        print("fig2 teacher")
        F.fig_teacher({
            "shared": Path("results/teacher_shared.json"),
            "heterogeneous": Path("results/teacher_heterogeneous.json"),
            "degenerate": Path("results/teacher_degenerate.json"),
        }, args.out)

    ds = None
    if args.cache.exists() and (do("dataset") or do("behavior") or do("operator")):
        with args.cache.open("rb") as fh:
            ds = pickle.load(fh)["dataset"]

    if ds is not None and do("dataset") and args.audit.exists():
        print("fig3 dataset")
        F.fig_dataset(ds, json.loads(args.audit.read_text()), args.out)

    if do("ladder") and args.ladder.exists():
        print("fig4 ladder")
        F.fig_ladder(json.loads(args.ladder.read_text()), args.out)

    if ds is not None and do("behavior"):
        print("fig5 behaviour")
        tr = behaviour_traces(ds)
        Path("results/tables").mkdir(parents=True, exist_ok=True)
        Path("results/tables/behaviour_traces.json").write_text(json.dumps(tr))
        F.fig_behavior(ds, tr, args.out)

    if ds is not None and do("operator"):
        print("fig6 operator")
        k = operator_kernel(ds)
        Path("results/tables/operator_kernel.json").write_text(json.dumps(k))
        F.fig_operator(k, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
