"""Is the ICMS response predictable across animals from physical variables alone?

This is a fast, model-free sanity probe, and also a legitimate baseline for the
paper. For every (session, condition, unit) it regresses the measured
time-resolved causal effect on quantities that require no intervention data in
the target animal:

    stimulation amplitude, depth of the stimulating contact, depth of the unit,
    their difference, the unit's baseline rate / class / coupling.

Leave-one-animal-out ridge regression. If this transfers, the conserved component
of the causal response is real and any failure of a dynamical model is a modelling
problem, not a data problem.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.baselines import measured_delta_set


def rows_for(s, max_depth=1900.0):
    dl, _ = measured_delta_set(s)
    amp = s.meta["cond_amp"]
    dep = s.meta["cond_depth_um"]
    uy = np.asarray(s.meta["unit_y_um"], float)
    ct = s.meta["cell_type"]
    feats = s.unit_features
    X, Y, meta = [], [], []
    for c, D in dl.items():
        a = float(amp[c]) if c in amp else float(amp[str(c)])
        sd = float(dep[c]) if c in dep else float(dep[str(c)])
        for n in range(s.n_obs):
            dz = (uy[n] - sd) / max_depth
            row = [
                a / 10.0,
                (a / 10.0) ** 2,
                sd / max_depth,
                uy[n] / max_depth,
                dz,
                abs(dz),
                np.exp(-((dz / 0.15) ** 2)),
                np.exp(-((dz / 0.35) ** 2)),
                (a / 10.0) * np.exp(-((dz / 0.15) ** 2)),
                (a / 10.0) * np.exp(-((dz / 0.35) ** 2)),
                (a / 10.0) * abs(dz),
                1.0 if "pyr" in str(ct[n]).lower() else 0.0,
                feats[n, 0], feats[n, 2], feats[n, 3], feats[n, 6],
                1.0,
            ]
            X.append(row)
            Y.append(D[:, n])
            meta.append((s.key, int(c), n))
    return np.array(X, float), np.array(Y, float), meta


def ridge_fit(X, Y, lam):
    G = X.T @ X + lam * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("results/tables/physical_probe.json"))
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]

    per = {s.key: rows_for(s) for s in ds.sets}
    animals = ds.animals
    report = {}
    print(f"{'held-out':14s} {'dR2':>8s} {'r':>8s} {'ceiling':>8s} {'norm_frac':>9s}")
    print("-" * 52)
    for a in animals:
        Xtr = np.concatenate([per[s.key][0] for s in ds.sets if s.animal != a])
        Ytr = np.concatenate([per[s.key][1] for s in ds.sets if s.animal != a])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        sd[-1] = 1.0; mu[-1] = 0.0
        W = ridge_fit((Xtr - mu) / sd, Ytr, args.lam)
        r2s, rs, ceils = [], [], []
        for s in ds.sets:
            if s.animal != a:
                continue
            X, Y, meta = per[s.key]
            P = ((X - mu) / sd) @ W
            conds = sorted({m[1] for m in meta})
            pred = {c: np.zeros((Y.shape[1], s.n_obs)) for c in conds}
            true = {c: np.zeros((Y.shape[1], s.n_obs)) for c in conds}
            for i, (_, c, n) in enumerate(meta):
                pred[c][:, n] = P[i]
                true[c][:, n] = Y[i]
            A = np.stack([true[c] for c in conds])
            Bm = np.stack([pred[c] for c in conds])
            r2s.append(M.delta_r2(A, Bm))
            rs.append(M.corr(A, Bm))
            ce = M.noise_ceiling(s.y[:, s.t0:], s.cond, s.perturbed, n_splits=120)
            ceils.append(ce["delta_r2_ceiling"])
        report[a] = {
            "delta_r2": float(np.mean(r2s)), "delta_corr": float(np.mean(rs)),
            "ceiling": float(np.mean(ceils)), "n_sets": len(r2s),
        }
        print(f"{a:14s} {report[a]['delta_r2']:+8.3f} {report[a]['delta_corr']:+8.3f} "
              f"{report[a]['ceiling']:8.3f}")
    vals = [report[a]["delta_r2"] for a in animals]
    mean, lo, hi = M.bootstrap_ci(vals)
    print(f"\nLOAO mean dR2 = {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"mean ceiling  = {np.mean([report[a]['ceiling'] for a in animals]):.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"per_animal": report, "mean": mean, "ci": [lo, hi]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
