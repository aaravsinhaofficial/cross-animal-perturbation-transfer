"""At what granularity does the ICMS causal response transfer across animals?

Four readouts of the same trials, from finest to coarsest, all scored with the
same Delta-R^2 against the same leave-one-animal-out protocol:

  unit          every sorted unit separately
  depth_band    population rate in 300 um bands of cortical depth
  population    mean rate across the recorded population
  behaviour     wheel speed and the time-resolved detection probability

A depth band is the natural cross-animal coordinate for a linear probe: it is
defined by anatomy rather than by which unit happened to be isolated, so it is
comparable between animals in a way that unit indices are not.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.baselines import measured_delta_set

MAX_D = 1900.0
BANDS = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1900)]


def band_project(s):
    """(n_trials, T, n_bands) mean rate per depth band."""
    uy = np.asarray(s.meta["unit_y_um"], float)
    cols = []
    for lo, hi in BANDS:
        m = (uy >= lo) & (uy < hi)
        cols.append(s.y[:, :, m].mean(2) if m.any() else np.full(s.y.shape[:2], np.nan))
    return np.stack(cols, -1)


def build(s, level: str):
    if level == "unit":
        y = s.y
        coord = np.asarray(s.meta["unit_y_um"], float) / MAX_D
    elif level == "depth_band":
        y = band_project(s)
        coord = np.array([(lo + hi) / 2 / MAX_D for lo, hi in BANDS])
    elif level == "population":
        y = s.y.mean(2, keepdims=True)
        coord = np.array([np.nanmean(np.asarray(s.meta["unit_y_um"], float)) / MAX_D])
    elif level == "behaviour":
        y = s.behavior
        coord = np.array([0.0, 0.5, 1.0][: y.shape[-1]])
    else:
        raise ValueError(level)
    return y, coord


def rows(s, level: str):
    y, coord = build(s, level)
    ypost = y[:, s.t0 :]
    base = np.nanmean(ypost[~s.perturbed], 0)
    amp = s.meta["cond_amp"]
    dep = s.meta["cond_depth_um"]
    X, Y, cid, ch = [], [], [], []
    for c in sorted({int(x) for x in s.cond[s.perturbed]}):
        D = np.nanmean(ypost[s.cond == c], 0) - base
        a = float(amp[c]) if c in amp else float(amp[str(c)])
        sd = (float(dep[c]) if c in dep else float(dep[str(c)])) / MAX_D
        an = a / 10.0
        for k in range(D.shape[1]):
            if not np.all(np.isfinite(D[:, k])):
                continue
            dz = coord[k] - sd
            g = [np.exp(-((dz / w) ** 2)) for w in (0.06, 0.15, 0.30, 0.60)]
            X.append([an, an**2, np.sqrt(an), sd, coord[k], dz, abs(dz),
                      *g, *[an * gi for gi in g], an * abs(dz),
                      float(k == 0), float(k), 1.0])
            Y.append(D[:, k])
            cid.append(c)
            ch.append(k)
    return np.array(X, float), np.array(Y, float), np.array(cid), np.array(ch)


def ridge(X, Y, lam):
    return np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--out", type=Path, default=Path("results/tables/granularity.json"))
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]

    out = {}
    print(f"{'level':13s} {'x-animal dR2 [95% CI]':>26s} {'r':>7s} {'ceiling':>8s} {'n_ch':>5s}")
    print("-" * 64)
    for level in ("unit", "depth_band", "population", "behaviour"):
        per = {}
        for s in ds.sets:
            if level == "behaviour" and s.behavior is None:
                continue
            per[s.key] = rows(s, level)
        keyani = {s.key: s.animal for s in ds.sets}
        r2s, rs, ce = [], [], []
        nch = []
        for a in ds.animals:
            tr = [k for k in per if keyani[k] != a]
            X = np.concatenate([per[k][0] for k in tr])
            Y = np.concatenate([per[k][1] for k in tr])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            sd[-1] = 1.0; mu[-1] = 0.0
            W = ridge((X - mu) / sd, Y, args.lam)
            for k in per:
                if keyani[k] != a:
                    continue
                Xs, Ys, cs, chs = per[k]
                P = ((Xs - mu) / sd) @ W
                conds = sorted(set(cs.tolist()))
                nc = int(chs.max()) + 1
                A = np.zeros((len(conds), Ys.shape[1], nc))
                Bm = np.zeros_like(A)
                for i in range(len(cs)):
                    j = conds.index(cs[i])
                    A[j, :, chs[i]] = Ys[i]
                    Bm[j, :, chs[i]] = P[i]
                r2s.append(M.delta_r2(A, Bm)); rs.append(M.corr(A, Bm)); nch.append(nc)
                s = next(x for x in ds.sets if x.key == k)
                yy, _ = build(s, level)
                ce.append(M.noise_ceiling(yy[:, s.t0:], s.cond, s.perturbed, n_splits=100)["delta_r2_ceiling"])
        m, lo, hi = M.bootstrap_ci(r2s)
        rm, _, _ = M.bootstrap_ci(rs)
        out[level] = {"delta_r2": m, "ci": [lo, hi], "delta_corr": rm,
                      "ceiling": float(np.nanmean(ce)), "n_channels": float(np.mean(nch))}
        print(f"{level:13s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'':4s} {rm:+.3f} "
              f"{np.nanmean(ce):8.3f} {np.mean(nch):5.1f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
