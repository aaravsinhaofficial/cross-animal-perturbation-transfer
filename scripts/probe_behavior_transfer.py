"""Cross-animal transfer of the time-resolved *behavioural* response to ICMS.

Behaviour is measured in the same physical units in every animal (wheel encoder
velocity; whether and when the animal reported detection), so unlike single-unit
activity it needs no alignment at all -- which makes it the cleanest possible test
of whether a shared causal operator transfers.

Readouts, per 25 ms bin:
  wheel_velocity    signed wheel velocity
  wheel_speed       unsigned wheel speed
  detection_prob    probability the animal has reported detection by time t

Leave-one-animal-out. Each channel is scored separately, with its own split-half
noise ceiling, and reported both raw and as a fraction of that ceiling.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import metrics as M

MAX_D = 1900.0
CHANNELS = ("wheel_velocity", "wheel_speed", "detection_prob")


def rows(s, ch: int):
    b = s.behavior[:, s.t0 :, ch]
    base = b[~s.perturbed].mean(0)
    amp = s.meta["cond_amp"]
    dep = s.meta["cond_depth_um"]
    X, Y, cid = [], [], []
    for c in sorted({int(x) for x in s.cond[s.perturbed]}):
        D = b[s.cond == c].mean(0) - base
        a = float(amp[c]) if c in amp else float(amp[str(c)])
        d = (float(dep[c]) if c in dep else float(dep[str(c)])) / MAX_D
        an = a / 10.0
        X.append([an, an**2, an**3, np.sqrt(an), np.log1p(an), d, an * d, d**2, 1.0])
        Y.append(D)
        cid.append(c)
    return np.array(X, float), np.array(Y, float), np.array(cid)


def ridge(X, Y, lam):
    return np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ Y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--out", type=Path, default=Path("results/tables/behavior_transfer.json"))
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    sets = [s for s in ds.sets if s.behavior is not None]
    print(f"{len(sets)} sessions with behaviour, {len(ds.animals)} animals\n")

    report = {}
    print(f"{'channel':16s} {'dR2 [95% CI]':>24s} {'r':>7s} {'ceiling':>8s} {'frac':>7s}")
    print("-" * 68)
    for ci, name in enumerate(CHANNELS):
        per = {s.key: rows(s, ci) for s in sets}
        keyani = {s.key: s.animal for s in sets}
        r2s, rs, ces, per_animal = [], [], [], {}
        for a in ds.animals:
            tr = [k for k in per if keyani[k] != a]
            if not tr:
                continue
            X = np.concatenate([per[k][0] for k in tr])
            Y = np.concatenate([per[k][1] for k in tr])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            sd[-1] = 1.0; mu[-1] = 0.0
            W = ridge((X - mu) / sd, Y, args.lam)
            av = []
            for s in sets:
                if s.animal != a:
                    continue
                Xs, Ys, cs = per[s.key]
                P = ((Xs - mu) / sd) @ W
                A = Ys[:, :, None]
                Bm = P[:, :, None]
                r2 = M.delta_r2(A, Bm)
                r2s.append(r2); rs.append(M.corr(A, Bm)); av.append(r2)
                bb = s.behavior[:, s.t0 :, ci : ci + 1]
                ces.append(
                    M.noise_ceiling(bb, s.cond, s.perturbed, n_splits=200)["delta_r2_ceiling"]
                )
            per_animal[a] = float(np.mean(av)) if av else float("nan")
        m, lo, hi = M.bootstrap_ci(r2s)
        rm, _, _ = M.bootstrap_ci(rs)
        cm = float(np.nanmean(ces))
        report[name] = {
            "delta_r2": m, "ci": [lo, hi], "delta_corr": rm, "ceiling": cm,
            "frac_of_ceiling": m / cm if cm > 0 else float("nan"),
            "per_animal": per_animal,
            "sessions_above_zero": f"{sum(x > 0 for x in r2s)}/{len(r2s)}",
        }
        print(f"{name:16s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'':2s} {rm:+.3f} {cm:8.3f} "
              f"{report[name]['frac_of_ceiling']:7.2f}")
        print(f"{'':16s} per-animal: " +
              " ".join(f"{k.replace('sub-ICMS','m')}={v:+.2f}" for k, v in per_animal.items()))
        print(f"{'':16s} sessions above zero: {report[name]['sessions_above_zero']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
