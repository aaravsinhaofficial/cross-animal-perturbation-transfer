"""Does the explanation hold up in simulation, and would more neurons help?

We claimed two things about the real data. First, that a shared rule gets each
neuron's response shape right and its amplitude wrong. Second, that the amplitude is
unguessable because the electrode drives a sparse scattered set of cells that is
private to that implant. Both are testable in a simulator where we control the
recruitment.

The sweep varies:
  * ``recruit``  local (a smooth rule shared by all animals) vs sparse (a private
    scattered draw per animal and contact),
  * ``n_obs``    how many neurons are recorded at once,
  * ``animal_het``  how different the animals' circuits are.

and scores unit-level, population-level and behavioural transfer with exactly the
machinery used on the mice, including animal-level statistics.

This also checks a prediction we made in the paper: that unit-level transfer should
improve with the number of simultaneously recorded neurons. If recruitment really is
private to the implant, more neurons should *not* rescue it, and we should say so.
"""

from __future__ import annotations

import argparse
import json
import warnings
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
from cadence.synthetic_cortex import CortexConfig, build_cortex_dataset

warnings.filterwarnings("ignore")
BANDS = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1900)]


def collapse(s, D, level):
    if level == "unit":
        return D
    if level == "population":
        return D.mean(1, keepdims=True)
    uy = np.asarray(s.meta["unit_y_um"], float)
    cols = [D[:, m].mean(1) for m in ((uy >= lo) & (uy < hi) for lo, hi in BANDS) if m.any()]
    return np.stack(cols, -1)


def evaluate(ds, levels=("unit", "population")) -> dict:
    cfg = LinearResponseConfig()
    props = {s.key: fit_propagator(s, cfg) for s in ds.sets}
    blocks = {s.key: precompute_blocks(s, cfg, props[s.key][0]) for s in ds.sets}
    out = {}
    for level in levels:
        vals, cors, groups, gains = [], [], [], []
        for s in ds.sets:
            train = [t for t in ds.sets if t.animal != s.animal]
            th = fit_shared_from_blocks(blocks, train, cfg)
            conds = sorted({int(c) for c in s.cond[s.perturbed]})
            pred = predict_delta(s, cfg, props[s.key][0], th, conds=conds)
            ypost = s.y[:, s.t0 :]
            base = ypost[~s.perturbed].mean(0)
            A = np.stack([collapse(s, ypost[s.cond == c].mean(0) - base, level)
                          for c in conds])
            B = np.stack([collapse(s, pred[c], level) for c in conds])
            vals.append(M.delta_r2(A, B))
            cors.append(M.corr(A, B))
            groups.append(s.animal)
            den = float((B * B).sum())
            g = float((A * B).sum() / den) if den > 1e-12 else 1.0
            gains.append(M.delta_r2(A, B * g))          # per-session oracle gain
        rep = M.animal_level_report(vals, groups)
        rep["delta_corr"] = float(np.nanmean(cors))
        rep["oracle_gain_delta_r2"] = float(np.nanmean(gains))
        out[level] = rep
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, nargs="*", default=[12, 24, 48, 96, 200])
    ap.add_argument("--recruit", nargs="*", default=["local", "sparse"])
    ap.add_argument("--animal-het", type=float, nargs="*", default=[0.25])
    ap.add_argument("--n-animals", type=int, default=8)
    ap.add_argument("--trials-per-cond", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("results/cortex_sweep.json"))
    args = ap.parse_args()

    rows = []
    hdr = (f"{'recruit':8s} {'n_obs':>6s} {'het':>5s} "
           f"{'unit dR2':>22s} {'unit r':>7s} {'unit oracle':>11s} "
           f"{'pop dR2':>22s} {'pop r':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for rec in args.recruit:
        for het in args.animal_het:
            for n_obs in args.n_obs:
                cfg = CortexConfig(n_obs=n_obs, recruit=rec, animal_het=het,
                                   n_animals=args.n_animals,
                                   trials_per_cond=args.trials_per_cond, seed=0)
                ds = build_cortex_dataset(cfg)
                res = evaluate(ds)
                u, p = res["unit"], res["population"]
                rows.append({"recruit": rec, "n_obs": n_obs, "animal_het": het,
                             "unit": u, "population": p})
                print(f"{rec:8s} {n_obs:6d} {het:5.2f} "
                      f"{u['animal_mean']:+.3f}[{u['ci_lo']:+.2f},{u['ci_hi']:+.2f}] "
                      f"{u['delta_corr']:+7.3f} {u['oracle_gain_delta_r2']:+11.3f} "
                      f"{p['animal_mean']:+.3f}[{p['ci_lo']:+.2f},{p['ci_hi']:+.2f}] "
                      f"{p['delta_corr']:+7.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
