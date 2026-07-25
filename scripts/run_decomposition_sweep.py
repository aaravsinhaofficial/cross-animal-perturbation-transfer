"""What the shared and individual parts of a response mean, checked against truth.

On real recordings we can measure how much of the individual part of a perturbation
response transfers across animals, but we cannot know what the right answer is. In
simulation we can, because the mechanism is a knob.

The synthetic cortex has two recruitment modes. Under ``local`` the set of cells the
stimulus drives is a smooth function of position, so it follows the same rule in every
animal. Under ``sparse`` the total drive is still governed by a shared rule but which
cells receive it is redrawn for each animal, which is what a scattered electrode or a
different implant location would do. A second knob, ``animal_het``, sets how much of
each animal's recurrent circuit is private.

Running the same decomposition used on the real data across that grid says what a
given measured value means: whether a shared operator recovering a tenth of the
individual part indicates mostly private wiring, or a measurement too small to see
what is there.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from cadence import individuality as I
from cadence import metrics as M
from cadence.synthetic_cortex import CortexConfig, build_cortex_dataset

warnings.filterwarnings("ignore")


def run_cell(private: float, het: float, n_obs: int, n_animals: int, seed: int,
             gain_cv: float = 0.3) -> dict:
    cfg = CortexConfig(recruit="mix", recruit_private=private, animal_het=het,
                       n_obs=n_obs, n_animals=n_animals, seed=seed,
                       obs_gain_cv=gain_cv,
                       unperturbed_trials=240, trials_per_cond=40)
    ds = build_cortex_dataset(cfg)
    op = I.SharedOperator()
    for s in ds.sets:
        op.add(s)
    res = op.loao()
    ceil = {s.animal: I.delta_ceiling(s) for s in ds.sets}
    animals = sorted(res)
    vals = np.array([res[a]["delta_r2"] for a in animals])
    ce = np.array([ceil[a] for a in animals])
    rep = M.animal_level_report(list(vals), animals)
    return dict(private=private, gain_cv=gain_cv, animal_het=het, n_obs=n_obs, n_animals=n_animals,
                delta_r2=float(np.nanmean(vals)), ceiling=float(np.nanmean(ce)),
                fraction=float(np.nanmean(vals) / max(np.nanmean(ce), 1e-9)),
                n_positive=int((vals > 0).sum()), n=len(vals),
                p=rep["permutation"]["p"], ci=[rep["ci_lo"], rep["ci_hi"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-animals", type=int, default=10)
    ap.add_argument("--n-obs", type=int, nargs="+", default=[32])
    ap.add_argument("--het", type=float, nargs="+", default=[0.25])
    ap.add_argument("--private", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--gain-cv", type=float, nargs="+", default=[0.3, 0.0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/decomposition_sweep.json"))
    args = ap.parse_args()

    rows = []
    print(f"{'private':>8s} {'gainCV':>7s} {'het':>5s} {'n_obs':>6s} {'delta R2':>9s} "
          f"{'ceiling':>8s} {'fraction':>9s} {'animals>0':>10s} {'p':>9s}")
    print("-" * 82)
    for gcv in args.gain_cv:
        for private in args.private:
            for het in args.het:
                for n_obs in args.n_obs:
                    r = run_cell(private, het, n_obs, args.n_animals, args.seed, gcv)
                    rows.append(r)
                    print(f"{private:8.2f} {gcv:7.2f} {het:5.2f} {n_obs:6d} "
                          f"{r['delta_r2']:+9.3f} {r['ceiling']:8.3f} "
                          f"{r['fraction']:9.3f} "
                          f"{r['n_positive']:>4d}/{r['n']:<4d} {r['p']:9.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nwrote {args.out}")

    lo = [r for r in rows if r["private"] <= 0.01]
    hi = [r for r in rows if r["private"] >= 0.99]
    if lo:
        print(f"a recruitment rule the animals share: "
              f"{np.mean([r['fraction'] for r in lo]):.2f} of the individual part "
              f"recovered")
    if hi:
        print(f"recruitment that belongs to one implant: "
              f"{np.mean([r['fraction'] for r in hi]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
