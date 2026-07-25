"""How much does the shared operator improve as more animals are added to it?

If what transfers between animals really is one operator, then fitting it on more
animals should make it better on a new one, and the curve should still be rising at
the size of the cohort we have. If instead the apparent transfer were an artefact of
a particular set of animals, adding animals would not help in any orderly way.

For each held-out animal the operator is fitted on random subsets of the remaining
animals, of every size from one upwards, and scored on the individual part of the
held-out animal's response. Several random subsets are drawn at each size so the
curve does not depend on which animals happen to be chosen.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from cadence import individuality as I

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, nargs="+", default=[Path("data/proc/alm.pkl")])
    ap.add_argument("--tag", default="alm")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets = []
    for c in args.cache:
        sets += pickle.load(c.open("rb"))["dataset"].sets
    op = I.SharedOperator()
    for s in sets:
        op.add(s)
    op._prepare()
    animals = sorted({e["animal"] for e in op.ex})
    print(f"{len(sets)} sessions, {len(animals)} animals, {op.K} shared parameters")

    rng = np.random.default_rng(args.seed)
    sizes = [k for k in (1, 2, 3, 5, 8, 12, 16, 19, 25, 30, 38)
             if k <= len(animals) - 1]
    curve = []
    print(f"\n{'n animals':>10s} {'dR2 on delta':>13s} {'sem':>7s} {'positive':>9s}")
    print("-" * 44)
    for k in sizes:
        vals = []
        for a in animals:
            pool = [b for b in animals if b != a]
            for _ in range(args.draws):
                sub = list(rng.choice(pool, size=k, replace=False))
                theta = op.fit(set(animals) - set(sub), args.ridge)
                v = op.evaluate(a, theta)
                if np.isfinite(v):
                    vals.append(v)
        per_animal = np.array([np.mean(vals[i * args.draws : (i + 1) * args.draws])
                               for i in range(len(animals))])
        m = float(np.mean(per_animal))
        sem = float(np.std(per_animal, ddof=1) / np.sqrt(len(per_animal)))
        curve.append(dict(n_animals=k, delta_r2=m, sem=sem,
                          n_positive=int((per_animal > 0).sum()),
                          per_animal=per_animal.tolist()))
        print(f"{k:10d} {m:+13.3f} {sem:7.3f} {int((per_animal > 0).sum()):>5d}/"
              f"{len(per_animal):<3d}", flush=True)

    out = Path(f"results/cohort_scaling_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(curve, indent=1, default=float))
    x = np.log([c["n_animals"] for c in curve])
    y = np.array([c["delta_r2"] for c in curve])
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"\ncorrelation with log cohort size: {r:+.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
