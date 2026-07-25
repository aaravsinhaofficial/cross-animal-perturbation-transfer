"""Does doubling the cohort make the operator better on the same held-out animals?

The cohort-size curve in Section 'How many animals an operator needs' is fitted with a
linear operator, where subsets can be swapped cheaply. This asks the same question of
the trained network, which cannot be refitted for every subset, by comparing two runs
that differ only in how many animals were available to train on.

Both runs hold out the same animals one at a time and score them the same way. The
smaller run had the other animals of one cohort to learn from; the larger run had those
plus a second cohort. Only the animals common to both are compared, so the test set is
identical and the paired difference is attributable to the training cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cadence import metrics as M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=Path, default=Path("results/operator_alm5.json"))
    ap.add_argument("--large", type=Path, default=Path("results/operator_almall.json"))
    ap.add_argument("--tag", default="alm")
    args = ap.parse_args()

    s = json.loads(args.small.read_text())
    b = json.loads(args.large.read_text())
    rep = {}
    print(f"{'readout':10s} {'small':>8s} {'large':>8s} {'diff':>8s} "
          f"{'better':>9s} {'p':>9s}")
    print("-" * 60)
    for key, label in (("blend", "model"), ("operator", "network"),
                       ("group", "stereotype")):
        if key not in s or key not in b:
            continue
        pa, pb = s[key]["per_animal"], b[key]["per_animal"]
        ks = sorted(set(pa) & set(pb))
        if len(ks) < 6:
            continue
        x = np.array([pa[k] for k in ks])
        y = np.array([pb[k] for k in ks])
        t = M.animal_permutation_test(list(y), list(x))
        rep[label] = dict(n=len(ks), small=float(x.mean()), large=float(y.mean()),
                          diff=t["mean_diff"], p=t["p"],
                          n_better=int((y > x).sum()))
        print(f"{label:10s} {x.mean():+8.3f} {y.mean():+8.3f} {t['mean_diff']:+8.3f} "
              f"{int((y > x).sum()):>4d}/{len(ks):<4d} {t['p']:9.3f}")

    rep["n_animals_small"] = len(s["group"]["per_animal"])
    rep["n_animals_large"] = len(b["group"]["per_animal"])
    print(f"\ntraining cohort: {rep['n_animals_small'] - 1} other animals "
          f"versus {rep['n_animals_large'] - 1}")
    out = Path(f"results/cohort_size_effect_{args.tag}.json")
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
