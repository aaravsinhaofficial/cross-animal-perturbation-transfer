"""How much of what is individual about a neuron's response transfers across animals?

The measured effect on neuron n is split into the part the whole population shares
and the part specific to that neuron,

    Delta_n(t)  =  mean over neurons of Delta(t)   +   delta_n(t)

and every model is scored on delta alone. A stereotype built from other animals
predicts the same curve for every neuron, so its delta is zero and its score is
exactly zero. Anything above zero is prediction of individual structure in an animal
whose perturbation trials were never seen.

Two models are scored: a shared linear operator acting on each neuron's own control
activity, fitted leave-one-animal-out with its ridge chosen inside the training
animals, and the learned operator whose per-neuron predictions were saved during
training. Alongside them is the ceiling, which is how much of delta is measurable
rather than noise, so the numbers can be read as a fraction of what is there.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from cadence import individuality as I
from cadence import metrics as M

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/alm.pkl"))
    ap.add_argument("--preds", type=Path, default=None)
    ap.add_argument("--tag", default="alm")
    args = ap.parse_args()

    ds = pickle.load(args.cache.open("rb"))["dataset"]
    print(f"{len(ds.sets)} sessions, {len(ds.animals)} animals")

    # --- the shared linear operator, leave one animal out -------------------
    op = I.SharedOperator()
    for s in ds.sets:
        op.add(s)
    shared = op.loao()

    # --- the same analysis where the answer is known to be zero -------------
    # the stimulation trials are replaced by a second group of unperturbed trials, so
    # there is no effect to find; anything above zero here would be an artefact
    nullop = I.SharedOperator()
    for s in ds.sets:
        nullop.add(s, null=True)
    null = nullop.loao()

    # --- ceilings, per session then per animal ------------------------------
    ceil_rows, ceil_groups = [], []
    for s in ds.sets:
        v = I.delta_ceiling(s)
        if np.isfinite(v):
            ceil_rows.append(v)
            ceil_groups.append(s.animal)

    # --- the learned operator, from its saved per-neuron predictions ---------
    learned: dict[str, list] = {}
    if args.preds is not None and args.preds.exists():
        z = np.load(args.preds, allow_pickle=True)
        for k in sorted({f.split("|")[0] for f in z.files}):
            if f"{k}|A" not in z.files:
                continue
            A = I.centre(z[f"{k}|A"])
            B = I.centre(z[f"{k}|B"])
            learned.setdefault(str(z[f"{k}|animal"][0]), []).append(I.score(A, B))

    animals = sorted(shared)
    rows = {"stereotype": [0.0] * len(animals),
            "no effect present": [null[a]["delta_r2"] for a in animals
                                  ] if null else [],
            "shared_operator": [shared[a]["delta_r2"] for a in animals]}
    rows = {k: v for k, v in rows.items() if v}
    if learned:
        rows["learned_operator"] = [float(np.nanmean(learned.get(a, [np.nan])))
                                    for a in animals]

    rep = {}
    # The per-animal score divides by that animal's own effect energy, so an animal
    # with a small effect can return a wildly negative number and drag a mean around.
    # The headline test is therefore the exact sign test over animals, which asks the
    # question we care about (does this help in an animal, more often than not) and
    # cannot be moved by one extreme value. The mean and its interval are reported
    # next to it.
    print(f"\n{'model':18s} {'median':>8s} {'mean':>8s} {'95% CI':>16s} "
          f"{'animals>0':>10s} {'sign p':>8s} {'perm p':>9s}")
    print("-" * 84)
    for k, v in rows.items():
        ok = [i for i, x in enumerate(v) if np.isfinite(x)]
        r = M.animal_level_report([v[i] for i in ok], [animals[i] for i in ok])
        r["median"] = float(np.median([v[i] for i in ok]))
        rep[k] = r
        print(f"{k:18s} {r['median']:+8.3f} {r['animal_mean']:+8.3f} "
              f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]".rjust(17) +
              f" {r['sign_test']['n_positive']:>4d}/{r['sign_test']['n']:<4d} "
              f"{r['sign_test']['p']:8.3f} {r['permutation']['p']:9.3f}")

    ce = M.animal_level_report(ceil_rows, ceil_groups)
    rep["ceiling"] = ce
    print(f"{'ceiling':18s} {ce['animal_mean']:+13.3f}")
    for k in rows:
        if k == "stereotype":
            continue
        f = rep[k]["animal_mean"] / max(ce["animal_mean"], 1e-9)
        rep[f"{k}_fraction_of_ceiling"] = float(f)
        print(f"  {k} recovers {100*f:.0f}% of the measurable individual structure")

    if "learned_operator" in rows:
        ks = [a for a in animals
              if a in rep["learned_operator"]["per_animal"]
              and a in rep["shared_operator"]["per_animal"]]
        t = M.animal_permutation_test(
            [rep["learned_operator"]["per_animal"][a] for a in ks],
            [rep["shared_operator"]["per_animal"][a] for a in ks])
        rep["test_learned_vs_shared"] = t
        print(f"learned vs shared operator: diff={t['mean_diff']:+.3f} p={t['p']:.2e}")

    per = {}
    print(f"\n{'animal':20s} {'ceiling':>8s} {'shared':>8s} {'learned':>8s} {'ridge':>7s}")
    cpa = ce["per_animal"]
    for a in animals:
        per[a] = dict(ceiling=float(cpa.get(a, np.nan)),
                      shared=float(shared[a]["delta_r2"]),
                      ridge=float(shared[a]["ridge"]),
                      learned=float(np.nanmean(learned[a])) if a in learned else None)
        lv = per[a]["learned"]
        print(f"{a:20s} {per[a]['ceiling']:8.3f} {per[a]['shared']:+8.3f} "
              f"{(lv if lv is not None else float('nan')):+8.3f} "
              f"{per[a]['ridge']:7.3g}")
    rep["per_animal_detail"] = per

    # Does what transfers track how well the animal could be measured? The ceiling is
    # fixed by trial counts and firing rates before any model is fitted, so this is a
    # statement about the recordings rather than a property of the fit.
    from scipy.stats import spearmanr

    c = np.array([per[a]["ceiling"] for a in animals])
    v = np.array([per[a]["shared"] for a in animals])
    ok = np.isfinite(c) & np.isfinite(v)
    if ok.sum() >= 6:
        r, p = spearmanr(c[ok], v[ok])
        rep["ceiling_vs_transfer"] = dict(rho=float(r), p=float(p), n=int(ok.sum()))
        print(f"\nhow measurable versus how much transfers: rho = {r:+.2f}, "
              f"p = {p:.3f}, n = {int(ok.sum())}")

    out = Path(f"results/individuality_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
