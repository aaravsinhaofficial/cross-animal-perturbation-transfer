"""From a predicted neural effect to a predicted change of mind.

Silencing frontal cortex during the delay changes what the animal does next. If the
shared operator really captures the causal effect of the light rather than a
correlation, then the behavioural consequence should follow from the neural one, in an
animal whose perturbation trials were never read.

So we build the chain explicitly. For a held out animal the operator predicts, neuron
by neuron, how the light will move activity. We project that predicted movement onto
that animal's own choice axis, which is measured from control trials as the difference
between the two trial types, and read off how far the light is predicted to push the
population toward one choice. One shared coefficient, fitted on the other animals,
turns that displacement into a predicted change in the probability of licking right.

Three things are compared, all leave one animal out and all scored on the measured
behavioural change:

  * the light does nothing,
  * the average behavioural change of the other animals at the same dose and site,
    which is the stereotype and a strong baseline,
  * the stereotype plus the neural chain above.

If the third beats the second, then knowing this animal's own neurons tells you
something about this animal's behaviour that the other animals could not.
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


def choice_axis(s, feat_idx) -> np.ndarray | None:
    """The direction in which the population differs between the two trial types."""
    if s.behavior is None:
        return None
    yc = s.y[feat_idx][:, s.t0 :]
    ch = s.behavior[feat_idx][:, 0, 0]
    fin = np.isfinite(ch)
    if not (fin.any() and np.all(np.isin(ch[fin], (0.0, 1.0)))):
        return None
    L, R = (ch == 0), (ch == 1)
    if L.sum() < 8 or R.sum() < 8:
        return None
    ax = np.nanmean(yc[R], 0) - np.nanmean(yc[L], 0)          # (T, n_obs)
    ax = np.nan_to_num(ax).mean(0)                             # per neuron
    n = np.linalg.norm(ax)
    return ax / n if n > 0 else None


def behaviour_effect(s, feat_idx, base_idx):
    """Measured change in the probability of licking right, per condition."""
    if s.behavior is None:
        return {}
    ch = s.behavior[:, 0, 0]
    base = np.nanmean(ch[base_idx])
    if not np.isfinite(base):
        return {}
    out = {}
    for c in np.unique(s.cond[s.perturbed]):
        m = s.cond == c
        if m.sum() < 6:
            continue
        v = np.nanmean(ch[m])
        if np.isfinite(v):
            out[int(c)] = float(v - base)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, nargs="+",
                    default=[Path("data/proc/alm.pkl"), Path("data/proc/alm_wide.pkl")])
    ap.add_argument("--preds", type=Path, nargs="*",
                    default=[Path("results/preds_almall.npz")])
    ap.add_argument("--tag", default="almall")
    args = ap.parse_args()

    ds = pickle.load(args.cache[0].open("rb"))["dataset"]
    for c in args.cache[1:]:
        ds.sets = list(ds.sets) + list(pickle.load(c.open("rb"))["dataset"].sets)

    pred = {}
    for p in args.preds:
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        for k in sorted({f.split("|")[0] for f in z.files}):
            if f"{k}|B" in z.files and f"{k}|cond" in z.files:
                pred[k] = (z[f"{k}|B"], [int(c) for c in z[f"{k}|cond"]])
    print(f"{len(ds.sets)} sessions, {len(ds.animals)} animals, "
          f"{len(pred)} with saved neural predictions")

    # one row per (session, condition): the measured behavioural change, the dose and
    # site, and how far the predicted neural effect moves the animal's own choice axis
    rows = []
    for s in ds.sets:
        feat_idx, base_idx = I.control_split(s)
        beh = behaviour_effect(s, feat_idx, base_idx)
        if not beh:
            continue
        ax = choice_axis(s, feat_idx)
        got = pred.get(s.key)
        for c, dv in beh.items():
            proj = np.nan
            if ax is not None and got is not None and c in got[1]:
                B = got[0][got[1].index(c)]                    # (T, n_obs)
                nb = int(s.meta.get("delay_bins", B.shape[0]))
                nb = int(np.clip(nb, 1, B.shape[0]))
                # only while the light is on, and in units of this animal's own
                # activity so that one shared coefficient means the same thing
                # in every animal
                yc = s.y[feat_idx][:, s.t0 :]
                sc = max(float(np.nanstd(yc.reshape(-1, s.n_obs), axis=0).mean()), 1e-3)
                proj = float(np.nanmean(np.nan_to_num(B[:nb]) @ ax) / sc)
            site = s.meta.get("cond_site", {}).get(c, "")
            gx = s.meta.get("cond_galvo", {}).get(c, (0.0, 0.0))[0]
            rows.append(dict(animal=s.animal, key=s.key, cond=c, dbeh=dv,
                             amp=float(s.meta["cond_amp"][c]),
                             side=float(gx), site=str(site), proj=proj))
    print(f"{len(rows)} (session, condition) rows, "
          f"{sum(np.isfinite(r['proj']) for r in rows)} with a neural projection")

    animals = sorted({r["animal"] for r in rows})

    def stereotype(exclude):
        """Mean behavioural change of the other animals, as a function of dose and
        which side the light went to."""
        tab: dict[tuple, list] = {}
        for r in rows:
            if r["animal"] in exclude:
                continue
            tab.setdefault((round(r["side"]), round(r["amp"] / 10.0)), []).append(r["dbeh"])
        glob = float(np.mean([r["dbeh"] for r in rows if r["animal"] not in exclude]))

        def f(r):
            k = (round(r["side"]), round(r["amp"] / 10.0))
            v = tab.get(k)
            return float(np.mean(v)) if v else glob

        return f

    def chain_coef(exclude):
        """One shared coefficient turning a predicted displacement along the choice
        axis into a predicted change in behaviour, fitted on the other animals."""
        num = den = 0.0
        for r in rows:
            if r["animal"] in exclude or not np.isfinite(r["proj"]):
                continue
            g = stereotype(exclude | {r["animal"]})(r)
            num += (r["dbeh"] - g) * r["proj"]
            den += r["proj"] ** 2
        return num / den if den > 0 else 0.0

    res = {k: [] for k in ("zero", "stereotype", "stereotype_plus_neural")}
    groups = []
    for a in animals:
        f = stereotype({a})
        k = chain_coef({a})
        mine = [r for r in rows if r["animal"] == a]
        y = np.array([r["dbeh"] for r in mine])
        g = np.array([f(r) for r in mine])
        p = np.array([r["proj"] if np.isfinite(r["proj"]) else 0.0 for r in mine])
        e = float(np.sum(y ** 2))
        if e <= 0 or len(y) < 2:
            continue
        res["zero"].append(0.0)
        res["stereotype"].append(1.0 - float(np.sum((y - g) ** 2)) / e)
        res["stereotype_plus_neural"].append(
            1.0 - float(np.sum((y - g - k * p) ** 2)) / e)
        groups.append(a)

    rep = {}
    print(f"\n{'model':24s} {'median':>8s} {'mean':>8s} {'95% CI':>16s} "
          f"{'animals>0':>10s} {'sign p':>8s}")
    print("-" * 80)
    for k, v in res.items():
        r = M.animal_level_report(v, groups)
        r["median"] = float(np.median(v))
        rep[k] = r
        print(f"{k:24s} {r['median']:+8.3f} {r['animal_mean']:+8.3f} "
              f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]".rjust(17) +
              f" {r['sign_test']['n_positive']:>4d}/{r['sign_test']['n']:<4d} "
              f"{r['sign_test']['p']:8.3f}")

    t = M.animal_permutation_test(res["stereotype_plus_neural"], res["stereotype"])
    rep["test_chain_vs_stereotype"] = t
    print(f"neural chain vs stereotype: diff={t['mean_diff']:+.3f} p={t['p']:.2e} "
          f"(n={t['n']})")
    t2 = M.animal_permutation_test(res["stereotype"], res["zero"])
    rep["test_stereotype_vs_zero"] = t2
    print(f"stereotype vs nothing:      diff={t2['mean_diff']:+.3f} p={t2['p']:.2e}")

    rep["n_rows"] = len(rows)
    out = Path(f"results/behaviour_transfer_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
