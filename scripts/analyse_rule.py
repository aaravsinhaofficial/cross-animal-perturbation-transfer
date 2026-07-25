"""What is the shared rule, stated in one sentence?

The operator transfers, so something about a neuron's ordinary activity must predict
how a perturbation will move it, and the same something must hold in every animal. This
asks what it is, without fitting anything, by correlating the individual part of a
neuron's measured response with three properties measured from its control trials:

  * how fast it fires,
  * how much its firing grows over the delay, which is its preparatory ramp,
  * how strongly it distinguishes the two trial types, which is its selectivity.

Every correlation is computed inside one recording, so nothing is pooled across animals
except the correlations themselves, and each animal contributes one number to an exact
sign test. The control trials that supply the properties are disjoint from the ones that
define the response, so a correlation cannot arise from shared noise. A null column
repeats the whole thing with the perturbed trials replaced by control trials.
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


def properties(s, feat_idx) -> dict[str, np.ndarray]:
    """Per-neuron descriptions, all from the half of the control trials the response
    is not measured against."""
    yc = s.y[feat_idx][:, s.t0 :]
    rate = np.nan_to_num(np.nanmean(yc, (0, 1)))
    pre = np.nan_to_num(np.nanmean(s.y[feat_idx][:, : max(s.t0, 1)], (0, 1)))
    ramp = np.nan_to_num(np.nanmean(yc, 0)).mean(0) - pre
    sel = np.zeros_like(rate)
    if s.behavior is not None:
        ch = s.behavior[feat_idx][:, 0, 0]
        fin = np.isfinite(ch)
        if fin.any() and np.all(np.isin(ch[fin], (0.0, 1.0))):
            L, R = ch == 0, ch == 1
            if L.sum() >= 8 and R.sum() >= 8:
                sel = np.abs(np.nan_to_num(
                    np.nanmean(yc[R], (0,)) - np.nanmean(yc[L], (0,))).mean(0))
    return {"firing rate": rate, "preparatory ramp": ramp, "selectivity": sel}


def response(s, feat_idx, base_idx, null=False):
    """The individual part of each neuron's response, averaged over the window, one
    value per (neuron, condition)."""
    Y = s.y[:, s.t0 :]
    if null:
        h = len(base_idx) // 2
        rng = np.random.default_rng(abs(hash(s.key)) % (2 ** 31))
        sh = rng.permutation(base_idx)
        src_all, base_idx = np.sort(sh[:h]), np.sort(sh[h:])
    base = np.nanmean(Y[base_idx], 0)
    out = []
    for c in np.unique(s.cond[s.perturbed]):
        m = np.flatnonzero(s.cond == c)
        if len(m) < 6:
            continue
        src = src_all if null else m
        d = I.centre(np.nanmean(Y[src], 0) - base)
        out.append(np.nan_to_num(d).mean(0))                  # (n_obs,)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, nargs="+",
                    default=[Path("data/proc/alm.pkl"), Path("data/proc/alm_wide.pkl")])
    ap.add_argument("--tag", default="almall")
    ap.add_argument("--min-units", type=int, default=8)
    args = ap.parse_args()

    ds = pickle.load(args.cache[0].open("rb"))["dataset"]
    for c in args.cache[1:]:
        ds.sets = list(ds.sets) + list(pickle.load(c.open("rb"))["dataset"].sets)
    print(f"{len(ds.sets)} sessions, {len(ds.animals)} animals")

    names = ["firing rate", "preparatory ramp", "selectivity"]
    per: dict[str, dict[str, list]] = {n: {} for n in names}
    per_null: dict[str, dict[str, list]] = {n: {} for n in names}
    for s in ds.sets:
        if s.n_obs < args.min_units:
            continue
        feat_idx, base_idx = I.control_split(s)
        props = properties(s, feat_idx)
        for null, store in ((False, per), (True, per_null)):
            for d in response(s, feat_idx, base_idx, null=null):
                for n in names:
                    x = props[n]
                    if np.std(x) < 1e-9 or np.std(d) < 1e-9:
                        continue
                    store[n].setdefault(s.animal, []).append(
                        float(np.corrcoef(x, d)[0, 1]))

    rep = {}
    print(f"\n{'property':20s} {'r':>7s} {'animals<0':>10s} {'sign p':>8s} "
          f"{'null r':>8s} {'null<0':>8s}")
    print("-" * 68)
    for n in names:
        an = sorted(per[n])
        v = np.array([float(np.nanmean(per[n][a])) for a in an])
        vn = np.array([float(np.nanmean(per_null[n].get(a, [np.nan]))) for a in an])
        st = M.animal_sign_test(list(-v))          # is the correlation negative?
        rep[n] = dict(mean_r=float(np.nanmean(v)), n=len(v),
                      n_negative=int((v < 0).sum()), p=st["p"],
                      null_mean_r=float(np.nanmean(vn)),
                      null_negative=int(np.nansum(vn < 0)),
                      per_animal={a: float(np.nanmean(per[n][a])) for a in an})
        print(f"{n:20s} {np.nanmean(v):+7.3f} {int((v < 0).sum()):>4d}/{len(v):<4d} "
              f"{st['p']:8.3f} {np.nanmean(vn):+8.3f} "
              f"{int(np.nansum(vn < 0)):>4d}/{len(vn):<4d}")

    out = Path(f"results/rule_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
