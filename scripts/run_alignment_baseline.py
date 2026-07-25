"""Align the animals first, then carry the perturbation response across.

This is the obvious thing to do and the thing most of the cross-animal literature does:
find a common latent space from unperturbed activity, put every animal in it, and move
whatever you want to transfer through that space. If two animals' activity really is
the same up to a change of basis, then a perturbation response should carry across the
change of basis too.

Concretely, for a held out animal and each training animal we take the mean control
activity of both, reduce each to its leading components over the scored window, and find
the orthogonal map that best takes one set of component time courses onto the other.
That map, and it alone, then carries the training animal's measured effect onto the held
out animal's neurons. Averaging over training animals gives the prediction. Nothing from
the held out animal's perturbation trials is used, which is the same protocol every other
model here obeys.

The point of running it is that it separates two things the paper keeps distinguishing.
An alignment is a statement about the geometry of ordinary activity. The operator is a
statement about what a perturbation does. If the geometry were enough, this baseline
would do as well.
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


def profile(s, feat_idx, n_comp):
    """Leading components of this recording's mean control activity.

    Returns the component time courses (T, k) and the loadings (n_obs, k) that put a
    pattern over neurons into component space and back.
    """
    yc = s.y[feat_idx][:, s.t0 :]
    r = np.nan_to_num(np.nanmean(yc, 0))                    # (T, n_obs)
    r = r - r.mean(0, keepdims=True)
    k = min(n_comp, min(r.shape) - 1)
    if k < 2:
        return None
    u, sv, vt = np.linalg.svd(r, full_matrices=False)
    return u[:, :k] * sv[:k], vt[:k].T                      # (T,k), (n_obs,k)


def orthogonal_map(a, b):
    """The rotation taking the columns of ``a`` onto the columns of ``b``."""
    m = a.T @ b
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def measured(s):
    Y = s.y[:, s.t0 :]
    _, base_idx = I.control_split(s)
    base = np.nanmean(Y[base_idx], 0)
    return {int(c): np.nan_to_num(np.nanmean(Y[s.cond == c], 0) - base)
            for c in np.unique(s.cond[s.perturbed])
            if (s.cond == c).sum() >= 6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, nargs="+",
                    default=[Path("data/proc/alm.pkl")])
    ap.add_argument("--tag", default="alm")
    ap.add_argument("--n-comp", type=int, default=8)
    args = ap.parse_args()

    ds = pickle.load(args.cache[0].open("rb"))["dataset"]
    for c in args.cache[1:]:
        ds.sets = list(ds.sets) + list(pickle.load(c.open("rb"))["dataset"].sets)
    print(f"{len(ds.sets)} sessions, {len(ds.animals)} animals, "
          f"{args.n_comp} components")

    pre = {}
    for s in ds.sets:
        feat_idx, _ = I.control_split(s)
        got = profile(s, feat_idx, args.n_comp)
        if got is None:
            continue
        dl = measured(s)
        if not dl:
            continue
        pre[s.key] = dict(s=s, time=got[0], load=got[1], meas=dl,
                          amp={c: float(s.meta["cond_amp"][c]) for c in dl})

    rows = {"aligned": [], "stereotype": []}
    groups = []
    for s in ds.sets:
        me = pre.get(s.key)
        if me is None:
            continue
        k = me["time"].shape[1]
        for c, truth in me["meas"].items():
            amp = me["amp"][c]
            preds, stereo = [], []
            for key, other in pre.items():
                if other["s"].animal == s.animal:
                    continue
                kk = min(k, other["time"].shape[1])
                # the training animal's condition closest in dose to this one
                cj = min(other["meas"], key=lambda x: abs(other["amp"][x] - amp))
                if abs(other["amp"][cj] - amp) > 0.5 * max(amp, 1.0):
                    continue
                dj = other["meas"][cj]                       # (T, n_j)
                # into the training animal's components, rotate into ours, back out
                zj = dj @ other["load"][:, :kk]              # (T, kk)
                R = orthogonal_map(other["time"][:, :kk], me["time"][:, :kk])
                preds.append(zj @ R @ me["load"][:, :kk].T)  # (T, n_i)
                stereo.append(np.tile(dj.mean(1)[:, None], (1, s.n_obs)))
            if not preds:
                continue
            P = np.mean(preds, 0)
            G = np.mean(stereo, 0)
            e = float(np.nansum(truth ** 2))
            if e <= 0:
                continue
            rows["aligned"].append(1.0 - float(np.nansum((truth - P) ** 2)) / e)
            rows["stereotype"].append(1.0 - float(np.nansum((truth - G) ** 2)) / e)
            groups.append(s.animal)

    rep = {}
    print(f"\n{'model':24s} {'median':>8s} {'mean':>8s} {'95% CI':>16s} "
          f"{'animals>0':>10s} {'sign p':>8s}")
    print("-" * 80)
    for kx, v in rows.items():
        r = M.animal_level_report(v, groups)
        r["median"] = float(np.median(list(r["per_animal"].values())))
        rep[kx] = r
        print(f"{kx:24s} {r['median']:+8.3f} {r['animal_mean']:+8.3f} "
              f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]".rjust(17) +
              f" {r['sign_test']['n_positive']:>4d}/{r['sign_test']['n']:<4d} "
              f"{r['sign_test']['p']:8.3f}")
    ks = sorted(set(rep["aligned"]["per_animal"]) & set(rep["stereotype"]["per_animal"]))
    t = M.animal_permutation_test([rep["aligned"]["per_animal"][k] for k in ks],
                                  [rep["stereotype"]["per_animal"][k] for k in ks])
    rep["test_aligned_vs_stereotype"] = t
    print(f"aligned vs stereotype: diff={t['mean_diff']:+.3f} p={t['p']:.3f} (n={t['n']})")

    out = Path(f"results/alignment_baseline_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
