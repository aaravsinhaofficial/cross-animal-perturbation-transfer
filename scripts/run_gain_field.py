"""Does a shared operator acting on each animal's own activity beat a group average?

The earlier models predicted a stereotyped response from the stimulus settings, so a
group average over the other animals matched them. This one predicts an individual
response, because it multiplies a shared modulation field by each neuron's own
control firing profile, which is available without ever stimulating that animal.

Runs leave-one-animal-out on either dataset, scores the time-resolved effect at
single-neuron resolution, and tests the model against the group average with an
exact sign-flip permutation over animals.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from cadence import gain_field as GF
from cadence import metrics as M

warnings.filterwarnings("ignore")


def load(cache: Path, min_ceiling: float):
    ds = pickle.load(cache.open("rb"))["dataset"]
    _M: dict = {}

    def measured(s, conds):
        if s.key not in _M:
            Y = s.y[:, s.t0 :]
            base = np.nanmean(Y[~s.perturbed], 0)
            _M[s.key] = {int(c): np.nanmean(Y[s.cond == c], 0) - base
                         for c in np.unique(s.cond[s.perturbed])}
        return {c: _M[s.key][c] for c in conds if c in _M[s.key]}

    def ceil(s):
        cs = [int(c) for c in np.unique(s.cond[s.perturbed])]
        if not cs:
            return 0.0
        y = np.nan_to_num(s.y[:, s.t0 :])
        k = np.isin(s.cond, cs) | (~s.perturbed)
        return M.noise_ceiling(y[k], s.cond[k], s.perturbed[k],
                               n_splits=80)["delta_r2_ceiling"]

    if min_ceiling > 0:
        ds.sets = [s for s in ds.sets if ceil(s) >= min_ceiling]
    return ds, measured, ceil


def group_average(train, s, conds, measured):
    """Interpolated group-average effect curve, broadcast to every neuron."""
    by: dict[float, list] = {}
    for t in train:
        cs = [int(c) for c in np.unique(t.cond[t.perturbed])]
        for c, D in measured(t, cs).items():
            by.setdefault(round(float(t.meta["cond_amp"][c]), 3), []).append(
                np.nanmean(D, 1))
    if not by:
        return {}
    amps = np.array(sorted(by))
    stack = np.stack([np.nanmean(by[a], 0) for a in amps])
    T = s.T - s.t0
    out = {}
    for c in conds:
        a = float(s.meta["cond_amp"][c])
        cv = (np.stack([np.interp(a, amps, stack[:, t]) for t in range(stack.shape[1])])
              if len(amps) > 1 else stack[0])
        out[c] = np.tile(cv[:, None], (1, s.n_obs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/alm.pkl"))
    ap.add_argument("--tag", default="alm")
    ap.add_argument("--min-ceiling", type=float, default=0.15)
    ap.add_argument("--ridge", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--n-time-basis", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"results/gain_field_{args.tag}.json")

    ds, measured, ceil = load(args.cache, args.min_ceiling)
    print(f"{ds.name}: {len(ds.animals)} animals, {len(ds.sets)} sessions", flush=True)
    conds_of = lambda t: [int(c) for c in np.unique(t.cond[t.perturbed])]  # noqa: E731

    base_cfg = GF.GainFieldConfig(n_time_basis=args.n_time_basis)
    print("precomputing designs ...", flush=True)
    # cache the design itself, so both the fit and every prediction reuse it
    designs = {s.key: GF.build_rows(s, conds_of(s), base_cfg, measured) for s in ds.sets}
    blocks = {k: (None if g is None else (g[0].T @ g[0], g[0].T @ g[1], len(g[1])))
              for k, g in designs.items()}
    pool = GF.BlockPool(blocks, ds.sets)
    print(f"done ({sum(g[0].nbytes for g in designs.values() if g) / 1e9:.1f} GB)",
          flush=True)

    # nested selection of the ridge, inside the training animals only
    def score_sets(sets_to_score, theta, cfg):
        v = []
        for s in sets_to_score:
            pr = GF.predict_from_design(designs.get(s.key), s, conds_of(s), theta)
            dl = measured(s, conds_of(s))
            cs = [c for c in pr if c in dl]
            if not cs:
                continue
            A = np.nan_to_num(np.stack([dl[c] for c in cs]))
            B = np.nan_to_num(np.stack([pr[c] for c in cs]))
            r = M.delta_r2(A, B)
            if np.isfinite(r):
                v.append(r)
        return v

    rows = {"gain_field": [], "group": [], "zero": []}
    groups, cors, ceils, chosen = [], [], [], []
    for a in ds.animals:
        train = [t for t in ds.sets if t.animal != a]
        test = [t for t in ds.sets if t.animal == a]
        if not train or not test:
            continue
        best, best_sc = args.ridge[0], -np.inf
        for lam in args.ridge:
            cfg = GF.GainFieldConfig(n_time_basis=args.n_time_basis, ridge=lam)
            inner = []
            for b in {t.animal for t in train}:
                ite = [t for t in train if t.animal == b]
                if not ite:
                    continue
                th = pool.solve([a, b], cfg)
                inner += score_sets(ite, th, cfg)
            sc = float(np.mean(inner)) if inner else -np.inf
            if sc > best_sc:
                best, best_sc = lam, sc
        chosen.append(best)
        cfg = GF.GainFieldConfig(n_time_basis=args.n_time_basis, ridge=best)
        theta = pool.solve([a], cfg)
        for s in test:
            cs0 = conds_of(s)
            dl = measured(s, cs0)
            pr = GF.predict_from_design(designs.get(s.key), s, cs0, theta)
            ga = group_average(train, s, cs0, measured)
            cs = [c for c in cs0 if c in dl and c in pr]
            if not cs:
                continue
            A = np.nan_to_num(np.stack([dl[c] for c in cs]))
            B = np.nan_to_num(np.stack([pr[c] for c in cs]))
            G = np.nan_to_num(np.stack([ga.get(c, np.zeros_like(dl[c])) for c in cs]))
            v = M.delta_r2(A, B)
            if not np.isfinite(v):
                continue
            rows["gain_field"].append(v)
            rows["group"].append(M.delta_r2(A, G))
            rows["zero"].append(0.0)
            cors.append(M.corr(A, B))
            ceils.append(ceil(s))
            groups.append(a)
        print(f"  [{a}] ridge={best:g} dR2={rows['gain_field'][-1]:+.3f} "
              f"group={rows['group'][-1]:+.3f}", flush=True)

    res = {}
    print(f"\n{'method':12s} {'dR2':>7s} {'95% CI':>18s} {'r':>7s} {'animals>0':>10s} {'p':>10s}")
    print("-" * 70)
    for k in ("zero", "group", "gain_field"):
        rep = M.animal_level_report(rows[k], groups)
        rep["delta_corr"] = float(np.nanmean(cors)) if k == "gain_field" else None
        res[k] = rep
        print(f"{k:12s} {rep['animal_mean']:+7.3f} "
              f"[{rep['ci_lo']:+.2f},{rep['ci_hi']:+.2f}]".rjust(19) +
              f" {(rep['delta_corr'] or float('nan')):+7.3f} "
              f"{rep['sign_test']['n_positive']:>4d}/{rep['sign_test']['n']:<4d} "
              f"{rep['permutation']['p']:10.2e}")
    pa_g = res["gain_field"]["per_animal"]
    pa_b = res["group"]["per_animal"]
    ks = [k for k in pa_g if k in pa_b]
    t = M.animal_permutation_test([pa_g[k] for k in ks], [pa_b[k] for k in ks])
    res["test_vs_group"] = t
    res["ceiling"] = float(np.nanmean(ceils))
    res["ridges"] = chosen
    print(f"\ngain field vs group average: diff={t['mean_diff']:+.3f} p={t['p']:.2e} "
          f"(n={t['n']} animals, floor {t['p_floor']:.2e})")
    print(f"ceiling {res['ceiling']:.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
