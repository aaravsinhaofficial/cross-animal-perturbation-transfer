"""Cross-animal transfer of the photoinhibition response in ALM (DANDI:000009).

Twenty mice, so the animal-level statistics have real power here: the exact
sign-flip permutation over 20 animals can reach p < 1e-5, rather than bottoming out
at 0.031 as it does with six.

This is also the decisive test of the mechanism. Microstimulation drives a sparse
scattered set of cells private to each implant, and single-neuron transfer failed.
Photoinhibition drives a dense local volume by a rule that is the same in every
animal, which is the condition the simulated cortex said should work.

Same protocol as before: hold out a whole animal, fit on the others, calibrate
nothing on the held-out animal's stimulation trials, score the time-resolved effect.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.dose import ridge_solve
from cadence.linear_response import (
    LinearResponseConfig,
    design_for_set,
    fit_propagator,
    fit_shared_from_blocks,
    precompute_blocks,
)

warnings.filterwarnings("ignore")

LEVELS = ("unit", "depth_band", "population", "choice")
N_BANDS = 5


def bands_of(s):
    d = np.asarray(s.meta["unit_y_um"], float)
    if not np.isfinite(d).any() or d.max() == d.min():
        return [np.ones(len(d), bool)]
    edges = np.quantile(d[np.isfinite(d)], np.linspace(0, 1, N_BANDS + 1))
    edges[0] -= 1; edges[-1] += 1
    out = [(d >= lo) & (d < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return [m for m in out if m.any()]


def readout(s, level):
    if level == "unit":
        return s.y[:, s.t0 :]
    if level == "depth_band":
        return np.stack([s.y[:, s.t0 :, m].mean(2) for m in bands_of(s)], -1)
    if level == "population":
        return s.y[:, s.t0 :].mean(2, keepdims=True)
    if level == "choice":
        return s.behavior[:, s.t0 :, 0:1]
    raise ValueError(level)


def coords(s, level):
    d = np.asarray(s.meta["unit_y_um"], float) / 6000.0
    if level == "unit":
        return d
    if level == "depth_band":
        return np.array([d[m].mean() for m in bands_of(s)])
    return np.array([0.0])


_M: dict = {}
_C: dict = {}


def measured(s, level, conds):
    k = (s.key, level)
    if k not in _M:
        Y = readout(s, level)
        base = np.nanmean(Y[~s.perturbed], 0)
        _M[k] = {int(c): np.nanmean(Y[s.cond == c], 0) - base
                 for c in np.unique(s.cond[s.perturbed])}
    return {c: _M[k][c] for c in conds if c in _M[k]}


def ceiling(s, level, conds):
    k = (s.key, level, tuple(sorted(conds)))
    if k not in _C:
        Y = np.nan_to_num(readout(s, level))
        keep = np.isin(s.cond, conds) | (~s.perturbed)
        _C[k] = M.noise_ceiling(Y[keep], s.cond[keep], s.perturbed[keep],
                                n_splits=100)["delta_r2_ceiling"]
    return _C[k]


def params(s, c):
    return (float(s.meta["cond_amp"][c]),
            float(s.meta["cond_galvo"][c][0]), float(s.meta["cond_galvo"][c][1]))


def design(power, gx, gy, coord):
    p = power / 5.0
    dz = coord - gy / 6.0
    g = [np.exp(-((dz / w) ** 2)) for w in (0.05, 0.15, 0.4)]
    return np.array([p, p**2, np.sqrt(max(p, 0)), np.log1p(p), gx, gy, coord,
                     *g, *[p * x for x in g], 1.0])


# ---------------------------------------------------------------------------
def m_zero(train, s, level, ev, ctx):
    T, n = readout(s, level).shape[1:]
    return {c: np.zeros((T, n)) for c in ev}


def _curves(train, level):
    by = {}
    for t in train:
        cs = [int(c) for c in np.unique(t.cond[t.perturbed])]
        dl = measured(t, level, cs)
        for c, D in dl.items():
            by.setdefault(round(params(t, c)[0], 2), []).append(np.nanmean(D, 1))
    return {a: np.nanmean(v, 0) for a, v in by.items()}


def m_group(train, s, level, ev, ctx):
    cur = _curves(train, level)
    if not cur:
        return m_zero(train, s, level, ev, ctx)
    amps = np.array(sorted(cur))
    stack = np.stack([cur[a] for a in amps])
    T, n = readout(s, level).shape[1:]
    out = {}
    for c in ev:
        a = params(s, c)[0]
        cv = (np.stack([np.interp(a, amps, stack[:, t]) for t in range(stack.shape[1])])
              if len(amps) > 1 else stack[0])
        out[c] = np.tile(cv[:, None], (1, n))
    return out


def _rows(t, level, conds):
    dl = measured(t, level, conds)
    co = coords(t, level)
    X, Y, cid, ch = [], [], [], []
    for c in conds:
        if c not in dl:
            continue
        p, gx, gy = params(t, c)
        D = dl[c]
        for k in range(D.shape[1]):
            if not np.all(np.isfinite(D[:, k])):
                continue
            X.append(design(p, gx, gy, co[min(k, len(co) - 1)]))
            Y.append(D[:, k]); cid.append(c); ch.append(k)
    return (np.array(X), np.array(Y), np.array(cid), np.array(ch)) if X else None


def m_dose(train, s, level, ev, ctx):
    got = [_rows(t, level, [int(c) for c in np.unique(t.cond[t.perturbed])]) for t in train]
    got = [g for g in got if g is not None]
    if not got:
        return m_zero(train, s, level, ev, ctx)
    X = np.concatenate([g[0] for g in got]); Y = np.concatenate([g[1] for g in got])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[-1] = 1.0; mu[-1] = 0.0
    W = ridge_solve((X - mu) / sd, Y, 1e-2)
    r = _rows(s, level, ev)
    if r is None:
        return m_zero(train, s, level, ev, ctx)
    Xs, _, cid, ch = r
    P = ((Xs - mu) / sd) @ W
    T, n = readout(s, level).shape[1:]
    out = {int(c): np.zeros((P.shape[1], n)) for c in ev}
    for i in range(len(cid)):
        out[int(cid[i])][:, ch[i]] = P[i]
    return out


def m_lr(train, s, level, ev, ctx):
    lr, blocks, props, dc = ctx["lr"], ctx["blocks"], ctx["props"], ctx["design"]
    th = fit_shared_from_blocks(blocks, train, lr)
    if s.key not in dc:
        dc[s.key] = design_for_set(s, lr, props[s.key][0],
                                   sorted({int(c) for c in s.cond[s.perturbed]}))
    X, _, index = dc[s.key]
    p = X @ th
    T = s.T - s.t0
    full = {c: np.zeros((T, s.n_obs)) for c in {int(i[0]) for i in index}}
    for i, (c, t, n) in enumerate(index):
        full[int(c)][t, n] = p[i]
    if level == "unit":
        return {c: full[c] for c in ev if c in full}
    out = {}
    for c in ev:
        if c not in full:
            continue
        if level == "population":
            out[c] = full[c].mean(1, keepdims=True)
        elif level == "depth_band":
            out[c] = np.stack([full[c][:, m].mean(1) for m in bands_of(s)], -1)
        else:
            out[c] = full[c].mean(1, keepdims=True)
    return out


METHODS = {"zero": m_zero, "group": m_group, "dose": m_dose, "linear_response": m_lr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/alm.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/alm_analysis.json"))
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--min-ceiling", type=float, default=0.15)
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]

    # drop sessions whose effect is too poorly estimated for any model to be scored
    keep = []
    for s in ds.sets:
        cs = [int(c) for c in np.unique(s.cond[s.perturbed])]
        if cs and ceiling(s, "unit", cs) >= args.min_ceiling:
            keep.append(s)
    ds.sets = keep
    print(f"{len(ds.animals)} animals, {len(ds.sets)} sessions after ceiling filter "
          f"(>= {args.min_ceiling})", flush=True)

    lr = LinearResponseConfig(sigmas_um=(200.0, 500.0, 1200.0), n_time_basis=10, max_lag=25)
    props = {s.key: fit_propagator(s, lr) for s in ds.sets}
    print("precomputing blocks ...", flush=True)
    blocks = {s.key: precompute_blocks(s, lr, props[s.key][0]) for s in ds.sets}
    ctx = {"lr": lr, "blocks": blocks, "props": props, "design": {}}

    results = {}
    print(f"\n{'readout':12s} {'method':16s} {'dR2':>7s} {'95% CI':>18s} {'r':>7s} "
          f"{'animals>0':>10s} {'p':>9s}")
    print("-" * 86)
    for level in args.levels:
        for mname, fn in METHODS.items():
            if level == "choice" and mname == "linear_response":
                continue
            vals, cors, ceils, groups = [], [], [], []
            for s in ds.sets:
                ev = [int(c) for c in np.unique(s.cond[s.perturbed])]
                train = [t for t in ds.sets if t.animal != s.animal]
                if not ev or not train:
                    continue
                try:
                    pred = fn(train, s, level, ev, ctx)
                except Exception:
                    continue
                dl = measured(s, level, ev)
                cs = [c for c in ev if c in pred and c in dl]
                if not cs:
                    continue
                A = np.nan_to_num(np.stack([dl[c] for c in cs]))
                B = np.nan_to_num(np.stack([pred[c] for c in cs]))
                v = M.delta_r2(A, B)
                if not np.isfinite(v):
                    continue
                vals.append(v); cors.append(M.corr(A, B))
                ceils.append(ceiling(s, level, cs)); groups.append(s.animal)
            if not vals:
                continue
            rep = M.animal_level_report(vals, groups)
            rep["delta_corr"] = float(np.nanmean(cors))
            rep["ceiling"] = float(np.nanmean(ceils))
            results[f"{level}|{mname}"] = rep
            print(f"{level:12s} {mname:16s} {rep['animal_mean']:+7.3f} "
                  f"[{rep['ci_lo']:+.2f},{rep['ci_hi']:+.2f}]".rjust(19) +
                  f" {rep['delta_corr']:+7.3f} "
                  f"{rep['sign_test']['n_positive']:>4d}/{rep['sign_test']['n']:<4d} "
                  f"{rep['permutation']['p']:9.2e}", flush=True)

    tests = {}
    for level in args.levels:
        a = results.get(f"{level}|linear_response") or results.get(f"{level}|dose")
        b = results.get(f"{level}|group")
        if not a or not b:
            continue
        ks = [k for k in a["per_animal"] if k in b["per_animal"]]
        t = M.animal_permutation_test([a["per_animal"][k] for k in ks],
                                      [b["per_animal"][k] for k in ks])
        tests[level] = t
        print(f"  {level:12s} model vs group average: diff={t['mean_diff']:+.3f} "
              f"p={t['p']:.2e} (n={t['n']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "tests": tests}, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
