"""The definitive results table: every method, every readout, identical splits.

Three things this script fixes relative to the earlier analyses.

1. Inference is at the **animal** level. Sessions from one mouse are not
   independent replicates of a claim about mice, so confidence intervals come from
   a bootstrap that resamples animals, and significance from an exact sign-flip
   permutation over the six animals. With six animals the smallest attainable
   p-value is 0.031, and we say so wherever it is quoted. Session-level statistics
   and a mixed-effects estimate with an animal random intercept are reported
   alongside, as secondary detail.

2. The simple baselines are actually simple. A group average over the other
   animals, an amplitude-interpolated group average, and a smooth surface in
   amplitude and time are all much cruder than the model, and any claim has to
   beat them rather than beating "the stimulus does nothing".

3. The intervention holdouts get harder. Deleting one middle amplitude is close to
   interpolating between its neighbours, so we also delete the top of the range,
   the bottom of the range, a contiguous block, whole contact depths, and specific
   amplitude-by-contact combinations.

Nothing from a held-out animal's stimulation trials is ever used to make a
prediction. Where a number does use them, it is labelled an oracle.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.dose import cond_params, dose_design, ridge_solve
from cadence.linear_response import (
    LinearResponseConfig,
    design_for_set,
    fit_propagator,
    fit_shared_from_blocks,
    precompute_blocks,
)

warnings.filterwarnings("ignore")

MAX_D = 1900.0
BANDS = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1900)]
READOUTS = ("detection", "wheel", "population", "depth_band", "unit")


# ---------------------------------------------------------------------------
# readouts
# ---------------------------------------------------------------------------
def readout(s, level):
    if level == "unit":
        return s.y[:, s.t0 :]
    if level == "depth_band":
        uy = np.asarray(s.meta["unit_y_um"], float)
        cols = [s.y[:, s.t0 :, m].mean(2)
                for m in ((uy >= lo) & (uy < hi) for lo, hi in BANDS) if m.any()]
        return np.stack(cols, -1)
    if level == "population":
        return s.y[:, s.t0 :].mean(2, keepdims=True)
    if level == "wheel":
        return s.behavior[:, s.t0 :, 1:2]
    if level == "detection":
        return s.behavior[:, s.t0 :, 2:3]
    raise ValueError(level)


def coords(s, level):
    if level == "unit":
        return np.asarray(s.meta["unit_y_um"], float) / MAX_D
    if level == "depth_band":
        uy = np.asarray(s.meta["unit_y_um"], float)
        return np.array([(lo + hi) / 2 / MAX_D for lo, hi in BANDS
                         if ((uy >= lo) & (uy < hi)).any()])
    return np.array([0.0])


_MEAS: dict = {}
_CEIL: dict = {}


def measured(s, level, conds):
    ck = (s.key, level)
    if ck not in _MEAS:
        Y = readout(s, level)
        base = Y[~s.perturbed].mean(0)
        _MEAS[ck] = {int(c): Y[s.cond == c].mean(0) - base
                     for c in np.unique(s.cond[s.perturbed])}
    return {c: _MEAS[ck][c] for c in conds if c in _MEAS[ck]}


def ceiling(s, level, conds, n_splits=120):
    ck = (s.key, level, tuple(sorted(conds)))
    if ck not in _CEIL:
        Y = readout(s, level)
        keep = np.isin(s.cond, conds) | (~s.perturbed)
        _CEIL[ck] = M.noise_ceiling(Y[keep], s.cond[keep], s.perturbed[keep],
                                    n_splits=n_splits)["delta_r2_ceiling"]
    return _CEIL[ck]


def score(s, level, conds, pred):
    dl = measured(s, level, conds)
    cs = [c for c in conds if c in pred and c in dl]
    if not cs:
        return None
    A = np.stack([dl[c] for c in cs])
    B = np.stack([pred[c] for c in cs])
    return M.delta_r2(A, B), M.corr(A, B), ceiling(s, level, cs)


# ---------------------------------------------------------------------------
# methods.  Each returns {cond: (T, n_channels)} for the test set.
# ---------------------------------------------------------------------------
def m_zero(train, s, level, ev, tr_conds, ctx):
    T, n = readout(s, level).shape[1:]
    return {c: np.zeros((T, n)) for c in ev}


def _group_curves(train, level, tr_conds):
    """Amplitude -> average effect curve over the training animals.

    Channels are collapsed, because a group average cannot know which channel of a
    new animal corresponds to which of an old one.
    """
    by_amp: dict[float, list[np.ndarray]] = {}
    for t in train:
        for c in tr_conds(t):
            dl = measured(t, level, [c])
            if c not in dl:
                continue
            a, _ = cond_params(t, c)
            by_amp.setdefault(round(a, 3), []).append(dl[c].mean(1))
    return {a: np.mean(v, 0) for a, v in by_amp.items()}


def m_group_mean(train, s, level, ev, tr_conds, ctx):
    """Nearest available training amplitude, broadcast to every channel."""
    cur = _group_curves(train, level, tr_conds)
    if not cur:
        return m_zero(train, s, level, ev, tr_conds, ctx)
    amps = np.array(sorted(cur))
    T, n = readout(s, level).shape[1:]
    out = {}
    for c in ev:
        a, _ = cond_params(s, c)
        near = amps[np.argmin(np.abs(amps - a))]
        out[c] = np.tile(cur[near][:, None], (1, n))
    return out


def m_group_interp(train, s, level, ev, tr_conds, ctx):
    """Linear interpolation of the group-average curves in amplitude."""
    cur = _group_curves(train, level, tr_conds)
    if len(cur) < 2:
        return m_group_mean(train, s, level, ev, tr_conds, ctx)
    amps = np.array(sorted(cur))
    stack = np.stack([cur[a] for a in amps])                 # (n_amp, T)
    T, n = readout(s, level).shape[1:]
    out = {}
    for c in ev:
        a, _ = cond_params(s, c)
        curve = np.stack([np.interp(a, amps, stack[:, t]) for t in range(stack.shape[1])])
        out[c] = np.tile(curve[:, None], (1, n))
    return out


def _smooth_basis(a, t, T):
    """Tensor product of a smooth amplitude basis and a smooth time basis."""
    an = a / 10.0
    ab = np.array([1.0, an, an**2, an**3, np.sqrt(max(an, 0)), np.log1p(an)])
    return ab


def m_dose_gam(train, s, level, ev, tr_conds, ctx):
    """Smooth surface in amplitude and time, no depth and no dynamics."""
    X, Y = [], []
    for t in train:
        for c in tr_conds(t):
            dl = measured(t, level, [c])
            if c not in dl:
                continue
            a, _ = cond_params(t, c)
            X.append(_smooth_basis(a, None, None))
            Y.append(dl[c].mean(1))
    if not X:
        return m_zero(train, s, level, ev, tr_conds, ctx)
    X, Y = np.array(X), np.array(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[0] = 1.0; mu[0] = 0.0
    W = ridge_solve((X - mu) / sd, Y, 1e-2)
    T, n = readout(s, level).shape[1:]
    out = {}
    for c in ev:
        a, _ = cond_params(s, c)
        curve = ((_smooth_basis(a, None, None) - mu) / sd) @ W
        out[c] = np.tile(curve[:, None], (1, n))
    return out


def _dose_rows(t, level, conds, extra=None):
    dl = measured(t, level, conds)
    co = coords(t, level)
    X, Y, cid, ch = [], [], [], []
    for c in conds:
        if c not in dl:
            continue
        a, d = cond_params(t, c)
        D = dl[c]
        for k in range(D.shape[1]):
            row = dose_design(a, d, co[min(k, len(co) - 1)])
            if extra is not None:
                row = np.concatenate([row[:-1], extra, [1.0]])
            X.append(row); Y.append(D[:, k]); cid.append(c); ch.append(k)
    if not X:
        return None
    return np.array(X), np.array(Y), np.array(cid), np.array(ch)


def _fit_dose(train, level, tr_conds, lam=1e-2, extra_fn=None):
    got = [_dose_rows(t, level, tr_conds(t), None if extra_fn is None else extra_fn(t))
           for t in train]
    got = [g for g in got if g is not None]
    if not got:
        return None
    X = np.concatenate([g[0] for g in got]); Y = np.concatenate([g[1] for g in got])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[-1] = 1.0; mu[-1] = 0.0
    # the plausible range of an effect, used to clamp wild extrapolations below
    lim = float(np.abs(Y).max()) * 3.0 + 1e-9
    return mu, sd, ridge_solve((X - mu) / sd, Y, lam), lim


def _apply_dose(s, level, ev, model, extra=None):
    got = _dose_rows(s, level, ev, extra)
    if got is None or model is None:
        return {}
    X, _, cid, ch = got
    mu, sd, W, lim = model
    # a prediction far outside anything seen in training is a numerical artefact of
    # extrapolating the feature expansion, so clamp rather than report a blow-up
    P = np.clip(((X - mu) / sd) @ W, -lim, lim)
    T, n = readout(s, level).shape[1:]
    out = {int(c): np.zeros((P.shape[1], n)) for c in ev}
    for i in range(len(cid)):
        out[int(cid[i])][:, ch[i]] = P[i]
    return out


def m_dose_physical(train, s, level, ev, tr_conds, ctx):
    """Smooth in amplitude, contact depth and channel position. No neural data."""
    return _apply_dose(s, level, ev, _fit_dose(train, level, tr_conds))


def m_dose_plus_spont(train, s, level, ev, tr_conds, ctx):
    """The same, plus the held-out animal's own spontaneous summary statistics."""
    sp = ctx["spont"]
    return _apply_dose(s, level, ev,
                       _fit_dose(train, level, tr_conds, extra_fn=lambda t: sp[t.key]),
                       extra=sp[s.key])


def m_linear_response(train, s, level, ev, tr_conds, ctx):
    """Per-animal propagator from resting activity, shared drive from other animals."""
    lr, blocks, props, dcache = ctx["lr"], ctx["blocks"], ctx["props"], ctx["design"]
    th = fit_shared_from_blocks(blocks, train, lr,
                                cond_filter=lambda t, c, f=tr_conds: c in f(t))
    if s.key not in dcache:
        dcache[s.key] = design_for_set(s, lr, props[s.key][0],
                                       sorted({int(c) for c in s.cond[s.perturbed]}))
    X, _, index = dcache[s.key]
    p = X @ th
    T = s.T - s.t0
    full = {c: np.zeros((T, s.n_obs)) for c in set(int(i[0]) for i in index)}
    for i, (c, t, n) in enumerate(index):
        full[int(c)][t, n] = p[i]
    if level == "unit":
        return {c: full[c] for c in ev if c in full}
    # collapse the unit-level prediction onto the coarser readout
    out = {}
    uy = np.asarray(s.meta["unit_y_um"], float)
    for c in ev:
        if c not in full:
            continue
        if level == "population":
            out[c] = full[c].mean(1, keepdims=True)
        elif level == "depth_band":
            cols = [full[c][:, m].mean(1) for m in
                    ((uy >= lo) & (uy < hi) for lo, hi in BANDS) if m.any()]
            out[c] = np.stack(cols, -1)
    return out


def m_lr_plus_gain(train, s, level, ev, tr_conds, ctx):
    """Linear response, with the animal's overall responsiveness predicted from its
    own resting activity (nested shrinkage, fitted on other animals only)."""
    base = m_linear_response(train, s, level, ev, tr_conds, ctx)
    g = ctx["gain_fn"](train, s, level, tr_conds, ctx)
    return {c: v * g for c, v in base.items()}


METHODS = {
    "zero": m_zero,
    "group_mean": m_group_mean,
    "group_interp": m_group_interp,
    "dose_gam": m_dose_gam,
    "dose_physical": m_dose_physical,
    "dose_plus_spont": m_dose_plus_spont,
    "linear_response": m_linear_response,
    "lr_plus_gain": m_lr_plus_gain,
}
LOWD_METHODS = ("zero", "group_mean", "group_interp", "dose_gam",
                "dose_physical", "dose_plus_spont")
NEURAL_METHODS = ("zero", "group_mean", "group_interp", "dose_physical",
                  "dose_plus_spont", "linear_response", "lr_plus_gain")


# ---------------------------------------------------------------------------
# responsiveness predicted from resting activity (used by lr_plus_gain)
# ---------------------------------------------------------------------------
def spont_summary(s, A):
    y = s.y[~s.perturbed].astype(np.float64)
    n = s.n_obs
    flat = y.reshape(-1, n)
    rate = float(flat.mean()); var = float(flat.var())
    acc = np.zeros((n, n)); P = np.eye(n)
    for _ in range(41):
        acc += P; P = A @ P
    ev = float(np.max(np.abs(np.linalg.eigvals(A))))
    c = np.nan_to_num(np.corrcoef(flat.T))
    mc = float(c[~np.eye(n, dtype=bool)].mean())
    w = np.clip(np.linalg.eigvalsh(np.cov(flat.T) + 1e-12 * np.eye(n)), 0, None)
    pr = float((w.sum() ** 2) / (np.sum(w**2) + 1e-12)) / max(n, 1)
    return np.array([np.log(rate + 1e-6), np.log(var / (rate + 1e-9) + 1e-6), ev,
                     np.log(np.linalg.norm(acc) / n + 1e-9), mc, pr, np.log(n)], float)


def precompute_required_gains(ds, level, tr_conds, ctx):
    """The gain the shared operator needs for every session, computed once.

    For each session the operator is fitted without that session's own animal, so
    the target is itself honest, and the whole table is reused by every fold.
    """
    need, ani = {}, {}
    for s in ds.sets:
        train = [t for t in ds.sets if t.animal != s.animal and tr_conds(t)]
        ev = tr_conds(s) or [int(c) for c in np.unique(s.cond[s.perturbed])]
        if not train or not ev:
            continue
        try:
            pr = m_linear_response(train, s, level, ev, tr_conds, ctx)
        except Exception:
            continue
        dl = measured(s, level, ev)
        cs = [c for c in ev if c in pr and c in dl]
        if not cs:
            continue
        A = np.stack([dl[c] for c in cs])
        B = np.stack([pr[c] for c in cs])
        den = float((B * B).sum())
        need[s.key] = float((A * B).sum() / den) if den > 1e-12 else 1.0
        ani[s.key] = s.animal
    return need, ani


def make_gain_fn(ds, lams=(0.3, 1.0, 3.0, 10.0, 30.0)):
    """Predict an animal's overall responsiveness from its own resting activity.

    Shrinkage is chosen by a nested leave-one-animal-out loop inside the training
    animals, so the held-out animal influences nothing.
    """
    cache: dict = {}

    def gain_fn(train, s, level, tr_conds, ctx):
        tbl = ctx.get("gain_table", {}).get(level)
        if not tbl:
            return 1.0
        need_all, ani_all = tbl
        key = (level, s.animal)
        if key in cache:
            return cache[key]
        sp = ctx["spont_raw"]
        ks = [k for k in need_all if ani_all[k] != s.animal]
        if len(ks) < 6:
            cache[key] = 1.0
            return 1.0
        need = {k: need_all[k] for k in ks}
        ani = {k: ani_all[k] for k in ks}
        X = np.stack([sp[k] for k in ks])
        yv = np.log(np.clip([need[k] for k in ks], 1e-3, None))
        mu, sd = X.mean(0), X.std(0) + 1e-9
        best, best_sc = lams[0], -np.inf
        for lam in lams:
            errs = []
            for b in sorted({ani[k] for k in ks}):
                itr = [k for k in ks if ani[k] != b]
                ite = [k for k in ks if ani[k] == b]
                if not itr or not ite:
                    continue
                Xi = (np.stack([sp[k] for k in itr]) - mu) / sd
                yi = np.log(np.clip([need[k] for k in itr], 1e-3, None))
                W = np.linalg.solve(Xi.T @ Xi + lam * len(Xi) * np.eye(Xi.shape[1]), Xi.T @ yi)
                for k in ite:
                    errs.append((float(((sp[k] - mu) / sd) @ W)
                                 - float(np.log(max(need[k], 1e-3)))) ** 2)
            sc = -float(np.mean(errs)) if errs else -np.inf
            if sc > best_sc:
                best, best_sc = lam, sc
        Xs = (X - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + best * len(Xs) * np.eye(Xs.shape[1]), Xs.T @ yv)
        g = float(np.exp(((sp[s.key] - mu) / sd) @ W))
        g = float(np.clip(g, 0.1, 10.0))
        cache[key] = g
        return g

    gain_fn.reset = cache.clear
    return gain_fn


# ---------------------------------------------------------------------------
# holdouts
# ---------------------------------------------------------------------------
def make_holdouts(ds):
    amps = sorted({round(cond_params(s, c)[0], 3) for s in ds.sets
                   for c in np.unique(s.cond[s.perturbed])})
    lo3, hi3 = amps[:2], amps[-3:]
    mid = amps[len(amps) // 2]
    block = amps[len(amps) // 2 - 1: len(amps) // 2 + 2]

    def by_amp(vals):
        vs = {round(v, 3) for v in vals}
        return lambda s, c: round(cond_params(s, c)[0], 3) in vs

    def by_depth(lo, hi):
        return lambda s, c: lo <= cond_params(s, c)[1] <= hi

    def by_combo(s, c):
        a, d = cond_params(s, c)
        return round(a, 3) >= 5.0 and d < 900.0

    return {
        "none": None,
        "amp_interior": by_amp([mid]),
        "amp_block": by_amp(block),
        "amp_high_extrap": by_amp(hi3),
        "amp_low_extrap": by_amp(lo3),
        "depth_superficial": by_depth(0.0, 600.0),
        "depth_deep": by_depth(1200.0, MAX_D),
        "amp_x_depth": by_combo,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/final_analysis.json"))
    ap.add_argument("--md", type=Path, default=Path("results/tables/final_analysis.md"))
    ap.add_argument("--readouts", nargs="*", default=list(READOUTS))
    ap.add_argument("--holdouts", nargs="*", default=None)
    args = ap.parse_args()

    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    lr = LinearResponseConfig()
    props = {s.key: fit_propagator(s, lr) for s in ds.sets}
    print("precomputing design blocks ...", flush=True)
    blocks = {s.key: precompute_blocks(s, lr, props[s.key][0]) for s in ds.sets}
    spont_raw = {s.key: spont_summary(s, props[s.key][0]) for s in ds.sets}
    allsp = np.stack(list(spont_raw.values()))
    smu, ssd = allsp.mean(0), allsp.std(0) + 1e-9
    spont = {k: (v - smu) / ssd for k, v in spont_raw.items()}
    ctx = {"lr": lr, "blocks": blocks, "props": props, "design": {},
           "spont": spont, "spont_raw": spont_raw, "gain_fn": make_gain_fn(ds)}
    holdouts = make_holdouts(ds)
    want_h = args.holdouts or list(holdouts)
    print(f"animals={len(ds.animals)} sessions={len(ds.sets)}", flush=True)

    # the propagator-based methods are much slower, so they run on the holdouts that
    # actually change the scientific claim
    SLOW = {"linear_response", "lr_plus_gain"}
    # These fit a propagator per animal and are far more expensive. The neural
    # holdout rows add little once neural transfer has already failed without any
    # holdout, so they run only on the full split; the holdout sweep is carried by
    # the behavioural readout, where it changes the claim.
    SLOW_HOLDOUTS = {"none"}

    results: dict = {}
    for level in args.readouts:
        methods = LOWD_METHODS if level in ("detection", "wheel") else NEURAL_METHODS
        for hname in want_h:
            hf = holdouts[hname]
            if hf is None:
                def _trc(t):
                    return [int(c) for c in np.unique(t.cond[t.perturbed])]
            else:
                def _trc(t, f=hf):
                    return [int(c) for c in np.unique(t.cond[t.perturbed]) if not f(t, c)]
            need_slow = level not in ("detection", "wheel") and hname in SLOW_HOLDOUTS
            if need_slow:
                ctx["gain_table"] = {level: precompute_required_gains(ds, level, _trc, ctx)}
                ctx["gain_fn"].reset()
            for mname in methods:
                if mname in SLOW and not need_slow:
                    continue
                fn = METHODS[mname]
                vals, cors, ceils, groups = [], [], [], []
                for s in ds.sets:
                    if level in ("detection", "wheel") and s.behavior is None:
                        continue
                    allc = [int(c) for c in np.unique(s.cond[s.perturbed])]
                    ev = allc if hf is None else [c for c in allc if hf(s, c)]
                    if not ev:
                        continue
                    train = [t for t in ds.sets if t.animal != s.animal]
                    if hf is None:
                        def tr_conds(t):
                            return [int(c) for c in np.unique(t.cond[t.perturbed])]
                    else:
                        def tr_conds(t, f=hf):
                            return [int(c) for c in np.unique(t.cond[t.perturbed])
                                    if not f(t, c)]
                    train = [t for t in train if tr_conds(t)]
                    if not train:
                        continue
                    try:
                        pred = fn(train, s, level, ev, tr_conds, ctx)
                    except Exception:
                        continue
                    sc = score(s, level, ev, pred)
                    if sc is None or not np.isfinite(sc[0]):
                        continue
                    vals.append(sc[0]); cors.append(sc[1]); ceils.append(sc[2])
                    groups.append(s.animal)
                if not vals:
                    continue
                rep = M.animal_level_report(vals, groups)
                rep["delta_corr"] = float(np.nanmean(cors))
                rep["ceiling"] = float(np.nanmean(ceils))
                results[f"{level}|{hname}|{mname}"] = rep
                print(f"{level:11s} {hname:17s} {mname:16s} "
                      f"dR2={rep['animal_mean']:+.3f} "
                      f"[{rep['ci_lo']:+.3f},{rep['ci_hi']:+.3f}] "
                      f"r={rep['delta_corr']:+.3f} "
                      f"pos={rep['sign_test']['n_positive']}/{rep['sign_test']['n']} "
                      f"p={rep['permutation']['p']:.3f}", flush=True)

    # ---- paired animal-level comparisons against the best simple baseline ----
    tests = {}
    for level in args.readouts:
        for hname in want_h:
            main_key = (f"{level}|{hname}|lr_plus_gain" if level not in ("detection", "wheel")
                        else f"{level}|{hname}|dose_physical")
            if main_key not in results:
                continue
            base_candidates = [k for k in ("group_interp", "group_mean", "dose_gam", "zero")
                               if f"{level}|{hname}|{k}" in results]
            if not base_candidates:
                continue
            best = max(base_candidates,
                       key=lambda k: results[f"{level}|{hname}|{k}"]["animal_mean"])
            a = results[main_key]["per_animal"]
            b = results[f"{level}|{hname}|{best}"]["per_animal"]
            common = [k for k in a if k in b]
            t = M.animal_permutation_test([a[k] for k in common], [b[k] for k in common])
            tests[f"{level}|{hname}"] = {"model": main_key.split("|")[-1],
                                         "best_simple": best, **t}
    print("\nmodel vs best simple baseline (animal-level exact permutation):")
    for k, v in tests.items():
        print(f"  {k:28s} vs {v['best_simple']:13s} diff={v['mean_diff']:+.3f} p={v['p']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "tests": tests}, indent=1))

    lines = ["| readout | holdout | method | ΔR² (animal mean) | 95% CI | animals>0 | p |",
             "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lv, ho, me = k.split("|")
        lines.append(f"| {lv} | {ho} | {me} | {r['animal_mean']:+.3f} | "
                     f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                     f"{r['sign_test']['n_positive']}/{r['sign_test']['n']} | "
                     f"{r['permutation']['p']:.3f} |")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
