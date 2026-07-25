"""The ICMS results table: what transfers, at what granularity, how far.

Rows are **readouts** of the same trials, from single units to behaviour. Columns
are **generalisation levels**, from in-sample to a new animal *and* an
intervention amplitude deleted from every training animal. Every cell is scored
with the same statistic (Delta-R^2 against the no-effect model) and carries its
own split-half noise ceiling.

Models
------
``shared_operator``
    The hierarchical controlled model. For unit-level activity the animal's
    propagator ``A_i`` is estimated from its unperturbed trials and convolved with
    a species-invariant drive fitted on other animals (``cadence.linear_response``).
    For the low-dimensional readouts the shared operator is a smooth function of
    the physical intervention parameters, applied to the held-out animal without
    any of its intervention data.
``physical_ridge``
    Same information, but a direct regression with no dynamics: the ablation that
    shows what the propagator buys.
``no_effect``
    Asserts the intervention does nothing. Defines Delta-R^2 = 0.

Nothing from a held-out animal's intervention trials is used at any point.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.linear_response import (
    LinearResponseConfig,
    design_for_set,
    fit_propagator,
    fit_shared_from_blocks,
    precompute_blocks,
    predict_delta,
)

MAX_D = 1900.0
BANDS = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1900)]
LEVELS = ("unit", "depth_band", "population", "wheel_speed", "detection_prob")


# ---------------------------------------------------------------------------
# readouts
# ---------------------------------------------------------------------------
def readout(s, level: str) -> np.ndarray:
    """(n_trials, T_post, n_channels) for the requested readout."""
    if level == "unit":
        return s.y[:, s.t0 :]
    if level == "depth_band":
        uy = np.asarray(s.meta["unit_y_um"], float)
        # only bands that actually contain units: zero-filling empty bands would
        # add noiseless channels and inflate the noise ceiling
        cols = [s.y[:, s.t0 :, m].mean(2)
                for m in ((uy >= lo) & (uy < hi) for lo, hi in BANDS) if m.any()]
        return np.stack(cols, -1)
    if level == "population":
        return s.y[:, s.t0 :].mean(2, keepdims=True)
    if level == "wheel_speed":
        return s.behavior[:, s.t0 :, 1:2]
    if level == "detection_prob":
        return s.behavior[:, s.t0 :, 2:3]
    raise ValueError(level)


def channel_coords(s, level: str) -> np.ndarray:
    if level == "unit":
        return np.asarray(s.meta["unit_y_um"], float) / MAX_D
    if level == "depth_band":
        uy = np.asarray(s.meta["unit_y_um"], float)
        return np.array([(lo + hi) / 2 / MAX_D for lo, hi in BANDS
                         if ((uy >= lo) & (uy < hi)).any()])
    return np.array([0.0])


def measured(s, level: str, conds):
    Y = readout(s, level)
    base = np.nanmean(Y[~s.perturbed], 0)
    return {int(c): np.nanmean(Y[s.cond == c], 0) - base for c in conds}


def cond_params(s, c):
    amp, dep = s.meta["cond_amp"], s.meta["cond_depth_um"]
    a = float(amp[c]) if c in amp else float(amp[str(c)])
    d = float(dep[c]) if c in dep else float(dep[str(c)])
    return a, d


# ---------------------------------------------------------------------------
# the low-dimensional shared operator (population / behaviour readouts)
# ---------------------------------------------------------------------------
from cadence.dose import dose_design, ridge_solve  # noqa: E402


def dose_rows(s, level, conds):
    coords = channel_coords(s, level)
    dl = measured_cached(s, level, conds)
    X, Y, cid, ch = [], [], [], []
    for c in conds:
        a, d = cond_params(s, c)
        D = dl[c]
        for k in range(D.shape[1]):
            if not np.all(np.isfinite(D[:, k])):
                continue
            X.append(dose_design(a, d, coords[min(k, len(coords) - 1)]))
            Y.append(D[:, k]); cid.append(c); ch.append(k)
    if not X:
        return None
    return np.array(X, float), np.array(Y, float), np.array(cid), np.array(ch)


def fit_dose(sets, level, conds_of, lam=1e-2):
    rows = [dose_rows(s, level, conds_of(s)) for s in sets]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    X = np.concatenate([r[0] for r in rows]); Y = np.concatenate([r[1] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[-1] = 1.0; mu[-1] = 0.0
    return mu, sd, ridge_solve((X - mu) / sd, Y, lam)


def predict_dose(s, level, conds, model):
    mu, sd, W = model
    r = dose_rows(s, level, conds)
    if r is None:
        return {}
    X, _, cid, ch = r
    P = ((X - mu) / sd) @ W
    nch = readout(s, level).shape[2]
    T = P.shape[1]
    out = {int(c): np.zeros((T, nch)) for c in conds}
    for i in range(len(cid)):
        out[int(cid[i])][:, ch[i]] = P[i]
    return out


# ---------------------------------------------------------------------------
_CEIL: dict = {}
_MEAS: dict = {}


def ceiling_for(s, level, conds, n_splits=120):
    """Split-half ceiling. Depends only on the data, the readout and which
    conditions are scored -- never on the model -- so it is cached."""
    ck = (s.key, level, tuple(sorted(conds)))
    if ck not in _CEIL:
        Y = readout(s, level)
        keep = np.isin(s.cond, conds) | (~s.perturbed)
        _CEIL[ck] = M.noise_ceiling(
            np.nan_to_num(Y[keep]), s.cond[keep], s.perturbed[keep], n_splits=n_splits
        )["delta_r2_ceiling"]
    return _CEIL[ck]


def measured_cached(s, level, conds):
    ck = (s.key, level, tuple(sorted(conds)))
    if ck not in _MEAS:
        _MEAS[ck] = measured(s, level, conds)
    return _MEAS[ck]


def score(s, level, conds, pred):
    dl = measured_cached(s, level, conds)
    cs = [c for c in conds if c in pred]
    if not cs:
        return None
    A = np.stack([np.nan_to_num(dl[c]) for c in cs])
    B = np.stack([np.nan_to_num(pred[c]) for c in cs])
    return {
        "delta_r2": M.delta_r2(A, B),
        "delta_corr": M.corr(A, B),
        "ceiling": ceiling_for(s, level, cs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/icms_ladder.json"))
    ap.add_argument("--md", type=Path, default=Path("results/tables/icms_ladder.md"))
    ap.add_argument("--holdout-amps", type=float, nargs="*", default=[5.0])
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    sets = ds.sets
    lr = LinearResponseConfig()
    props = {s.key: fit_propagator(s, lr) for s in sets}
    all_conds = {s.key: sorted({int(c) for c in s.cond[s.perturbed]}) for s in sets}
    print("precomputing unit-level design blocks ...", flush=True)
    blocks = {s.key: precompute_blocks(s, lr, props[s.key][0]) for s in sets}
    design_cache: dict[str, tuple] = {}
    print("done", flush=True)

    def predict_unit(s, th, ev):
        if s.key not in design_cache:
            design_cache[s.key] = design_for_set(s, lr, props[s.key][0], all_conds[s.key])
        X, _, index = design_cache[s.key]
        p = X @ th
        T = s.T - s.t0
        out = {c: np.zeros((T, s.n_obs)) for c in ev}
        for i, (c, t, n) in enumerate(index):
            if c in out:
                out[c][t, n] = p[i]
        return out

    def amps_of(s, cs):
        return {c: cond_params(s, c)[0] for c in cs}

    held = set(args.holdout_amps)

    def conds_excl_held(s):
        return [c for c in all_conds[s.key] if cond_params(s, c)[0] not in held]

    def conds_only_held(s):
        return [c for c in all_conds[s.key] if cond_params(s, c)[0] in held]

    results: dict = {}
    print(f"{'readout':15s} {'generalisation':26s} {'method':16s} "
          f"{'dR2 [95% CI]':>24s} {'r':>7s} {'ceil':>6s} {'frac':>6s} {'>0':>7s}")
    print("-" * 118)

    for level in LEVELS:
        use = [s for s in sets if not (level in ("wheel_speed", "detection_prob")
                                       and s.behavior is None)]
        for gen in ("in_sample", "cross_session", "cross_animal", "cross_animal_unseen_amp"):
            for method in ("shared_operator", "physical_ridge", "no_effect"):
                if level != "unit" and method == "physical_ridge":
                    continue          # for low-D readouts the two coincide
                if level == "unit" and method == "shared_operator":
                    pass
                r2s, rs, ces = [], [], []
                per_animal: dict[str, list[float]] = {}
                for s in use:
                    if gen == "cross_animal_unseen_amp":
                        ev = conds_only_held(s)
                        train = [t for t in use if t.animal != s.animal]
                        tr_conds = conds_excl_held
                    elif gen == "cross_animal":
                        ev = all_conds[s.key]
                        train = [t for t in use if t.animal != s.animal]
                        tr_conds = lambda t: all_conds[t.key]  # noqa: E731
                    elif gen == "cross_session":
                        ev = all_conds[s.key]
                        train = [t for t in use if t.animal == s.animal and t.key != s.key]
                        tr_conds = lambda t: all_conds[t.key]  # noqa: E731
                    else:
                        ev = all_conds[s.key]
                        train = [s]
                        tr_conds = lambda t: all_conds[t.key]  # noqa: E731
                    if not ev or not train:
                        continue
                    if method == "no_effect":
                        T = readout(s, level).shape[1]
                        nch = readout(s, level).shape[2]
                        pred = {c: np.zeros((T, nch)) for c in ev}
                    elif level == "unit" and method == "shared_operator":
                        try:
                            th = fit_shared_from_blocks(
                                blocks, train, lr,
                                cond_filter=lambda t, c: c in tr_conds(t),
                            )
                            pred = predict_unit(s, th, ev)
                        except Exception:
                            continue
                    else:
                        mdl = fit_dose(train, level, tr_conds)
                        if mdl is None:
                            continue
                        pred = predict_dose(s, level, ev, mdl)
                    sc = score(s, level, ev, pred)
                    if sc is None or not np.isfinite(sc["delta_r2"]):
                        continue
                    r2s.append(sc["delta_r2"]); rs.append(sc["delta_corr"])
                    ces.append(sc["ceiling"])
                    per_animal.setdefault(s.animal, []).append(sc["delta_r2"])
                if not r2s:
                    continue
                m, lo, hi = M.bootstrap_ci(r2s)
                rm, _, _ = M.bootstrap_ci(rs)
                cm = float(np.nanmean(ces))
                key = f"{level}|{gen}|{method}"
                results[key] = {
                    "level": level, "generalisation": gen, "method": method,
                    "delta_r2": m, "ci": [lo, hi], "delta_corr": rm, "ceiling": cm,
                    "frac_of_ceiling": m / cm if cm > 0 else float("nan"),
                    "n_sessions": len(r2s),
                    "sessions_above_zero": int(sum(x > 0 for x in r2s)),
                    "per_animal": {k: float(np.mean(v)) for k, v in per_animal.items()},
                    "per_session": r2s,
                }
                print(f"{level:15s} {gen:26s} {method:16s} "
                      f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]{'':2s} {rm:+.3f} {cm:6.3f} "
                      f"{results[key]['frac_of_ceiling']:6.2f} "
                      f"{results[key]['sessions_above_zero']:3d}/{len(r2s)}", flush=True)

    # paired tests: shared operator vs no-effect, per readout, cross-animal
    tests = {}
    for level in LEVELS:
        a = results.get(f"{level}|cross_animal|shared_operator")
        b = results.get(f"{level}|cross_animal|no_effect")
        if not a or not b:
            continue
        diff, p = M.paired_permutation_test(a["per_session"], b["per_session"])
        _, pw = M.wilcoxon_signed_rank(a["per_session"], b["per_session"])
        tests[level] = {"mean_diff": diff, "p_perm": p, "wilcoxon_p": pw,
                        "n": len(a["per_session"])}
    print("\npaired tests (cross-animal, shared_operator vs no_effect):")
    for k, v in tests.items():
        print(f"  {k:15s} diff={v['mean_diff']:+.3f} p_perm={v['p_perm']:.2e} "
              f"wilcoxon={v['wilcoxon_p']:.2e} n={v['n']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "tests": tests}, indent=1))

    # markdown table
    lines = ["| readout | generalisation | ΔR² | 95% CI | r | ceiling | frac | sessions>0 |",
             "|---|---|---|---|---|---|---|---|"]
    for level in LEVELS:
        for gen in ("in_sample", "cross_session", "cross_animal", "cross_animal_unseen_amp"):
            k = f"{level}|{gen}|shared_operator"
            if k not in results:
                continue
            r = results[k]
            lines.append(
                f"| {level} | {gen} | {r['delta_r2']:+.3f} | "
                f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['delta_corr']:+.3f} | "
                f"{r['ceiling']:.3f} | {r['frac_of_ceiling']:.2f} | "
                f"{r['sessions_above_zero']}/{r['n_sessions']} |"
            )
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
