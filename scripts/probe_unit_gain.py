"""Is a *per-unit* gain predictable from non-interventional data?

The readout decomposition showed that one scalar per unit takes cross-animal
unit-level Delta-R^2 from ~0 to ~0.4, so the missing quantity is precisely a per-unit
amplitude. This script asks whether that amplitude can be predicted without any
intervention data from the held-out animal.

Target: for each (session, unit), the scalar the shared operator needs.
Predictors, all computable from unperturbed trials and static metadata:
  * the unit's depth, and its offset from the session's stimulating contacts
  * its firing statistics (rate, Fano, autocorrelation, population coupling, rank)
  * its spontaneous fluctuation-response drive -- the covariance-weighted and
    lagged-covariance-weighted coupling to the units near each stimulating contact,
    which is the linear-response proxy for how strongly the stimulus reaches it

Leave-one-animal-out, with ridge shrinkage chosen by a *nested* leave-one-animal-out
loop inside the training animals. If this works, unit-level zero-shot transfer
improves legitimately; if it does not, the negative result is definitive, because the
per-unit gain is exactly what an oracle needs and nothing observable predicts it.
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
    fit_propagator,
    fit_shared_from_blocks,
    precompute_blocks,
    predict_delta,
)

MAX_D = 1900.0


def fr_features(s, stim_depths, widths=(100.0, 250.0, 600.0), lags=(0, 1, 2)):
    """(n_obs, n_feat) spontaneous drive from each stim site, averaged over sites."""
    y = s.y[~s.perturbed].astype(np.float64)
    n = s.n_obs
    flat = y.reshape(-1, n)
    mu = flat.mean(0, keepdims=True)
    Xc = flat - mu
    cov0 = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    sd = np.sqrt(np.diag(cov0)) + 1e-9
    lagged = {0: cov0}
    for L in lags:
        if L == 0:
            continue
        a = y[:, :-L, :].reshape(-1, n) - mu
        b = y[:, L:, :].reshape(-1, n) - mu
        lagged[L] = (b.T @ a) / max(len(a) - 1, 1)
    uy = np.asarray(s.meta["unit_y_um"], float)
    cols = []
    for w in widths:
        for L in lags:
            acc = np.zeros(n)
            for d in stim_depths:
                g = np.exp(-(((uy - d) / w) ** 2))
                g = g / (g.sum() + 1e-9)
                acc += lagged[L] @ g
            acc /= max(len(stim_depths), 1)
            # scale-free within session so only the pattern across units transfers
            cols.append(acc / (np.abs(acc).mean() + 1e-9))
            cols.append((acc / sd) / (np.abs(acc / sd).mean() + 1e-9))
    return np.stack(cols, 1)


def unit_design(s):
    uy = np.asarray(s.meta["unit_y_um"], float)
    dep = s.meta["cond_depth_um"]
    stim_depths = sorted({float(dep[c]) if c in dep else float(dep[str(c)])
                          for c in np.unique(s.cond[s.perturbed])})
    fr = fr_features(s, stim_depths)
    uf = s.unit_features
    dz = np.array([np.mean([(uy[k] - d) / MAX_D for d in stim_depths]) for k in range(s.n_obs)])
    adz = np.array([np.mean([abs(uy[k] - d) / MAX_D for d in stim_depths])
                    for k in range(s.n_obs)])
    extra = np.stack([
        uy / MAX_D, dz, adz,
        np.exp(-((adz / 0.10) ** 2)), np.exp(-((adz / 0.30) ** 2)),
        np.ones(s.n_obs),
    ], 1)
    return np.concatenate([uf, fr, extra], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--lams", type=float, nargs="*", default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    ap.add_argument("--out", type=Path, default=Path("results/tables/unit_gain.json"))
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    cfg = LinearResponseConfig()
    props = {s.key: fit_propagator(s, cfg) for s in ds.sets}
    blocks = {s.key: precompute_blocks(s, cfg, props[s.key][0]) for s in ds.sets}
    design = {s.key: unit_design(s) for s in ds.sets}

    # ---- per-unit required gain, with each session's own animal held out ----
    need: dict[str, np.ndarray] = {}
    cache: dict[str, tuple] = {}
    for a in ds.animals:
        theta = fit_shared_from_blocks(blocks, [s for s in ds.sets if s.animal != a], cfg)
        for s in ds.sets:
            if s.animal != a:
                continue
            conds = sorted({int(c) for c in s.cond[s.perturbed]})
            pred = predict_delta(s, cfg, props[s.key][0], theta, conds=conds)
            dl, _ = M.measured_delta(s.y[:, s.t0 :], s.cond, s.perturbed)
            A = np.stack([dl[c] for c in conds])
            B = np.stack([pred[c] for c in conds])
            g = np.ones(s.n_obs)
            for k in range(s.n_obs):
                x = B[:, :, k].ravel()
                den = float(x @ x)
                g[k] = float(x @ A[:, :, k].ravel()) / den if den > 1e-12 else 0.0
            need[s.key] = g
            cache[s.key] = (A, B)

    keys = [s.key for s in ds.sets]
    ani = {s.key: s.animal for s in ds.sets}

    def fit(train_keys, lam):
        X = np.concatenate([design[k] for k in train_keys])
        yv = np.concatenate([need[k] for k in train_keys])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        sd[-1] = 1.0; mu[-1] = 0.0
        Xs = (X - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(Xs.shape[1]), Xs.T @ yv)
        return mu, sd, W, float(yv.mean())

    def score(ks, model, mode):
        mu, sd, W, gbar = model
        out = []
        for k in ks:
            A, B = cache[k]
            if mode == "none":
                g = np.ones(B.shape[2])
            elif mode == "global":
                g = np.full(B.shape[2], gbar)
            elif mode == "predicted":
                g = ((design[k] - mu) / sd) @ W
            else:
                g = need[k]
            out.append(M.delta_r2(A, B * g[None, None, :]))
        return out

    modes = ("none", "global", "predicted", "oracle")
    rows = {m: [] for m in modes}
    groups: list[str] = []
    per_animal = {m: {} for m in modes}
    gp, gt, chosen = [], [], []
    for a in ds.animals:
        tr_ani = [x for x in ds.animals if x != a]
        tr = [k for k in keys if ani[k] != a]
        best_lam, best = args.lams[0], -np.inf
        for lam in args.lams:
            inner = []
            for b in tr_ani:
                itr = [k for k in tr if ani[k] != b]
                ite = [k for k in tr if ani[k] == b]
                if itr and ite:
                    inner += score(ite, fit(itr, lam), "predicted")
            sc = float(np.mean(inner)) if inner else -np.inf
            if sc > best:
                best_lam, best = lam, sc
        chosen.append(best_lam)
        model = fit(tr, best_lam)
        te = [k for k in keys if ani[k] == a]
        groups += [a] * len(te)
        for m in modes:
            v = score(te, model, m)
            rows[m] += v
            per_animal[m][a] = float(np.mean(v))
        mu, sd, W, _ = model
        for k in te:
            gp.append(((design[k] - mu) / sd) @ W)
            gt.append(need[k])

    print(f"{'per-unit gain':16s} {'dR2 [95% CI]':>26s} {'>0':>8s}   per-animal")
    print("-" * 100)
    out = {"lams_chosen_nested": chosen}
    for m in modes:
        mean, lo, hi = M.bootstrap_ci(rows[m])
        rep = M.animal_level_report(rows[m], groups)
        out[m] = {"delta_r2": mean, "ci": [lo, hi], "n": len(rows[m]),
                  "sessions_above_zero": int(sum(x > 0 for x in rows[m])),
                  "per_animal": per_animal[m],
                  "animal_mean": rep["animal_mean"],
                  "animal_ci": [rep["ci_lo"], rep["ci_hi"]],
                  "animals_positive":
                      f"{rep['sign_test']['n_positive']}/{rep['sign_test']['n']}",
                  "animal_p": rep["permutation"]["p"]}
        pa = " ".join(f"{x.replace('sub-ICMS','m')}={v:+.2f}" for x, v in per_animal[m].items())
        print(f"{m:16s} animal {out[m]['animal_mean']:+.3f} "
              f"[{out[m]['animal_ci'][0]:+.3f},{out[m]['animal_ci'][1]:+.3f}] "
              f"pos {out[m]['animals_positive']} p={out[m]['animal_p']:.3f} | "
              f"session {mean:+.3f}   {pa}")
    r = M.corr(np.concatenate(gp), np.concatenate(gt))
    out["unit_gain_corr"] = r
    print(f"\ncorr(predicted, required) per-unit gain = {r:+.3f} "
          f"(LOAO, n={len(np.concatenate(gt))} units)")
    for m in ("predicted", "global"):
        d, p = M.paired_permutation_test(rows[m], rows["none"])
        at = M.animal_permutation_test([per_animal[m][a] for a in per_animal[m]],
                                       [per_animal["none"][a] for a in per_animal["none"]])
        out[f"test_{m}_vs_none"] = {"mean_diff": d, "p_perm": p, "animal": at}
        print(f"{m:10s} vs no rescaling: session diff={d:+.3f} p={p:.2e} | "
              f"ANIMAL diff={at['mean_diff']:+.3f} p={at['p']:.3f}")
    print(f"nested shrinkages: {chosen}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
