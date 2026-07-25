"""Can an animal's *responsiveness* be predicted from its unperturbed activity?

Cross-animal transfer of the neural response is limited by gain, not shape: the
population time course correlates at r ~ 0.67 with the measurement while Delta-R^2
sits near 0.15. So the question is whether the missing scalar -- how strongly this
particular animal responds -- can be read off its spontaneous activity, which the
protocol *does* allow.

Linear-response theory says it should be: the size of the response to an injected
current is set by the integrated impulse response of the animal's own propagator,
sum_k A_i^k. We therefore regress the gain the shared operator needs against
predictors computed from unperturbed trials only:

    spectral radius of A_i, norm of the integrated impulse response, DC gain
    (I - A_i)^-1, baseline rate, Fano factor, mean pairwise correlation,
    participation ratio, autocorrelation timescale.

Leave-one-animal-out throughout: the gain model is fitted on other animals and
applied to the held-out one. If this works it is a legitimate improvement to the
zero-shot prediction, because nothing but unperturbed activity is used.
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


def spontaneous_features(s, A: np.ndarray, mu: np.ndarray, max_lag: int = 40) -> np.ndarray:
    """Predictors of responsiveness, from unperturbed trials and A_i alone."""
    y = s.y[~s.perturbed].astype(np.float64)
    flat = y.reshape(-1, s.n_obs)
    rate = float(flat.mean())
    var = float(flat.var())
    fano = var / (rate + 1e-9)
    # integrated impulse response: how much a unit impulse accumulates
    n = s.n_obs
    acc = np.zeros((n, n))
    P = np.eye(n)
    for _ in range(max_lag + 1):
        acc += P
        P = A @ P
    dc = np.linalg.inv(np.eye(n) - A) if np.max(np.abs(np.linalg.eigvals(A))) < 0.999 else acc
    ev = float(np.max(np.abs(np.linalg.eigvals(A))))
    # population structure
    c = np.corrcoef(flat.T)
    c = np.nan_to_num(c)
    off = c[~np.eye(n, dtype=bool)]
    mean_corr = float(off.mean())
    w = np.linalg.eigvalsh(np.cov(flat.T) + 1e-12 * np.eye(n))
    w = np.clip(w, 0, None)
    pr = float((w.sum() ** 2) / (np.sum(w**2) + 1e-12)) / max(n, 1)
    # autocorrelation timescale of the population rate
    pop = y.mean(2)
    a = pop[:, :-1].ravel() - pop.mean()
    b = pop[:, 1:].ravel() - pop.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    ac1 = float(a @ b / den) if den > 0 else 0.0
    return np.array([
        np.log(rate + 1e-6),
        np.log(fano + 1e-6),
        ev,
        np.log(np.linalg.norm(acc) / n + 1e-9),
        np.log(np.linalg.norm(dc) / n + 1e-9),
        mean_corr,
        pr,
        ac1,
        np.log(n),
        1.0,
    ], float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--level", default="population", choices=["population", "unit"])
    ap.add_argument("--lams", type=float, nargs="*",
                    default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                    help="candidate shrinkages; chosen by nested LOAO inside the training animals")
    ap.add_argument("--out", type=Path, default=Path("results/tables/gain_from_spontaneous.json"))
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    cfg = LinearResponseConfig()
    props = {s.key: fit_propagator(s, cfg) for s in ds.sets}
    blocks = {s.key: precompute_blocks(s, cfg, props[s.key][0]) for s in ds.sets}
    feats = {s.key: spontaneous_features(s, *props[s.key]) for s in ds.sets}

    def readout(s, D):
        return D.mean(1, keepdims=True) if args.level == "population" else D

    # ---- step 1: for every session, the gain the shared operator would need ----
    # (fitted with that session's own animal held out, so it is an honest target)
    need: dict[str, float] = {}
    cache_pred: dict[str, tuple] = {}
    for a in ds.animals:
        theta = fit_shared_from_blocks(blocks, [s for s in ds.sets if s.animal != a], cfg)
        for s in ds.sets:
            if s.animal != a:
                continue
            conds = sorted({int(c) for c in s.cond[s.perturbed]})
            pred = predict_delta(s, cfg, props[s.key][0], theta, conds=conds)
            dl, _ = M.measured_delta(s.y[:, s.t0 :], s.cond, s.perturbed)
            A = np.stack([readout(s, dl[c]) for c in conds])
            B = np.stack([readout(s, pred[c]) for c in conds])
            den = float((B * B).sum())
            need[s.key] = float((A * B).sum() / den) if den > 1e-12 else 1.0
            cache_pred[s.key] = (A, B)

    # ---- step 2: predict that gain from spontaneous features, LOAO ----
    # The shrinkage is selected by a *nested* leave-one-animal-out loop inside the
    # training animals, so the held-out animal never influences any choice.
    keys = [s.key for s in ds.sets]
    ani = {s.key: s.animal for s in ds.sets}

    def fit_gain(train_keys, lam):
        X = np.stack([feats[k] for k in train_keys])
        yv = np.log(np.clip([need[k] for k in train_keys], 1e-3, None))
        mu, sd = X.mean(0), X.std(0) + 1e-9
        sd[-1] = 1.0; mu[-1] = 0.0
        Xs = (X - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(Xs.shape[1]), Xs.T @ yv)
        return mu, sd, W, float(np.exp(yv.mean()))

    def score_keys(ks, model):
        mu, sd, W, _ = model
        out = []
        for k in ks:
            A, B = cache_pred[k]
            g = float(np.exp(((feats[k] - mu) / sd) @ W))
            out.append(M.delta_r2(A, B * g))
        return out

    rows = {"none": [], "predicted": [], "global": [], "oracle": []}
    per_animal: dict[str, dict[str, float]] = {k: {} for k in rows}
    ghat_all, gtrue_all, chosen = [], [], []
    for a in ds.animals:
        tr_ani = [x for x in ds.animals if x != a]
        tr = [k for k in keys if ani[k] != a]
        # nested selection of the shrinkage
        best_lam, best_score = args.lams[0], -np.inf
        for lam in args.lams:
            inner = []
            for b in tr_ani:
                inner_tr = [k for k in tr if ani[k] != b]
                inner_te = [k for k in tr if ani[k] == b]
                if not inner_tr or not inner_te:
                    continue
                inner += score_keys(inner_te, fit_gain(inner_tr, lam))
            sc = float(np.mean(inner)) if inner else -np.inf
            if sc > best_score:
                best_lam, best_score = lam, sc
        chosen.append(best_lam)
        model = fit_gain(tr, best_lam)
        mu, sd, W, g_global = model
        acc = {k: [] for k in rows}
        for k in keys:
            if ani[k] != a:
                continue
            A, B = cache_pred[k]
            ghat = float(np.exp(((feats[k] - mu) / sd) @ W))
            ghat_all.append(ghat); gtrue_all.append(need[k])
            acc["none"].append(M.delta_r2(A, B))
            acc["predicted"].append(M.delta_r2(A, B * ghat))
            acc["global"].append(M.delta_r2(A, B * g_global))
            acc["oracle"].append(M.delta_r2(A, B * need[k]))
        for kk in rows:
            rows[kk] += acc[kk]
            per_animal[kk][a] = float(np.mean(acc[kk]))

    print(f"readout = {args.level}\n")
    print(f"{'gain source':22s} {'dR2 [95% CI]':>26s} {'>0':>8s}   per-animal")
    print("-" * 100)
    out = {"level": args.level, "lams_chosen_nested": chosen}
    for k in ("none", "global", "predicted", "oracle"):
        m, lo, hi = M.bootstrap_ci(rows[k])
        out[k] = {"delta_r2": m, "ci": [lo, hi], "n": len(rows[k]),
                  "sessions_above_zero": int(sum(x > 0 for x in rows[k])),
                  "per_animal": per_animal[k]}
        pa = " ".join(f"{x.replace('sub-ICMS','m')}={v:+.2f}" for x, v in per_animal[k].items())
        print(f"{k:22s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'':4s} "
              f"{out[k]['sessions_above_zero']:3d}/{len(rows[k])}   {pa}")
    r = M.corr(np.log(np.clip(ghat_all, 1e-3, None)), np.log(np.clip(gtrue_all, 1e-3, None)))
    out["gain_pred_corr_log"] = r
    out["gains"] = {"predicted": list(map(float, ghat_all)),
                    "required": list(map(float, gtrue_all))}
    print(f"\ncorr(log predicted gain, log required gain) = {r:+.3f}  (LOAO, n={len(ghat_all)})")
    print(f"shrinkage chosen by nested LOAO per fold: {chosen}")
    for k in ("predicted", "global"):
        d, p = M.paired_permutation_test(rows[k], rows["none"])
        out[f"test_{k}_vs_none"] = {"mean_diff": d, "p_perm": p}
        print(f"{k:10s} vs no rescaling: diff={d:+.3f}  p_perm={p:.2e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
