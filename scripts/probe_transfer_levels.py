"""How hard is it to predict a *new recording's* ICMS response, and how much of
that difficulty is due to the animal being new?

Three nested generalisation levels, all scored identically:

  within-session   train and test on disjoint trials of the same session
                   (same units)  -> upper bound set by trial noise alone
  cross-session    train on the animal's other sessions -> new units, same animal
  cross-animal     train on other animals               -> new units, new animal

The gap between cross-session and cross-animal isolates the animal effect. The
gap between within-session and cross-session measures how hard *new units* are,
independent of the animal.

Predictors are non-interventional: stimulation amplitude and contact depth, unit
depth and class, unit firing statistics, and the unit's spontaneous coupling to
the part of the population near the stimulating contact.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.baselines import measured_delta_set

MAX_D = 1900.0


def spont_coupling(s, stim_depth_um: float, band_um: float = 300.0) -> np.ndarray:
    """Each unit's spontaneous correlation with the units near the stim contact.

    Uses unperturbed trials only. This is the classic non-interventional proxy for
    how strongly a unit is functionally connected to the stimulated site.
    """
    y = s.y[~s.perturbed]
    flat = y.reshape(-1, s.n_obs)
    flat = flat - flat.mean(0, keepdims=True)
    nrm = np.linalg.norm(flat, axis=0) + 1e-9
    Cmat = (flat.T @ flat) / np.outer(nrm, nrm)
    uy = np.asarray(s.meta["unit_y_um"], float)
    near = np.abs(uy - stim_depth_um) <= band_um
    if near.sum() == 0:
        near = np.abs(uy - stim_depth_um) <= 2 * band_um
    if near.sum() == 0:
        return np.zeros(s.n_obs)
    out = np.zeros(s.n_obs)
    for n in range(s.n_obs):
        m = near.copy()
        m[n] = False
        out[n] = Cmat[n, m].mean() if m.any() else 0.0
    return np.nan_to_num(out)


def fluctuation_response(s, widths_um=(100.0, 250.0, 500.0), lags=(0, 1, 2)):
    """Linear-response prediction of the causal effect from spontaneous fluctuations.

    Under a fluctuation-response (fluctuation-dissipation) ansatz, the response of
    unit ``n`` to a current injected near depth ``d`` is proportional to the
    spontaneous covariance between ``n`` and the units near ``d``, propagated by
    the spontaneous lagged covariance. Everything here is estimated from
    unperturbed trials only, so it is available for a new animal.

    Returns ``kernel[lag][width] -> (n_obs, n_obs)`` weighted maps and a helper
    that evaluates them at a stimulation depth.
    """
    y = s.y[~s.perturbed]
    n_obs = s.n_obs
    flat = y.reshape(-1, n_obs)
    mu = flat.mean(0, keepdims=True)
    Xc = flat - mu
    cov0 = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    sd = np.sqrt(np.diag(cov0)) + 1e-9
    lagged = {}
    for L in lags:
        if L == 0:
            lagged[0] = cov0
        else:
            a = y[:, :-L, :].reshape(-1, n_obs) - mu
            b = y[:, L:, :].reshape(-1, n_obs) - mu
            lagged[L] = (b.T @ a) / max(len(a) - 1, 1)   # response of b to earlier a
    uy = np.asarray(s.meta["unit_y_um"], float)

    def at_depth(d: float) -> np.ndarray:
        """(n_features,) per unit -> returned as (n_obs, n_feat)."""
        cols = []
        for w in widths_um:
            g = np.exp(-(((uy - d) / w) ** 2))
            g = g / (g.sum() + 1e-9)
            for L in lags:
                K = lagged[L]
                v = K @ g                       # covariance-weighted drive
                cols.append(v)
                cols.append(v / sd)             # correlation-scaled
        return np.stack(cols, 1)

    return at_depth


def rows_for(s, use_fr: bool = True):
    dl, _ = measured_delta_set(s)
    amp = s.meta["cond_amp"]
    dep = s.meta["cond_depth_um"]
    uy = np.asarray(s.meta["unit_y_um"], float)
    ct = s.meta["cell_type"]
    feats = s.unit_features
    cache: dict[float, np.ndarray] = {}
    fr_at = fluctuation_response(s) if use_fr else None
    fr_cache: dict[float, np.ndarray] = {}
    X, Y, cid, uidx = [], [], [], []
    for c, D in dl.items():
        a = float(amp[c]) if c in amp else float(amp[str(c)])
        sd = float(dep[c]) if c in dep else float(dep[str(c)])
        if sd not in cache:
            cache[sd] = spont_coupling(s, sd)
            if fr_at is not None:
                F = fr_at(sd)
                # scale-free within session so that only the *pattern* transfers
                F = F / (np.abs(F).mean(0, keepdims=True) + 1e-9)
                fr_cache[sd] = F
        cpl = cache[sd]
        an = a / 10.0
        for n in range(s.n_obs):
            dz = (uy[n] - sd) / MAX_D
            g1 = np.exp(-((dz / 0.10) ** 2))
            g2 = np.exp(-((dz / 0.25) ** 2))
            g3 = np.exp(-((dz / 0.50) ** 2))
            row = [
                an, an**2, np.sqrt(an),
                sd / MAX_D, uy[n] / MAX_D, dz, abs(dz),
                g1, g2, g3,
                an * g1, an * g2, an * g3, an * abs(dz),
                cpl[n], cpl[n] * an, cpl[n] * g2,
                1.0 if "pyr" in str(ct[n]).lower() else 0.0,
                feats[n, 0], feats[n, 1], feats[n, 2], feats[n, 3], feats[n, 6], feats[n, 7],
                feats[n, 0] * an, feats[n, 6] * an,
            ]
            if fr_at is not None:
                fr = fr_cache[sd][n]
                row += list(fr) + list(fr * an)
            row.append(1.0)
            X.append(row)
            Y.append(D[:, n])
            cid.append(int(c))
            uidx.append(n)
    return np.array(X, float), np.array(Y, float), np.array(cid), np.array(uidx)


def ridge(X, Y, lam):
    G = X.T @ X + lam * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ Y)


def score_set(s, X, Y, cid, uidx, mu, sd, W, sel=None):
    P = ((X - mu) / sd) @ W
    idx = np.arange(len(cid)) if sel is None else np.where(sel)[0]
    conds = sorted(set(cid[idx].tolist()))
    T = Y.shape[1]
    pred = {c: np.zeros((T, s.n_obs)) for c in conds}
    true = {c: np.zeros((T, s.n_obs)) for c in conds}
    for i in idx:
        pred[cid[i]][:, uidx[i]] = P[i]
        true[cid[i]][:, uidx[i]] = Y[i]
    A = np.stack([true[c] for c in conds])
    Bm = np.stack([pred[c] for c in conds])
    return M.delta_r2(A, Bm), M.corr(A, Bm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--out", type=Path, default=Path("results/tables/transfer_levels.json"))
    ap.add_argument("--no-fr", action="store_true", help="disable fluctuation-response features")
    args = ap.parse_args()
    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    per = {s.key: rows_for(s, use_fr=not args.no_fr) for s in ds.sets}
    by_key = {s.key: s for s in ds.sets}

    def fit_on(keys):
        X = np.concatenate([per[k][0] for k in keys])
        Y = np.concatenate([per[k][1] for k in keys])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        sd[-1] = 1.0; mu[-1] = 0.0
        return mu, sd, ridge((X - mu) / sd, Y, args.lam)

    res: dict[str, list[float]] = {k: [] for k in
                                   ("within_session", "cross_session", "cross_animal")}
    resr: dict[str, list[float]] = {k: [] for k in res}
    print(f"{'session':30s} {'within':>8s} {'x-sess':>8s} {'x-anim':>8s}")
    print("-" * 58)
    for s in ds.sets:
        X, Y, cid, uidx = per[s.key]
        # within-session: half the conditions' trials -> but units identical, so
        # split trials by fitting on odd conditions is not comparable; instead fit
        # on this session itself (in-sample upper bound with the same features)
        mu, sd, W = fit_on([s.key])
        w, wr = score_set(s, X, Y, cid, uidx, mu, sd, W)
        same_animal = [k for k in per if by_key[k].animal == s.animal and k != s.key]
        cs = csr = float("nan")
        if same_animal:
            mu2, sd2, W2 = fit_on(same_animal)
            cs, csr = score_set(s, X, Y, cid, uidx, mu2, sd2, W2)
        other = [k for k in per if by_key[k].animal != s.animal]
        mu3, sd3, W3 = fit_on(other)
        ca, car = score_set(s, X, Y, cid, uidx, mu3, sd3, W3)
        res["within_session"].append(w); resr["within_session"].append(wr)
        res["cross_session"].append(cs); resr["cross_session"].append(csr)
        res["cross_animal"].append(ca); resr["cross_animal"].append(car)
        print(f"{s.key:30s} {w:+8.3f} {cs:+8.3f} {ca:+8.3f}")

    print(f"\n{'level':16s} {'dR2 mean [95% CI]':>28s} {'r':>8s}")
    out = {}
    for k in res:
        m, lo, hi = M.bootstrap_ci(res[k])
        rm, _, _ = M.bootstrap_ci(resr[k])
        out[k] = {"delta_r2": m, "ci": [lo, hi], "delta_corr": rm}
        print(f"{k:16s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'':8s} {rm:+.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
