"""The low-dimensional shared causal operator: a smooth function of the physical
intervention parameters, applied to a held-out animal without using any of its
intervention data.

Used for the population and behavioural readouts, and shared between the results
table and the figures so that every reported number comes from the same model.
"""

from __future__ import annotations

import numpy as np

MAX_D = 1900.0


def dose_design(amp_ua: float, depth_um: float, coord: float) -> np.ndarray:
    """Feature expansion of (amplitude, contact depth, channel coordinate).

    The expansion is smooth in the physical parameters, which is what allows
    interpolation and extrapolation to intervention settings never trained on.
    """
    an = amp_ua / 10.0
    dn = depth_um / MAX_D
    dz = coord - dn
    g = [np.exp(-((dz / w) ** 2)) for w in (0.10, 0.25, 0.60)]
    return np.array([an, an**2, an**3, np.sqrt(an), np.log1p(an), dn, an * dn, dn**2,
                     *g, *[an * x for x in g], 1.0])


def ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    G = X.T @ X + lam * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ Y)


def cond_params(s, c: int) -> tuple[float, float]:
    amp, dep = s.meta["cond_amp"], s.meta["cond_depth_um"]
    a = float(amp[c]) if c in amp else float(amp[str(c)])
    d = float(dep[c]) if c in dep else float(dep[str(c)])
    return a, d


def rows(readout_fn, coords_fn, s, conds, measured_fn):
    """Design rows for one observation set."""
    coords = coords_fn(s)
    dl = measured_fn(s, conds)
    X, Y, cid, ch = [], [], [], []
    for c in conds:
        a, d = cond_params(s, c)
        D = dl[c]
        for k in range(D.shape[1]):
            if not np.all(np.isfinite(D[:, k])):
                continue
            X.append(dose_design(a, d, coords[min(k, len(coords) - 1)]))
            Y.append(D[:, k])
            cid.append(c)
            ch.append(k)
    if not X:
        return None
    return np.array(X, float), np.array(Y, float), np.array(cid), np.array(ch)


def fit(sets, conds_of, row_fn, lam: float = 1e-2):
    got = [row_fn(s, conds_of(s)) for s in sets]
    got = [r for r in got if r is not None]
    if not got:
        return None
    X = np.concatenate([r[0] for r in got])
    Y = np.concatenate([r[1] for r in got])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    sd[-1] = 1.0
    mu[-1] = 0.0
    return mu, sd, ridge_solve((X - mu) / sd, Y, lam)


def predict(s, conds, model, row_fn, n_channels: int):
    mu, sd, W = model
    r = row_fn(s, conds)
    if r is None:
        return {}
    X, _, cid, ch = r
    P = ((X - mu) / sd) @ W
    out = {int(c): np.zeros((P.shape[1], n_channels)) for c in conds}
    for i in range(len(cid)):
        out[int(cid[i])][:, ch[i]] = P[i]
    return out
