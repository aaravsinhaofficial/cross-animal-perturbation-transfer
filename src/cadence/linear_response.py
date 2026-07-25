"""Shared causal operator in the observed space: a linear-response model.

Motivation
----------
Fitting a latent observation map for a *new* animal on spontaneous activity does
not reliably pin the latent coordinate frame, because spontaneous activity sits
near a fixed point where the dynamics are weak and close to isotropic. Any error
in that frame corrupts the transferred causal operator.

This model removes the problem by never asking for a latent frame. Each animal's
propagator is estimated **directly in its own recorded space** from unperturbed
activity, and the only thing shared across animals is the *drive*: how a physical
stimulus (amplitude, contact depth) injects current as a function of a unit's
distance from the contact and of time.

    y_{t+1} - mu_i = A_i (y_t - mu_i) + u_t^{(i)} + noise          (unperturbed: u = 0)
    u_t^{(i)}(n)   = sum_{j,l,b} theta_{jlb} psi_l(amp) B_b(t) exp(-((depth_n - d)/sigma_j)^2)

so the predicted causal effect is the animal's own impulse response convolved
with a species-invariant drive,

    Delta_i(t, n) = sum_{k=0}^{t} [A_i^k u_{t-k}^{(i)}]_n .

``A_i`` and ``mu_i`` come from unperturbed trials only; ``theta`` is fitted across
*other* animals. Delta is linear in ``theta``, so the shared operator is obtained
in closed form -- there is no optimisation to get stuck and nothing to tune per
animal.

This is a discrete-time fluctuation-response (linear-response) statement: the
spontaneous propagator predicts the response to a perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.containers import AnimalTrials

MAX_D = 1900.0


@dataclass
class LinearResponseConfig:
    sigmas_um: tuple[float, ...] = (80.0, 160.0, 320.0, 640.0, 1280.0)
    n_time_basis: int = 14
    amp_basis: tuple[str, ...] = ("const", "lin", "quad", "sqrt")
    var_ridge: float = 1e-2          # ridge for the per-animal propagator
    theta_ridge: float = 1e-3        # ridge for the shared drive
    max_lag: int = 40                # truncate the impulse response
    spectral_clip: float = 0.995     # keep A_i stable
    include_intercept: bool = True
    normalise_target: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
def raised_cosine_basis(T: int, n: int) -> np.ndarray:
    """(n, T) smooth non-negative temporal basis covering [0, T)."""
    centers = np.linspace(0, T - 1, n)
    width = max((centers[1] - centers[0]) if n > 1 else T, 1.0) * 2.0
    t = np.arange(T)[None, :]
    d = (t - centers[:, None]) / width
    B = np.where(np.abs(d) <= 1, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
    B[0, : int(centers[0]) + 1] = np.maximum(B[0, : int(centers[0]) + 1], 1e-6)
    return B


def amp_features(amp: float, kinds: tuple[str, ...]) -> np.ndarray:
    a = amp / 10.0
    m = {"const": 1.0, "lin": a, "quad": a * a, "sqrt": np.sqrt(max(a, 0.0)),
         "cube": a ** 3, "log": np.log1p(a)}
    return np.array([m[k] for k in kinds], float)


def fit_propagator(s: AnimalTrials, cfg: LinearResponseConfig) -> tuple[np.ndarray, np.ndarray]:
    """One-step linear propagator from UNPERTURBED trials only."""
    y = s.y[~s.perturbed].astype(np.float64)
    n = s.n_obs
    flat = y.reshape(-1, n)
    mu = flat.mean(0)
    Y0 = (y[:, :-1, :] - mu).reshape(-1, n)
    Y1 = (y[:, 1:, :] - mu).reshape(-1, n)
    G = Y0.T @ Y0 + cfg.var_ridge * len(Y0) * np.eye(n)
    A = np.linalg.solve(G, Y0.T @ Y1).T
    # keep the impulse response bounded
    ev = np.max(np.abs(np.linalg.eigvals(A)))
    if ev > cfg.spectral_clip:
        A = A * (cfg.spectral_clip / ev)
    return A, mu


def design_for_set(
    s: AnimalTrials, cfg: LinearResponseConfig, A: np.ndarray, conds: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Design tensor X (rows = (cond, t, unit)) and target Delta for one set.

    Column order is (sigma, amp_basis, time_basis), plus an optional intercept.
    """
    T = s.T - s.t0
    B = raised_cosine_basis(T, cfg.n_time_basis)
    uy = np.asarray(s.meta["unit_y_um"], float)
    amp_map = s.meta["cond_amp"]
    dep_map = s.meta["cond_depth_um"]
    conds = conds or sorted({int(c) for c in s.cond[s.perturbed]})

    ypost = s.y[:, s.t0 :].astype(np.float64)
    base = ypost[~s.perturbed].mean(0)

    n_s, n_a, n_b = len(cfg.sigmas_um), len(cfg.amp_basis), cfg.n_time_basis
    n_col = n_s * n_a * n_b + (1 if cfg.include_intercept else 0)
    rows_X, rows_Y, index = [], [], []

    # powers of A applied to each spatial profile, once per (cond, sigma)
    for c in conds:
        a = float(amp_map[c]) if c in amp_map else float(amp_map[str(c)])
        d = float(dep_map[c]) if c in dep_map else float(dep_map[str(c)])
        psi = amp_features(a, cfg.amp_basis)
        Q = np.zeros((n_s, n_b, T, s.n_obs))
        for j, sig in enumerate(cfg.sigmas_um):
            v = np.exp(-(((uy - d) / sig) ** 2))
            P = np.zeros((cfg.max_lag + 1, s.n_obs))
            cur = v.copy()
            P[0] = cur
            for k in range(1, cfg.max_lag + 1):
                cur = A @ cur
                P[k] = cur
            # causal convolution of the basis with the impulse response
            for b in range(n_b):
                acc = np.zeros((T, s.n_obs))
                for k in range(0, min(cfg.max_lag, T - 1) + 1):
                    w = B[b, : T - k]
                    if not np.any(w):
                        continue
                    acc[k:] += w[:, None] * P[k][None, :]
                Q[j, b] = acc
        D = ypost[s.cond == c].mean(0) - base
        X = np.zeros((T * s.n_obs, n_col))
        col = 0
        for j in range(n_s):
            for l in range(n_a):
                for b in range(n_b):
                    X[:, col] = (psi[l] * Q[j, b]).reshape(-1)
                    col += 1
        if cfg.include_intercept:
            X[:, col] = 1.0
        rows_X.append(X)
        rows_Y.append(D.reshape(-1))
        index += [(c, t, n) for t in range(T) for n in range(s.n_obs)]
    return np.concatenate(rows_X), np.concatenate(rows_Y), index


def precompute_blocks(
    s: AnimalTrials, cfg: LinearResponseConfig, A: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    """Per-condition normal-equation blocks for one set.

    The design depends only on the set (through ``A`` and the unit depths) and on
    the condition's physical parameters -- never on which other animals are being
    trained on. Caching these blocks makes every leave-one-out refit a sum of
    precomputed matrices.
    """
    out = {}
    conds = sorted({int(c) for c in s.cond[s.perturbed]})
    for c in conds:
        X, y, _ = design_for_set(s, cfg, A, [c])
        out[c] = (X.T @ X, X.T @ y, len(y))
    return out


def fit_shared_from_blocks(
    blocks: dict[str, dict[int, tuple[np.ndarray, np.ndarray, int]]],
    sets: list[AnimalTrials],
    cfg: LinearResponseConfig,
    cond_filter=None,
) -> np.ndarray:
    """Closed-form least squares for the shared drive from cached blocks."""
    XtX = Xty = None
    n_col = None
    for s in sets:
        bl = blocks.get(s.key)
        if not bl:
            continue
        conds = [c for c in bl if (cond_filter is None or cond_filter(s, c))]
        if not conds:
            continue
        n_tot = sum(bl[c][2] for c in conds)
        w = 1.0 / max(n_tot, 1)          # weight each set equally
        for c in conds:
            gxx, gxy, _ = bl[c]
            if XtX is None:
                n_col = gxx.shape[0]
                XtX = np.zeros((n_col, n_col))
                Xty = np.zeros(n_col)
            XtX += w * gxx
            Xty += w * gxy
    if XtX is None:
        raise RuntimeError("no training conditions")
    scale = np.trace(XtX) / n_col
    return np.linalg.solve(XtX + cfg.theta_ridge * scale * np.eye(n_col), Xty)


def fit_shared(
    sets: list[AnimalTrials],
    cfg: LinearResponseConfig,
    props: dict[str, tuple[np.ndarray, np.ndarray]],
    cond_filter=None,
) -> np.ndarray:
    """Convenience wrapper that builds the blocks on the fly."""
    blocks = {s.key: precompute_blocks(s, cfg, props[s.key][0]) for s in sets}
    return fit_shared_from_blocks(blocks, sets, cfg, cond_filter)


def predict_delta(
    s: AnimalTrials, cfg: LinearResponseConfig, A: np.ndarray, theta: np.ndarray,
    conds: list[int] | None = None
) -> dict[int, np.ndarray]:
    T = s.T - s.t0
    conds = conds or sorted({int(c) for c in s.cond[s.perturbed]})
    X, _, index = design_for_set(s, cfg, A, conds)
    p = X @ theta
    out = {c: np.zeros((T, s.n_obs)) for c in conds}
    for i, (c, t, n) in enumerate(index):
        out[c][t, n] = p[i]
    return out
