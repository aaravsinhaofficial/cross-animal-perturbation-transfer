"""Baselines for cross-animal prediction of intervention responses.

The claim under test is that a *shared, state-dependent causal operator*
transfers. The baselines are therefore built to be as strong as possible while
lacking exactly that ingredient:

``no_effect``
    Assert the intervention does nothing. Defines Delta-R^2 = 0.

``ma_cca``
    Model-free manifold alignment. Each animal's unperturbed activity is reduced
    by PCA and animals are brought into a common frame by orthogonal Procrustes
    on their unperturbed latent dynamics (both the trajectory geometry and the
    fitted linear-dynamics eigenbasis). The measured average intervention effect
    of the training animals is transported into the test animal's neuron space
    through that alignment. This is the "another manifold-alignment method"
    control: it aligns representations and transfers an *observed* effect, but
    has no causal operator.

``ma_latent``
    The same idea, but using CADENCE's own calibrated observation maps for the
    alignment. This isolates the contribution of the state-dependent shared
    operator from the contribution of having a good aligned latent space, so it
    is the sharpest ablation of the central claim.

``unit_feature_ridge``
    A per-unit encoding model. Each unit's effect time course is predicted by
    ridge regression from features of that unit's *unperturbed* activity
    (baseline rate, loadings on the leading unperturbed PCs, coupling to the
    population, autocorrelation timescale) together with the intervention
    descriptor. Fitted across all units of the training animals and applied to
    the test animal's units. Uses no dynamical model at all.

All baselines are given the same intervention descriptors as CADENCE, including
the ability to interpolate/extrapolate across intervention settings, so that the
comparison on unseen interventions is fair.
"""

from __future__ import annotations

import numpy as np

from .data.containers import AnimalTrials


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def post(s: AnimalTrials, arr: np.ndarray) -> np.ndarray:
    return arr[:, s.t0 :]


def condition_descriptors(s: AnimalTrials) -> dict[int, np.ndarray]:
    """Mean physical intervention descriptor for each perturbation condition."""
    out = {}
    raw = post(s, s.interv_raw)
    on = post(s, s.interv_on)
    for c in np.unique(s.cond[s.perturbed]):
        m = s.cond == c
        w = on[m] > 0
        if w.sum() == 0:
            out[int(c)] = np.zeros(raw.shape[-1])
        else:
            out[int(c)] = raw[m][w].mean(0)
    return out


def measured_delta_set(s: AnimalTrials) -> tuple[dict[int, np.ndarray], np.ndarray]:
    y = post(s, s.y)
    base = y[~s.perturbed].mean(0)
    out = {int(c): y[s.cond == c].mean(0) - base for c in np.unique(s.cond[s.perturbed])}
    return out, base


def unperturbed_pcs(s: AnimalTrials, d: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA basis of unperturbed activity. Returns (P: d x n_obs, mean: n_obs)."""
    y = s.y[~s.perturbed]
    X = y.reshape(-1, y.shape[-1])
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD on the covariance for speed when n_obs << n_samples
    cov = Xc.T @ Xc / max(len(Xc) - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:d]
    P = v[:, order].T
    return P, mu


def fit_lds(latents: np.ndarray) -> np.ndarray:
    """Least-squares one-step linear dynamics on latent trajectories.

    latents: (n_trials, T, d) -> A: (d, d) with z_{t+1} ~ A z_t.
    """
    Z0 = latents[:, :-1].reshape(-1, latents.shape[-1])
    Z1 = latents[:, 1:].reshape(-1, latents.shape[-1])
    A, *_ = np.linalg.lstsq(Z0, Z1, rcond=None)
    return A.T


def procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||X R - Y||_F (rows are observations)."""
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def _dose_basis(desc: np.ndarray) -> np.ndarray:
    """Feature expansion of the intervention descriptor allowing smooth
    interpolation and extrapolation across intervention settings."""
    d = np.asarray(desc, float).ravel()
    return np.concatenate([[1.0], d, d**2, np.sqrt(np.abs(d))])


def _fit_dose_map(descs: list[np.ndarray], targets: list[np.ndarray], ridge: float = 1e-3):
    """Regress condition-level effect tensors on the intervention descriptor."""
    Phi = np.stack([_dose_basis(d) for d in descs])
    Y = np.stack([t.ravel() for t in targets])
    G = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    W = np.linalg.solve(G, Phi.T @ Y)
    shape = targets[0].shape
    return lambda d: (_dose_basis(d) @ W).reshape(shape)


# ---------------------------------------------------------------------------
# baseline 1: no effect
# ---------------------------------------------------------------------------
def no_effect(test: AnimalTrials) -> dict[int, np.ndarray]:
    T = test.T - test.t0
    return {int(c): np.zeros((T, test.n_obs)) for c in np.unique(test.cond[test.perturbed])}


# ---------------------------------------------------------------------------
# baseline 2: model-free manifold alignment
# ---------------------------------------------------------------------------
def ma_cca(
    train: list[AnimalTrials],
    test: AnimalTrials,
    d: int = 12,
    ridge: float = 1e-3,
    align_on: str = "both",
) -> dict[int, np.ndarray]:
    """Align unperturbed manifolds, then transport the group-average effect."""
    ref = None
    ref_feat = None
    latent_deltas: dict[int, list[np.ndarray]] = {}
    descs: dict[int, list[np.ndarray]] = {}

    def alignment_features(s: AnimalTrials, P: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Animal-level descriptors of the unperturbed dynamics in its own PCA
        frame: the mean trajectory over the trial and the eigenbasis of the
        fitted linear dynamics. Concatenated, these give a d-dimensional frame
        that can be Procrustes-matched across animals."""
        y = s.y[~s.perturbed]
        Z = (y - mu) @ P.T                     # (n_trials, T, d)
        traj = Z.mean(0)                       # (T, d) shared task time base
        A = fit_lds(Z)
        w, v = np.linalg.eig(A)
        order = np.argsort(-np.abs(w))
        V = np.real(v[:, order]).T             # (d, d)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        cov = np.cov(Z.reshape(-1, d).T)
        wc, vc = np.linalg.eigh(cov)
        vc = vc[:, np.argsort(-wc)].T
        if align_on == "traj":
            return traj
        if align_on == "dyn":
            return np.concatenate([V, vc])
        return np.concatenate([traj, V, vc])

    for s in train:
        P, mu = unperturbed_pcs(s, d)
        feat = alignment_features(s, P, mu)
        if ref is None:
            ref, ref_feat = s.key, feat
            R = np.eye(d)
        else:
            R = procrustes(feat, ref_feat)     # feat @ R ~ ref_feat
        dl, _ = measured_delta_set(s)
        dd = condition_descriptors(s)
        Ppinv = np.linalg.pinv(P)              # (n_obs, d)
        for c, D in dl.items():
            z = (D @ Ppinv) @ R                # (T, d) in the reference frame
            latent_deltas.setdefault(c, []).append(z)
            descs.setdefault(c, []).append(dd[c])

    # group average in the reference frame, as a smooth function of the
    # intervention descriptor
    all_desc, all_target = [], []
    for c, zs in latent_deltas.items():
        L = min(z.shape[0] for z in zs)
        z = np.mean([x[:L] for x in zs], axis=0)
        all_desc.append(np.mean(descs[c], axis=0))
        all_target.append(z)
    Lmin = min(t.shape[0] for t in all_target)
    all_target = [t[:Lmin] for t in all_target]
    dose = _fit_dose_map(all_desc, all_target, ridge)

    # map into the test animal's neuron space
    P_t, mu_t = unperturbed_pcs(test, d)
    feat_t = alignment_features(test, P_t, mu_t)
    R_t = procrustes(feat_t, ref_feat)
    dd_t = condition_descriptors(test)
    T_post = test.T - test.t0
    out = {}
    for c, desc in dd_t.items():
        z_ref = dose(desc)                      # (Lmin, d) reference frame
        z_test = z_ref @ R_t.T                  # back to the test animal's frame
        D = z_test @ P_t                        # (Lmin, n_obs)
        if D.shape[0] < T_post:
            D = np.vstack([D, np.repeat(D[-1:], T_post - D.shape[0], axis=0)])
        out[int(c)] = D[:T_post]
    return out


# ---------------------------------------------------------------------------
# baseline 3: group-average latent effect through CADENCE's own alignment
# ---------------------------------------------------------------------------
def ma_latent(
    train: list[AnimalTrials],
    test: AnimalTrials,
    obs_maps: dict[str, np.ndarray],
    obs_biases: dict[str, np.ndarray] | None = None,
    ridge: float = 1e-3,
) -> dict[int, np.ndarray]:
    """Transport the group-average *latent* effect through the model's own
    calibrated observation maps.

    ``obs_maps[key]`` is the (n_obs, d) matrix C for that observation set. The
    effect is pulled back to the latent space with the pseudo-inverse of C,
    averaged across training animals, and pushed into the test animal's neurons.
    Everything about the alignment is shared with CADENCE; only the
    state-dependent causal operator is removed.
    """
    latent, descs = {}, {}
    for s in train:
        C = obs_maps[s.key]
        Cp = np.linalg.pinv(C)                  # (d, n_obs)
        dl, _ = measured_delta_set(s)
        dd = condition_descriptors(s)
        for c, D in dl.items():
            latent.setdefault(c, []).append(D @ Cp.T)
            descs.setdefault(c, []).append(dd[c])
    all_desc, all_target = [], []
    for c, zs in latent.items():
        L = min(z.shape[0] for z in zs)
        all_target.append(np.mean([z[:L] for z in zs], axis=0))
        all_desc.append(np.mean(descs[c], axis=0))
    Lmin = min(t.shape[0] for t in all_target)
    all_target = [t[:Lmin] for t in all_target]
    dose = _fit_dose_map(all_desc, all_target, ridge)

    C_t = obs_maps[test.key]
    T_post = test.T - test.t0
    out = {}
    for c, desc in condition_descriptors(test).items():
        z = dose(desc)
        D = z @ C_t.T
        if D.shape[0] < T_post:
            D = np.vstack([D, np.repeat(D[-1:], T_post - D.shape[0], axis=0)])
        out[int(c)] = D[:T_post]
    return out


# ---------------------------------------------------------------------------
# baseline 4: per-unit encoding model from unperturbed features
# ---------------------------------------------------------------------------
def unit_features(s: AnimalTrials, d: int = 8) -> np.ndarray:
    """Features of each unit computed from that animal's UNPERTURBED data only."""
    y = s.y[~s.perturbed]                       # (n, T, N)
    N = y.shape[-1]
    flat = y.reshape(-1, N)
    mu = flat.mean(0)
    sd = flat.std(0) + 1e-6
    P, m0 = unperturbed_pcs(s, d)
    load = P                                    # (d, N)
    pop = flat.mean(1)
    coup = np.array([np.corrcoef(flat[:, n], pop)[0, 1] for n in range(N)])
    coup = np.nan_to_num(coup)
    # lag-1 autocorrelation within trials
    a = y[:, :-1, :].reshape(-1, N)
    b = y[:, 1:, :].reshape(-1, N)
    ac = []
    for n in range(N):
        x, yy = a[:, n] - a[:, n].mean(), b[:, n] - b[:, n].mean()
        den = np.linalg.norm(x) * np.linalg.norm(yy)
        ac.append(float(x @ yy / den) if den > 0 else 0.0)
    ac = np.array(ac)
    # temporal modulation across the trial in unperturbed condition
    tm = y.mean(0)
    tm = (tm - tm.mean(0)) / (tm.std(0) + 1e-6)
    tprof = tm[:: max(1, tm.shape[0] // 6)][:6]  # 6 coarse time samples
    feats = np.concatenate(
        [
            np.log1p(mu)[None],
            np.log1p(sd)[None],
            coup[None],
            ac[None],
            load,
            tprof,
        ],
        axis=0,
    ).T                                          # (N, n_feat)
    return np.nan_to_num(feats)


def unit_feature_ridge(
    train: list[AnimalTrials],
    test: AnimalTrials,
    d: int = 8,
    ridge: float = 1.0,
) -> dict[int, np.ndarray]:
    """Predict each test unit's effect time course from its unperturbed features
    and the intervention descriptor."""
    X_rows, Y_rows = [], []
    T_ref = None
    for s in train:
        F = unit_features(s, d)
        dl, _ = measured_delta_set(s)
        dd = condition_descriptors(s)
        for c, D in dl.items():
            if T_ref is None:
                T_ref = D.shape[0]
            L = min(T_ref, D.shape[0])
            db = _dose_basis(dd[c])
            for n in range(F.shape[0]):
                X_rows.append(np.concatenate([F[n], db, np.outer(F[n], db).ravel()]))
                row = np.zeros(T_ref)
                row[:L] = D[:L, n]
                Y_rows.append(row)
    if not X_rows:
        return no_effect(test)
    X = np.stack(X_rows)
    Y = np.stack(Y_rows)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    G = Xs.T @ Xs + ridge * len(Xs) * np.eye(Xs.shape[1])
    W = np.linalg.solve(G, Xs.T @ Y)
    b = Y.mean(0)

    F_t = unit_features(test, d)
    T_post = test.T - test.t0
    out = {}
    for c, desc in condition_descriptors(test).items():
        db = _dose_basis(desc)
        rows = np.stack(
            [np.concatenate([F_t[n], db, np.outer(F_t[n], db).ravel()]) for n in range(F_t.shape[0])]
        )
        pred = ((rows - mu) / sd) @ W           # (N, T_ref)
        D = pred.T
        if D.shape[0] < T_post:
            D = np.vstack([D, np.repeat(D[-1:], T_post - D.shape[0], axis=0)])
        out[int(c)] = D[:T_post]
    _ = b
    return out
