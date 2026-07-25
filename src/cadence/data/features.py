"""Per-unit features computed from **unperturbed activity and metadata only**.

Why this exists
---------------
Fitting a completely free observation map ``C_i`` for a new animal on spontaneous
activity does not pin down the latent coordinate frame: spontaneous activity sits
near a fixed point, where the dynamics are weak and close to isotropic, so many
different ``C_i`` explain it equally well. That is precisely the degenerate case
in which transfer of a shared causal operator is not identifiable.

The remedy is to make ``C_i`` a *shared* function of quantities that are
comparable across animals and observable without ever intervening: where the unit
sits on the probe, what kind of cell it is, how fast it fires, how correlated it
is with its population, and how long its autocorrelation is. The observation map
stays animal-specific -- different animals have different units, hence different
loadings -- but it is now expressed in a common frame, which is what makes the
causal operator transferable.

Every feature here is a function of unperturbed trials and static metadata.
Nothing derived from an intervention trial may enter.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "log_rate",
    "log_std",
    "fano",
    "autocorr_lag1",
    "autocorr_lag2",
    "autocorr_lag3",
    "population_corr",
    "rate_rank",
    "depth_norm",
    "is_pyramidal",
    "is_interneuron",
)
N_FEATURES = len(FEATURE_NAMES)


def _autocorr(y: np.ndarray, lag: int) -> np.ndarray:
    """Within-trial autocorrelation at ``lag`` bins, per unit. y: (n, T, N)."""
    if y.shape[1] <= lag:
        return np.zeros(y.shape[2])
    a = y[:, :-lag, :].reshape(-1, y.shape[2])
    b = y[:, lag:, :].reshape(-1, y.shape[2])
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    num = (a * b).sum(0)
    den = np.sqrt((a**2).sum(0) * (b**2).sum(0)) + 1e-9
    return np.nan_to_num(num / den)


def unit_features(
    y_unperturbed: np.ndarray,
    depth_um: np.ndarray | None = None,
    cell_type: list[str] | None = None,
    max_depth_um: float = 1900.0,
) -> np.ndarray:
    """(n_obs, N_FEATURES) features from unperturbed trials only.

    ``y_unperturbed``: (n_trials, T, n_obs) counts on unperturbed trials.
    """
    y = np.asarray(y_unperturbed, dtype=np.float64)
    n_obs = y.shape[-1]
    flat = y.reshape(-1, n_obs)
    mu = flat.mean(0)
    sd = flat.std(0)
    fano = sd**2 / (mu + 1e-9)
    pop = flat.mean(1)
    pc = np.zeros(n_obs)
    pcen = pop - pop.mean()
    denom_pop = np.linalg.norm(pcen) + 1e-9
    for n in range(n_obs):
        x = flat[:, n] - flat[:, n].mean()
        d = np.linalg.norm(x) * denom_pop
        pc[n] = float(x @ pcen / d) if d > 0 else 0.0
    order = np.argsort(np.argsort(mu))
    rank = order / max(n_obs - 1, 1)

    feats = np.zeros((n_obs, N_FEATURES), dtype=np.float32)
    feats[:, 0] = np.log1p(mu)
    feats[:, 1] = np.log1p(sd)
    feats[:, 2] = np.clip(fano, 0, 10)
    feats[:, 3] = _autocorr(y, 1)
    feats[:, 4] = _autocorr(y, 2)
    feats[:, 5] = _autocorr(y, 3)
    feats[:, 6] = pc
    feats[:, 7] = rank
    if depth_um is not None:
        feats[:, 8] = np.asarray(depth_um, float) / max_depth_um
    if cell_type is not None:
        ct = [str(c).lower() for c in cell_type]
        feats[:, 9] = [1.0 if "pyr" in c else 0.0 for c in ct]
        feats[:, 10] = [1.0 if ("inter" in c or "fs" in c) else 0.0 for c in ct]
    return np.nan_to_num(feats)
