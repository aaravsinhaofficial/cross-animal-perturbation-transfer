"""Ground-truth recovery analyses for the teacher-RNN benchmark.

Two questions:

1. **Is the latent state recovered?** The model's latent ``z`` is compared with
   the teacher's true state by canonical correlation analysis, which is invariant
   to the linear reparameterisation that any latent variable model leaves free.

2. **Is the causal operator recovered?** The intervention was delivered along a
   known direction in the teacher's state space. The model's latent intervention
   direction is mapped into the teacher's state space through the linear map
   fitted between ``z`` and the true state on *unperturbed* trials, and compared
   with the true injection direction by cosine similarity.

Both are computed for a **held-out** animal, whose private parameters were fitted
on unperturbed data only.
"""

from __future__ import annotations

import numpy as np


def cca_correlations(A: np.ndarray, B: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Canonical correlations between two sets of observations (rows = samples)."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False) if False else np.linalg.svd(
        qa.T @ qb, compute_uv=False
    )
    return np.clip(s, 0.0, 1.0)


def linear_map(Z: np.ndarray, X: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """Least-squares map W with X ~ [Z, 1] W."""
    Z1 = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    G = Z1.T @ Z1 + reg * len(Z1) * np.eye(Z1.shape[1])
    return np.linalg.solve(G, Z1.T @ X)


def latent_recovery(z: np.ndarray, x_true: np.ndarray) -> dict:
    """z: (n, T, d) model latents; x_true: (n, T, N) teacher state."""
    Z = z.reshape(-1, z.shape[-1])
    X = x_true.reshape(-1, x_true.shape[-1])
    n = min(len(Z), 40000)
    idx = np.linspace(0, len(Z) - 1, n).astype(int)
    Z, X = Z[idx], X[idx]
    cc = cca_correlations(Z, X)
    W = linear_map(Z, X)
    pred = np.concatenate([Z, np.ones((len(Z), 1))], 1) @ W
    ss_res = float(((X - pred) ** 2).sum())
    ss_tot = float(((X - X.mean(0, keepdims=True)) ** 2).sum())
    return {
        "cca_mean_top": float(cc[: min(8, len(cc))].mean()),
        "cca_all_mean": float(cc.mean()),
        "cca_spectrum": cc.tolist(),
        "linear_readout_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "map": W,
    }


def direction_recovery(
    z_unpert: np.ndarray,
    x_true: np.ndarray,
    model_dirs: dict[str, np.ndarray],
    true_dirs: dict[str, np.ndarray],
) -> dict:
    """Cosine similarity between the model's recovered intervention directions and
    the true injection directions, after mapping the latent space onto the
    teacher's state space using unperturbed data only."""
    Z = z_unpert.reshape(-1, z_unpert.shape[-1])
    X = x_true.reshape(-1, x_true.shape[-1])
    n = min(len(Z), 40000)
    idx = np.linspace(0, len(Z) - 1, n).astype(int)
    W = linear_map(Z[idx], X[idx])
    A = W[:-1]                                    # (d, N) latent -> state
    out = {}
    for name, md in model_dirs.items():
        if name not in true_dirs:
            continue
        mapped = md @ A
        td = true_dirs[name]
        nm = np.linalg.norm(mapped) * np.linalg.norm(td)
        out[name] = float(abs(mapped @ td) / nm) if nm > 0 else float("nan")
    # a chance level for this cosine: random directions in N dimensions
    N = X.shape[1]
    out["_chance"] = float(np.sqrt(2.0 / (np.pi * N)))
    return out
