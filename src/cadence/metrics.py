"""Metrics for time-resolved causal-response prediction.

The quantity of interest is never the raw activity -- a model can score well on
raw activity while knowing nothing about the intervention. What matters is the
*causal effect*

    Delta_c(t, n) = E[ y(t, n) | intervention c ] - E[ y(t, n) | no intervention ]

and its behavioural analogue. ``delta_r2`` scores a predicted effect against the
measured effect with the *no-effect* model as the reference, so 0 means "no
better than asserting the intervention does nothing" and 1 means perfect.

Because Delta is estimated from finitely many trials it carries noise, so
``noise_ceiling`` estimates the highest Delta-R^2 any model could achieve given
the trial counts, via repeated random splits of the trials into halves.
"""

from __future__ import annotations

import numpy as np


def _sse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))


def delta_r2(delta_true: np.ndarray, delta_pred: np.ndarray) -> float:
    """Fraction of causal-effect variance explained, relative to predicting 0."""
    denom = float(np.sum(delta_true**2))
    if denom <= 0:
        return float("nan")
    return 1.0 - _sse(delta_true, delta_pred) / denom


def delta_r2_centered(delta_true: np.ndarray, delta_pred: np.ndarray) -> float:
    """Conventional R^2 with the mean of the true effect as reference."""
    mu = delta_true.mean()
    denom = float(np.sum((delta_true - mu) ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - _sse(delta_true, delta_pred) / denom


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def condition_average(y: np.ndarray, cond: np.ndarray) -> dict[int, np.ndarray]:
    return {int(c): y[cond == c].mean(0) for c in np.unique(cond)}


def measured_delta(
    y: np.ndarray, cond: np.ndarray, perturbed: np.ndarray, base_cond: int | None = None
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Measured causal effect per perturbation condition.

    Returns ``({cond: Delta}, baseline)`` where ``baseline`` is the average
    unperturbed response.
    """
    if base_cond is None:
        base = y[~perturbed].mean(0)
    else:
        base = y[cond == base_cond].mean(0)
    out = {}
    for c in np.unique(cond[perturbed]):
        out[int(c)] = y[cond == c].mean(0) - base
    return out, base


def noise_ceiling(
    y: np.ndarray,
    cond: np.ndarray,
    perturbed: np.ndarray,
    n_splits: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Split-half estimate of the attainable Delta-R^2 and Delta correlation.

    Each perturbation condition and the unperturbed pool are independently split
    in half; Delta is computed from each half; the two estimates are compared.
    The expected Delta-R^2 of a *perfect* model is obtained by correcting the
    split-half agreement for the fact that each half has half the trials.
    """
    rng = np.random.default_rng(seed)
    unp = np.where(~perturbed)[0]
    conds = [int(c) for c in np.unique(cond[perturbed])]
    r2s, rs = [], []
    for _ in range(n_splits):
        pu = rng.permutation(unp)
        b1 = y[pu[: len(pu) // 2]].mean(0)
        b2 = y[pu[len(pu) // 2 :]].mean(0)
        d1, d2 = [], []
        for c in conds:
            idx = np.where(cond == c)[0]
            if len(idx) < 4:
                continue
            p = rng.permutation(idx)
            d1.append(y[p[: len(p) // 2]].mean(0) - b1)
            d2.append(y[p[len(p) // 2 :]].mean(0) - b2)
        if not d1:
            continue
        d1 = np.stack(d1)
        d2 = np.stack(d2)
        r2s.append(delta_r2(d1, d2))
        rs.append(corr(d1, d2))
    if not r2s:
        return {"delta_r2_ceiling": float("nan"), "delta_corr_ceiling": float("nan")}
    # Split-half Delta uses n/2 trials per side, so its noise variance is twice
    # that of the full-data Delta. If s is the split-half agreement and
    # signal/noise decomposition holds, the full-data ceiling is
    #   R2_ceiling = 1 / (1 + v/2) where v solves s = (1 - v) / (1 + v).
    s = float(np.mean(r2s))
    v = max((1.0 - s) / (1.0 + s), 0.0) if s > -1 else np.inf
    ceiling = 1.0 / (1.0 + v / 2.0)
    r = float(np.mean(rs))
    r_ceiling = float(np.sqrt(max(2 * r / (1 + r), 0.0))) if r > -1 else float("nan")
    return {
        "delta_r2_ceiling": float(ceiling),
        "delta_r2_splithalf": s,
        "delta_corr_ceiling": min(r_ceiling, 1.0),
        "delta_corr_splithalf": r,
    }


def evaluate_delta(
    y_true: np.ndarray,
    cond: np.ndarray,
    perturbed: np.ndarray,
    delta_pred: dict[int, np.ndarray],
    t_eval: slice | None = None,
) -> dict[str, float]:
    """Score predicted causal effects against measured ones.

    ``y_true``: (n_trials, T, n_obs); ``delta_pred``: {cond: (T, n_obs)}.
    """
    d_true, _ = measured_delta(y_true, cond, perturbed)
    conds = sorted(set(d_true) & set(delta_pred))
    if not conds:
        return {"delta_r2": float("nan"), "delta_corr": float("nan"), "n_cond": 0}
    A = np.stack([d_true[c] for c in conds])
    B = np.stack([delta_pred[c] for c in conds])
    if t_eval is not None:
        A, B = A[:, t_eval], B[:, t_eval]
    per_cond = {int(c): delta_r2(A[i], B[i]) for i, c in enumerate(conds)}
    # per-unit correlation of the effect time course, averaged over units
    unit_r = [corr(A[:, :, n], B[:, :, n]) for n in range(A.shape[2])]
    if not np.any(np.isfinite(unit_r)):
        unit_r = [float("nan")]
        u_mean = u_med = float("nan")
    else:
        u_mean = float(np.nanmean(unit_r))
        u_med = float(np.nanmedian(unit_r))
    return {
        "delta_r2": delta_r2(A, B),
        "delta_r2_centered": delta_r2_centered(A, B),
        "delta_corr": corr(A, B),
        "delta_cosine": cosine(A, B),
        "delta_r2_per_cond": per_cond,
        "delta_r2_per_cond_mean": float(np.nanmean(list(per_cond.values()))),
        "unit_delta_corr_mean": u_mean,
        "unit_delta_corr_median": u_med,
        "n_cond": len(conds),
        "effect_norm_true": float(np.sqrt(np.mean(A**2))),
        "effect_norm_pred": float(np.sqrt(np.mean(B**2))),
    }


def raw_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 of the raw condition-averaged activity (reported for completeness)."""
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return 1.0 - _sse(y_true, y_pred) / denom


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(1)
    return (
        float(v.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_permutation_test(a, b, n_perm: int = 100000, seed: int = 0):
    """Two-sided paired permutation test on the mean difference a - b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return float("nan"), float("nan")
    d = a - b
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (signs * d).mean(1)
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    # exact-test flooring: with n paired observations the smallest attainable
    # two-sided p is 2^-(n-1)
    p = max(p, 2.0 ** (-(d.size - 1)))
    return float(obs), p


# ---------------------------------------------------------------------------
# Animal-level inference.
#
# Sessions from one animal are not independent replicates of a claim about
# animals. Treating them as such inflates significance (pseudoreplication), so
# every headline number in this project is inferred at the animal level, with
# session-level statistics reported only as secondary detail.
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(values, groups, n_boot: int = 20000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap that resamples *groups* (animals), not rows (sessions).

    Each resampled animal contributes all of its sessions, so the interval
    reflects between-animal variability.
    """
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if v.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_groups": 0}
    uniq = list(dict.fromkeys(g.tolist()))
    per = {u: v[g == u] for u in uniq}
    # animal-weighted point estimate: each animal counts once
    point = float(np.mean([per[u].mean() for u in uniq]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        means[i] = np.mean([per[uniq[j]].mean() for j in pick])
    return {
        "mean": point,
        "ci_lo": float(np.quantile(means, alpha / 2)),
        "ci_hi": float(np.quantile(means, 1 - alpha / 2)),
        "n_groups": len(uniq),
        "per_group": {str(u): float(per[u].mean()) for u in uniq},
    }


def animal_sign_test(per_animal, mu: float = 0.0):
    """Exact two-sided sign test over animals.

    With six animals the smallest attainable p is 2/2^6 = 0.031, reached when all
    six fall on the same side. The paper says so wherever this is quoted.
    """
    from scipy.stats import binomtest

    v = np.asarray([x for x in per_animal if np.isfinite(x)], float)
    n = v.size
    if n == 0:
        return {"n": 0, "n_positive": 0, "p": float("nan")}
    # a value that is zero to numerical precision is a tie, not a success; without
    # this the all-zero "the stimulus does nothing" row scores 6/6
    k = int(np.sum(v > mu + 1e-9))
    p = float(binomtest(k, n, 0.5, alternative="two-sided").pvalue) if n else float("nan")
    return {"n": n, "n_positive": k, "p": p, "p_floor": 2.0 ** (-(n - 1))}


MAX_EXACT_ANIMALS = 20        # 2^20 sign flips is a second; 2^39 is not a lifetime


def animal_permutation_test(per_animal_a, per_animal_b=None, seed: int = 0,
                            n_draws: int = 200_000):
    """Sign-flip permutation over animals.

    Every assignment is enumerated while that is cheap, which is the case for the
    cohort sizes this kind of experiment usually has. Beyond ``MAX_EXACT_ANIMALS`` the
    enumeration is 2^n and quietly becomes impossible, so the flips are sampled
    instead and the result is flagged as such. The alternative, silently waiting
    forever, is worse.
    """
    import itertools

    a = np.asarray(per_animal_a, float)
    b = np.zeros_like(a) if per_animal_b is None else np.asarray(per_animal_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    n = d.size
    if n == 0:
        return {"mean_diff": float("nan"), "p": float("nan"), "n": 0}
    obs = float(d.mean())
    if n <= MAX_EXACT_ANIMALS:
        null = np.asarray([float(np.mean(np.array(s) * d))
                           for s in itertools.product([-1, 1], repeat=n)])
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_draws, n))
        null = (signs * d).mean(axis=1)
        exact = False
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-15))
    if not exact:
        p = max(p, 1.0 / n_draws)          # a sampled p cannot be smaller than this
    return {"mean_diff": obs, "p": p, "n": n, "p_floor": 2.0 ** (-(n - 1)),
            "exact": exact, "n_draws": None if exact else n_draws}


def mixed_effects_intercept(values, groups):
    """Session-level estimate with an animal random intercept.

    Answers "is the mean effect different from zero once between-animal variance is
    accounted for", which is the right question when sessions are nested in animals.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    df = pd.DataFrame({"y": v[ok], "animal": g[ok]})
    if df["animal"].nunique() < 3:
        return {"error": "too few groups"}
    try:
        res = smf.mixedlm("y ~ 1", df, groups=df["animal"]).fit(reml=True)
        return {
            "estimate": float(res.params["Intercept"]),
            "se": float(res.bse["Intercept"]),
            "p": float(res.pvalues["Intercept"]),
            "n_obs": len(df),
            "n_groups": int(df["animal"].nunique()),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


def animal_level_report(values, groups, seed: int = 0) -> dict:
    """The full animal-level inference bundle for one method/readout."""
    boot = cluster_bootstrap_ci(values, groups, seed=seed)
    per = list(boot.get("per_group", {}).values())
    return {
        "animal_mean": boot["mean"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "n_animals": boot["n_groups"],
        "per_animal": boot.get("per_group", {}),
        "sign_test": animal_sign_test(per),
        "permutation": animal_permutation_test(per),
        "mixed_effects": mixed_effects_intercept(values, groups),
        "session_mean": float(np.nanmean(values)) if len(values) else float("nan"),
        "n_sessions": int(np.sum(np.isfinite(values))),
    }


def wilcoxon_signed_rank(a, b):
    from scipy.stats import wilcoxon

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(a[ok], b[ok])
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")
