"""Grouping of per-condition scores by intervention attribute.

The headline number aggregates every intervention condition, but the scientific
content lives in the breakdown: which *kinds* of intervention transfer across
animals and which do not. For the teacher benchmark the grouping variable is the
intervention direction (functionally-defined vs idiosyncratic); for ICMS it is
amplitude and the cortical depth of the stimulating contact.
"""

from __future__ import annotations

import numpy as np

from .. import metrics as M
from ..data.containers import AnimalTrials


def per_condition_scores(
    s: AnimalTrials, delta_pred: dict[int, np.ndarray], t_eval: slice | None = None
) -> dict[int, dict[str, float]]:
    y = s.y[:, s.t0 :]
    d_true, _ = M.measured_delta(y, s.cond, s.perturbed)
    out = {}
    for c in sorted(set(d_true) & set(delta_pred)):
        A, Bm = d_true[c], delta_pred[c]
        if t_eval is not None:
            A, Bm = A[t_eval], Bm[t_eval]
        out[int(c)] = {
            "delta_r2": M.delta_r2(A, Bm),
            "delta_corr": M.corr(A, Bm),
            "effect_norm_true": float(np.sqrt(np.mean(A**2))),
            "effect_norm_pred": float(np.sqrt(np.mean(Bm**2))),
            "n_trials": int((s.cond == c).sum()),
        }
    return out


def per_condition_ceiling(
    s: AnimalTrials, n_splits: int = 200, seed: int = 0
) -> dict[int, float]:
    """Split-half ceiling computed separately for each condition."""
    y = s.y[:, s.t0 :]
    out = {}
    for c in np.unique(s.cond[s.perturbed]):
        mask = (s.cond == c) | (~s.perturbed)
        sub_cond = s.cond[mask]
        sub_pert = s.perturbed[mask]
        ce = M.noise_ceiling(y[mask], sub_cond, sub_pert, n_splits=n_splits, seed=seed)
        out[int(c)] = ce["delta_r2_ceiling"]
    return out


def teacher_group_of(s: AnimalTrials, cond_id: int) -> tuple[str, float]:
    """(intervention type, amplitude) for a teacher-benchmark condition."""
    conds = s.meta["conds"]
    name, amp = conds[cond_id]
    return name, float(amp)


def group_teacher(
    s: AnimalTrials, scores: dict[int, dict[str, float]]
) -> dict[str, dict[str, float]]:
    by: dict[str, list[float]] = {}
    by_amp: dict[float, list[float]] = {}
    for c, sc in scores.items():
        name, amp = teacher_group_of(s, c)
        by.setdefault(name, []).append(sc["delta_r2"])
        by_amp.setdefault(amp, []).append(sc["delta_r2"])
    out = {f"type:{k}": {"delta_r2": float(np.nanmean(v)), "n": len(v)} for k, v in by.items()}
    out.update(
        {f"amp:{k:g}": {"delta_r2": float(np.nanmean(v)), "n": len(v)} for k, v in by_amp.items()}
    )
    conserved = [
        sc["delta_r2"]
        for c, sc in scores.items()
        if teacher_group_of(s, c)[0] != "idiosyncratic"
    ]
    idio = [
        sc["delta_r2"]
        for c, sc in scores.items()
        if teacher_group_of(s, c)[0] == "idiosyncratic"
    ]
    if conserved:
        out["group:conserved"] = {"delta_r2": float(np.nanmean(conserved)), "n": len(conserved)}
    if idio:
        out["group:idiosyncratic"] = {"delta_r2": float(np.nanmean(idio)), "n": len(idio)}
    return out


def group_icms(
    s: AnimalTrials, scores: dict[int, dict[str, float]]
) -> dict[str, dict[str, float]]:
    amp = s.meta.get("cond_amp", {})
    dep = s.meta.get("cond_depth_um", {})
    by_amp: dict[float, list[float]] = {}
    by_dep: dict[str, list[float]] = {}
    for c, sc in scores.items():
        a = amp.get(c, amp.get(str(c)))
        d = dep.get(c, dep.get(str(c)))
        if a is not None:
            by_amp.setdefault(float(a), []).append(sc["delta_r2"])
        if d is not None and np.isfinite(d):
            band = f"{int(d // 400) * 400}-{int(d // 400) * 400 + 400}um"
            by_dep.setdefault(band, []).append(sc["delta_r2"])
    out = {f"amp:{k:g}uA": {"delta_r2": float(np.nanmean(v)), "n": len(v)} for k, v in by_amp.items()}
    out.update(
        {f"depth:{k}": {"delta_r2": float(np.nanmean(v)), "n": len(v)} for k, v in by_dep.items()}
    )
    return out


def dose_response(
    s: AnimalTrials, delta_pred: dict[int, np.ndarray], amp_key: str = "cond_amp"
) -> dict:
    """Measured vs predicted effect magnitude as a function of amplitude."""
    y = s.y[:, s.t0 :]
    d_true, _ = M.measured_delta(y, s.cond, s.perturbed)
    amp = s.meta.get(amp_key, {})
    rows = []
    for c in sorted(set(d_true) & set(delta_pred)):
        a = amp.get(c, amp.get(str(c)))
        if a is None:
            continue
        rows.append(
            (
                float(a),
                float(np.sqrt(np.mean(d_true[c] ** 2))),
                float(np.sqrt(np.mean(delta_pred[c] ** 2))),
            )
        )
    if not rows:
        return {}
    rows.sort()
    a, t, p = map(np.array, zip(*rows))
    return {
        "amplitude": a.tolist(),
        "measured_norm": t.tolist(),
        "predicted_norm": p.tolist(),
        "corr_measured_vs_predicted": M.corr(t, p),
        "slope_ratio": float(np.polyfit(a, p, 1)[0] / (np.polyfit(a, t, 1)[0] + 1e-12)),
    }
