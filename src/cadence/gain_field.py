"""The gain-field model: a shared operator acting on each animal's own activity.

Every model tried so far predicts a stereotyped response from the stimulus
parameters, so it can only ever produce the average neuron. That is why a group
average over other animals matched it: there was nothing individual in the
prediction.

This model changes what is shared. A perturbation that suppresses or excites a
population does not add a fixed waveform to every cell, it *rescales what each cell
was already doing*. So write the effect on neuron n as

    Delta_n(t) = m(t, theta, x_n) * r_n(t)  +  a(t, theta, x_n)

where r_n(t) is that neuron's own mean firing on unperturbed trials, which is
available for a new animal without ever stimulating it, and x_n collects the
neuron's static properties (depth, distance from the light, cell class). The shared
part is the modulation field m and the additive field a, both smooth functions of
time, of the stimulus setting theta, and of the neuron's position.

The prediction is therefore individual: two neurons in the same animal under the
same light get different predicted responses because they were doing different
things. Nothing about the held-out animal's perturbation trials is used, only its
control trials, which is what the protocol allows.

Delta is linear in the shared coefficients, so the fit is a single ridge solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.containers import AnimalTrials


@dataclass
class GainFieldConfig:
    n_time_basis: int = 12          # smooth basis over the scored window
    ridge: float = 1e-2
    include_additive: bool = True   # the a(...) term
    include_multiplicative: bool = True
    use_depth: bool = True
    depth_sigmas: tuple[float, ...] = (0.08, 0.25, 0.7)
    amp_powers: tuple[float, ...] = (0.0, 1.0, 2.0, 0.5)
    smooth_control: int = 3         # bins of smoothing on the control profile
    center_control: bool = True     # also supply the mean-removed profile
    extra: dict = field(default_factory=dict)


def raised_cosine(T: int, n: int) -> np.ndarray:
    c = np.linspace(0, T - 1, n)
    w = max((c[1] - c[0]) if n > 1 else T, 1.0) * 2.0
    t = np.arange(T)[None, :]
    d = (t - c[:, None]) / w
    B = np.where(np.abs(d) <= 1, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
    B[0, : int(c[0]) + 1] = np.maximum(B[0, : int(c[0]) + 1], 1e-6)
    B[-1, int(c[-1]):] = np.maximum(B[-1, int(c[-1]):], 1e-6)
    return B


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, x)


def control_profile(s: AnimalTrials, cfg: GainFieldConfig) -> np.ndarray:
    """(T_post, n_obs) mean firing on unperturbed trials, per neuron."""
    y = s.y[~s.perturbed][:, s.t0 :]
    r = np.nanmean(y, 0)
    return _smooth(r, cfg.smooth_control)


def stim_features(power: float, gx: float, gy: float, cfg: GainFieldConfig) -> np.ndarray:
    p = power
    return np.array([p**q if q != 0.0 else 1.0 for q in cfg.amp_powers]
                    + [gx, gy], float)


def unit_features_gf(s: AnimalTrials, gy: float, cfg: GainFieldConfig) -> np.ndarray:
    """(n_obs, n_uf) static per-neuron features, including distance from the light."""
    d = np.asarray(s.meta["unit_y_um"], float)
    scale = max(np.nanmax(np.abs(d)), 1.0)
    dn = d / scale
    cols = [np.ones_like(dn), dn]
    if cfg.use_depth:
        dz = dn - gy / scale if np.isfinite(gy) else np.zeros_like(dn)
        cols += [dz, np.abs(dz)]
        cols += [np.exp(-((dz / w) ** 2)) for w in cfg.depth_sigmas]
    ct = [str(c).lower() for c in s.meta.get("cell_type", ["?"] * len(dn))]
    cols.append(np.array([1.0 if ("wide" in c or "pyr" in c) else 0.0 for c in ct]))
    return np.nan_to_num(np.stack(cols, 1))


def build_rows(s: AnimalTrials, conds, cfg: GainFieldConfig, measured):
    """Design rows for one observation set. Returns (X, y, cond_id, unit_id)."""
    T = s.T - s.t0
    B = raised_cosine(T, cfg.n_time_basis)          # (n_b, T)
    r = control_profile(s, cfg)                      # (T, n_obs)
    rc = r - r.mean(0, keepdims=True) if cfg.center_control else r
    dl = measured(s, conds)
    amp = s.meta["cond_amp"]
    galvo = s.meta.get("cond_galvo", {})
    dep = s.meta.get("cond_depth_um", {})

    X_rows, y_rows, cids, uids = [], [], [], []
    for c in conds:
        if c not in dl:
            continue
        p = float(amp[c]) if c in amp else float(amp[str(c)])
        if c in galvo:
            gx, gy = float(galvo[c][0]), float(galvo[c][1])
        else:
            gx, gy = 0.0, float(dep.get(c, dep.get(str(c), 0.0)))
        sf = stim_features(p, gx, gy, cfg)            # (n_sf,)
        uf = unit_features_gf(s, gy, cfg)             # (n_obs, n_uf)
        D = dl[c]                                    # (T, n_obs)
        # outer structure: time basis x stim features x unit features, applied
        # multiplicatively to the neuron's own control profile and additively
        for n in range(D.shape[1]):
            if not np.all(np.isfinite(D[:, n])):
                continue
            blocks = []
            su = np.outer(sf, uf[n]).ravel()          # (n_sf * n_uf,)
            if cfg.include_multiplicative:
                # B (n_b, T) times control profile of this neuron
                mult = B * rc[:, n][None, :]          # (n_b, T)
                blocks.append(np.einsum("bt,k->tbk", mult, su).reshape(T, -1))
            if cfg.include_additive:
                blocks.append(np.einsum("bt,k->tbk", B, su).reshape(T, -1))
            X_rows.append(np.concatenate(blocks, axis=1))
            y_rows.append(D[:, n])
            cids.append(c)
            uids.append(n)
    if not X_rows:
        return None
    return np.concatenate(X_rows, 0), np.concatenate(y_rows, 0), np.array(cids), np.array(uids)


def precompute(s: AnimalTrials, conds, cfg: GainFieldConfig, measured):
    """Normal-equation block for one set, computed once and reused by every fold.

    The design depends only on this set and on the stimulus settings it received,
    never on which other animals are being trained on, so leave-one-animal-out
    becomes a sum of cached matrices.
    """
    got = build_rows(s, conds, cfg, measured)
    if got is None:
        return None
    X, y, _, _ = got
    return X.T @ X, X.T @ y, len(y)


class BlockPool:
    """Normal equations aggregated per animal, so leave-one-out is a subtraction.

    Summing every session's block on every fold dominates the runtime once there
    are tens of animals and a nested loop for the ridge. Aggregating once per
    animal turns each fit into ``total - excluded``.
    """

    def __init__(self, blocks: dict, sets):
        self.per_animal: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        tot_xx = tot_xy = None
        for s in sets:
            b = blocks.get(s.key)
            if b is None:
                continue
            gxx, gxy, n = b
            w = 1.0 / max(n, 1)
            if tot_xx is None:
                tot_xx = np.zeros_like(gxx)
                tot_xy = np.zeros_like(gxy)
            tot_xx += w * gxx
            tot_xy += w * gxy
            if s.animal not in self.per_animal:
                self.per_animal[s.animal] = [np.zeros_like(gxx), np.zeros_like(gxy)]
            self.per_animal[s.animal][0] += w * gxx
            self.per_animal[s.animal][1] += w * gxy
        self.total = (tot_xx, tot_xy)

    def solve(self, exclude, cfg: GainFieldConfig) -> np.ndarray | None:
        if self.total[0] is None:
            return None
        xx = self.total[0].copy()
        xy = self.total[1].copy()
        for a in exclude:
            p = self.per_animal.get(a)
            if p is not None:
                xx -= p[0]
                xy -= p[1]
        scale = np.trace(xx) / xx.shape[0]
        if not np.isfinite(scale) or scale <= 0:
            return None
        return np.linalg.solve(xx + cfg.ridge * scale * np.eye(xx.shape[0]), xy)


def fit_from_blocks(blocks: dict, sets, cfg: GainFieldConfig) -> np.ndarray | None:
    XtX = Xty = None
    for s in sets:
        b = blocks.get(s.key)
        if b is None:
            continue
        gxx, gxy, n = b
        w = 1.0 / max(n, 1)                          # each set counts once
        if XtX is None:
            XtX = np.zeros_like(gxx)
            Xty = np.zeros_like(gxy)
        XtX += w * gxx
        Xty += w * gxy
    if XtX is None:
        return None
    scale = np.trace(XtX) / XtX.shape[0]
    return np.linalg.solve(XtX + cfg.ridge * scale * np.eye(XtX.shape[0]), Xty)


def fit(sets, conds_of, cfg: GainFieldConfig, measured) -> np.ndarray | None:
    """Convenience wrapper that builds the blocks on the fly."""
    blocks = {s.key: precompute(s, conds_of(s), cfg, measured) for s in sets}
    return fit_from_blocks(blocks, sets, cfg)


def predict_from_design(got, s: AnimalTrials, conds, theta):
    """Apply fitted coefficients to a cached design, avoiding a rebuild."""
    if got is None or theta is None:
        return {}
    X, _, cids, uids = got
    p = X @ theta
    T = s.T - s.t0
    out = {int(c): np.zeros((T, s.n_obs)) for c in conds}
    off = 0
    for i in range(len(cids)):
        out[int(cids[i])][:, uids[i]] = p[off : off + T]
        off += T
    return out


def predict(s: AnimalTrials, conds, cfg: GainFieldConfig, theta, measured):
    got = build_rows(s, conds, cfg, measured)
    if got is None or theta is None:
        return {}
    X, _, cids, uids = got
    p = X @ theta
    T = s.T - s.t0
    out = {int(c): np.zeros((T, s.n_obs)) for c in conds}
    # rows were concatenated in blocks of length T, one block per (condition, unit)
    off = 0
    for i in range(len(cids)):
        out[int(cids[i])][:, uids[i]] = p[off : off + T]
        off += T
    return out
