"""Splitting a perturbation response into a shared part and an individual part.

A group average over other animals hands every neuron in a new animal the same
predicted response. It is a strong baseline for the total response, and that is
exactly what makes the total response a blunt instrument: a model can score well on
it while knowing nothing about individual neurons.

So write the measured effect on neuron n at time t as

    Delta_n(t)  =  mean over neurons of Delta(t)   +   delta_n(t)

The first term is what every neuron in the population does together. The second is
what is specific to that neuron. Scored on delta, a stereotype gets exactly zero, not
approximately zero, because its prediction is constant across neurons and so has no
delta at all. Anything above zero is prediction of individual structure in an animal
whose perturbation trials were never seen.

This module provides

  * ``delta`` and ``delta_ceiling``, the quantity and how much of it is measurable,
  * ``SharedOperator``, a shared linear operator that acts on each neuron's own
    control activity and is fitted leave-one-animal-out,

so that the same decomposition can be applied to real recordings and to simulated
cohorts where the true amount of shared structure is known.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np


def centre(x: np.ndarray) -> np.ndarray:
    """Remove the across-neuron mean at each time point."""
    return x - np.nanmean(x, axis=-1, keepdims=True)


def raised_cosine(T: int, n: int) -> np.ndarray:
    c = np.linspace(0, T - 1, n)
    w = max((c[1] - c[0]) if n > 1 else T, 1.0) * 2.0
    t = np.arange(T)[None, :]
    d = (t - c[:, None]) / w
    return np.where(np.abs(d) <= 1, 0.5 * (1 + np.cos(np.pi * d)), 0.0)


def _smooth(x, k):
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, x)


def control_split(s, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves of the unperturbed trials.

    The measured effect is the stimulated mean minus the control mean, and the model
    is handed each neuron's control mean as a feature. If the same trials produced
    both, their noise is shared, with opposite signs, and a model could score above
    zero on noise alone. Splitting the control trials removes that path entirely:
    one half builds the features the model sees, the other half defines the effect it
    is scored against.
    """
    idx = np.flatnonzero(~s.perturbed)
    rng = np.random.default_rng(zlib.crc32(s.key.encode()) + seed)
    idx = rng.permutation(idx)
    h = len(idx) // 2
    return np.sort(idx[:h]), np.sort(idx[h:])


def score(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.nansum((a - b) ** 2))
    e = float(np.nansum(a ** 2))
    return 1.0 - d / e if e > 0 else float("nan")


def delta_ceiling(s, min_trials: int = 6) -> float:
    """How much of the individual part is measurable rather than noise.

    The effect is a difference of two trial means, so its noise variance is
    v(1/n + 1/n0) with v the trial-to-trial variance, n stimulation trials and n0
    control trials. Removing the across-neuron mean costs a factor (1 - 1/N). The
    ceiling is the score a perfect predictor of the true individual part would get
    against this estimate of it.
    """
    Y = s.y[:, s.t0 :]
    _, ci = control_split(s)                 # the half that defines the effect
    if len(ci) < 4:
        return float("nan")
    n0 = len(ci)
    b = np.nanmean(Y[ci], 0)
    v0 = np.nanvar(Y[ci], 0)
    noise = sig = 0.0
    for c in np.unique(s.cond[s.perturbed]):
        idx = np.flatnonzero(s.cond == c)
        if len(idx) < min_trials:
            continue
        vc = np.nanvar(Y[idx], 0)
        d = centre(np.nanmean(Y[idx], 0) - b)
        N = d.shape[1]
        noise += float(np.nansum((vc / len(idx) + v0 / n0) * (1.0 - 1.0 / N)))
        sig += float(np.nansum(d ** 2))
    return 1.0 - noise / sig if sig > 0 else float("nan")


# ---------------------------------------------------------------------------
@dataclass
class SharedOperatorConfig:
    n_time_basis: int = 6
    dose_powers: tuple[float, ...] = (0.0, 1.0, 0.5)
    use_depth: bool = True
    depth_bumps: tuple[float, ...] = (0.15, 0.4, 0.65, 0.9)
    depth_width: float = 0.22
    use_selectivity: bool = True
    smooth: int = 3
    min_trials: int = 6
    min_type_trials: int = 8
    ridge: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    extra: dict = field(default_factory=dict)


class SharedOperator:
    """One operator, shared by every animal, acting on each animal's own activity.

    The predicted individual part of neuron n's response is a linear combination of
    that neuron's own control activity channels, with coefficients that depend
    smoothly on time, on the dose, and on the neuron's position. All the
    coefficients are shared across animals; everything animal-specific enters only
    through the neuron's own measured activity.

    Fitting is a ridge solve on cached normal equations, so leaving an animal out
    costs a subtraction.
    """

    def __init__(self, cfg: SharedOperatorConfig | None = None):
        self.cfg = cfg or SharedOperatorConfig()
        self.ex: list[dict] = []
        self._blocks: list[tuple] = []

    # -- feature construction, from control trials and the stimulus settings only --
    def _channels(self, s, feat_idx) -> np.ndarray:
        cfg = self.cfg
        yc = s.y[feat_idx][:, s.t0 :]
        ctrl = _smooth(np.nanmean(yc, 0), cfg.smooth)
        pre = np.nanmean(s.y[feat_idx][:, : max(s.t0, 1)], axis=(0, 1))
        # a constant channel lets the operator place an effect by position alone,
        # which is the only thing available when a neuron's ongoing activity carries
        # no structure; the remaining channels act on what the neuron was doing
        chans = [np.ones_like(ctrl), centre(ctrl), centre(ctrl - pre[None, :])]
        if cfg.use_selectivity and s.behavior is not None:
            ch = s.behavior[feat_idx][:, 0, 0]
            fin = np.isfinite(ch)
            binary = fin.any() and np.all(np.isin(ch[fin], (0.0, 1.0)))
            L, R = (ch == 0) & binary, (ch == 1) & binary
            if L.sum() >= cfg.min_type_trials and R.sum() >= cfg.min_type_trials:
                chans.append(centre(np.nanmean(yc[R], 0) - np.nanmean(yc[L], 0)))
            else:
                chans.append(np.zeros_like(ctrl))
        return np.nan_to_num(np.stack(chans, -1))            # (T, N, P)

    def add(self, s, amp_scale: float = 1.0, tag: str = "", null: bool = False) -> None:
        """Add one recording.

        With ``null`` set, the stimulation trials are ignored and the effect is
        formed from two disjoint groups of unperturbed trials instead, so the thing
        being predicted is known to be zero. Everything else about the analysis is
        identical, which makes it a direct check that the procedure cannot manufacture
        a positive score out of noise.

        ``amp_scale`` divides the stimulus setting, which is what lets recordings
        from different perturbation techniques be pooled: microamps and milliwatts
        are not comparable, but each expressed as a fraction of the largest dose that
        technique delivered is. ``tag`` marks which cohort a recording came from so
        that a fit can be held out by cohort as well as by animal.
        """
        cfg = self.cfg
        Y = s.y[:, s.t0 :]
        feat_idx, base_idx = control_split(s)
        if null:
            # split at random rather than in time order. Splitting a sorted list puts
            # every early trial on one side and every late trial on the other, so any
            # drift through a session shows up as a systematic effect, and the null
            # stops being a null.
            rng = np.random.default_rng(zlib.crc32(("null" + s.key).encode()))
            shuffled = rng.permutation(base_idx)
            h = len(shuffled) // 2
            pseudo_stim, base_idx = np.sort(shuffled[:h]), np.sort(shuffled[h:])
        base = np.nanmean(Y[base_idx], 0)
        P = self._channels(s, feat_idx)
        T = Y.shape[1]
        B = raised_cosine(T, cfg.n_time_basis)
        raw_d = np.asarray(s.meta["unit_y_um"], float)
        dscale = max(np.nanmax(np.abs(raw_d)), 1.0)
        d = np.nan_to_num(raw_d / dscale)
        depth_of = s.meta.get("cond_depth_um", {})
        for c in np.unique(s.cond[s.perturbed]):
            idx = s.cond == c
            if idx.sum() < cfg.min_trials:
                continue
            src = pseudo_stim if null else np.flatnonzero(idx)
            y = centre(np.nanmean(Y[src], 0) - base)
            p = float(s.meta["cond_amp"][c]) / max(amp_scale, 1e-9)
            pf = np.array([1.0 if q == 0.0 else p ** q for q in cfg.dose_powers])
            if cfg.use_depth:
                cols = [np.ones_like(d), d]
                cols += [np.exp(-((d - m) / cfg.depth_width) ** 2)
                         for m in cfg.depth_bumps]
                # position relative to where this condition delivered the perturbation
                gy = float(depth_of.get(int(c), depth_of.get(str(int(c)), 0.0))) / dscale
                cols += [np.exp(-(((d - gy) / w) ** 2)) for w in (0.1, 0.3)]
                uf = np.stack(cols, 1)
            else:
                uf = np.ones((len(d), 1))
            X = np.einsum("tnp,bt,k,nu->tnpbku", P, B, pf, uf)
            self.ex.append(dict(animal=s.animal, key=s.key, cond=int(c), tag=tag,
                                X=np.nan_to_num(
                                    X.reshape(T, P.shape[1], -1).astype(np.float32)),
                                y=y.astype(np.float32)))

    def _prepare(self) -> None:
        """Normal equations, aggregated per animal so that leaving one out is a
        subtraction rather than a re-sum over every recording."""
        if self._blocks:
            return
        K = self.ex[0]["X"].shape[-1]
        self.K = K
        self._per: dict[tuple, list] = {}
        for e in self.ex:
            Xf = e["X"].reshape(-1, K).astype(np.float64)
            yf = e["y"].ravel().astype(np.float64)
            m = np.isfinite(yf)
            w = 1.0 / max(int(m.sum()), 1)          # every example counts once
            blk = (w * Xf[m].T @ Xf[m], w * Xf[m].T @ yf[m])
            self._blocks.append(blk)
            key = (e["animal"], e["tag"])
            if key not in self._per:
                self._per[key] = [np.zeros((K, K)), np.zeros(K)]
            self._per[key][0] += blk[0]
            self._per[key][1] += blk[1]

    def fit(self, exclude, ridge: float, keep_tags=None) -> np.ndarray:
        self._prepare()
        XX = np.zeros((self.K, self.K))
        Xy = np.zeros(self.K)
        for (animal, tag), (a, b) in self._per.items():
            if animal in exclude:
                continue
            if keep_tags is not None and tag not in keep_tags:
                continue
            XX += a
            Xy += b
        return np.linalg.solve(XX + ridge * (np.trace(XX) / self.K) * np.eye(self.K),
                               Xy)

    def evaluate(self, animal: str, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float32)
        n = de = 0.0
        for e in self.ex:
            if e["animal"] != animal:
                continue
            n += float(np.nansum((e["y"] - e["X"] @ theta) ** 2))
            de += float(np.nansum(e["y"] ** 2))
        return 1.0 - n / de if de > 0 else float("nan")

    def cross_cohort(self, train_tag: str, test_tag: str, ridge: float = 1.0) -> dict:
        """Fit on one cohort, predict the individual part in every animal of another.

        Nothing about the test cohort enters the fit, not one animal and not one
        session, so this asks whether the operator is shared across a change of
        perturbation technique and brain area rather than across animals alone.
        """
        self._prepare()
        theta = self.fit(set(), ridge, keep_tags={train_tag})
        animals = sorted({e["animal"] for e in self.ex if e["tag"] == test_tag})
        return {a: self.evaluate(a, theta) for a in animals}

    def loao(self) -> dict:
        """Leave one animal out, with the ridge chosen inside the training animals.

        For each held-out animal the ridge is selected by a further leave-one-out
        over the remaining animals, so the held-out animal never influences any
        choice made about the model.
        """
        self._prepare()
        animals = sorted({e["animal"] for e in self.ex})
        out, preds = {}, {}
        for a in animals:
            inner = [b for b in animals if b != a]
            best, best_lam = -np.inf, self.cfg.ridge[0]
            for lam in self.cfg.ridge:
                # pooled error over the inner animals rather than the mean of their
                # individual scores. An animal with a very small effect divides by a
                # very small number, so a mean of per-animal scores lets one such
                # animal choose the ridge for everybody, and it chooses badly.
                n = de = 0.0
                for b in inner:
                    th = self.fit({a, b}, lam)
                    for e in self.ex:
                        if e["animal"] != b:
                            continue
                        n += float(np.nansum((e["y"] - e["X"] @ th) ** 2))
                        de += float(np.nansum(e["y"] ** 2))
                v = 1.0 - n / de if de > 0 else -np.inf
                if v > best:
                    best, best_lam = v, lam
            th = self.fit({a}, best_lam)
            n = de = 0.0
            for e in self.ex:
                if e["animal"] != a:
                    continue
                p = e["X"] @ th
                preds[(e["key"], e["cond"])] = p
                n += float(np.nansum((e["y"] - p) ** 2))
                de += float(np.nansum(e["y"] ** 2))
            out[a] = dict(delta_r2=1.0 - n / de if de > 0 else np.nan, ridge=best_lam)
        self.preds = preds
        return out
