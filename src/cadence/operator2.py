"""The cross-animal operator, written as a correction to the stereotyped response.

The first version of this model had to discover two things at once: the average
shape of a perturbation response, and the part of the response that is specific to
an individual neuron in an individual animal. The first part is easy and a group
average over other animals already captures it. The second part is the whole
scientific question. Asking one network to do both means its errors on the easy part
swamp whatever it learns about the hard part, and that is exactly what we saw: the
network matched the group average on well measured animals and fell apart on poorly
measured ones.

So the model is rewritten around the decomposition

    Delta_n(t)  =  g(t, theta)              stereotype, from other animals
                +  m_n(t) * r_n(t)          gain on what this neuron was doing
                +  a_n(t)                   an additive correction

where g is the group average over the training animals, r_n is neuron n's own mean
firing on unperturbed trials, and m and a are produced by a set transformer that
reads the whole simultaneously recorded population. The stereotype is handed to the
network both as an input and as a skip connection, so the network starts from the
baseline and every parameter it spends goes on the individual part.

One detail decides whether any of this means anything. The measured effect is the
stimulated mean minus the control mean, and the model is handed each neuron's control
mean as a feature. If the same trials produced both, their noise is shared with
opposite signs and a model can score above zero on noise alone. So the unperturbed
trials of every recording are split in two: one half builds every feature the model
sees, the other half defines the effect it is scored against. Nothing the model reads
shares a trial with what it is asked to predict.

Three further changes matter for animals where the recording is small or short:

  * neurons are randomly dropped during training, so the model is trained on
    populations of many sizes rather than only on the large ones,
  * the control profile is jittered by its own standard error, so the model learns
    that a profile measured from few trials is less trustworthy,
  * each animal contributes equally to the loss regardless of how many sessions it
    has, which is what the animal-level score asks for.

The stereotype handed to a training session is always computed with that session's
own animal left out, so no session is ever told its own answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import individuality as IND


@dataclass
class Operator2Config:
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.1
    n_coupling: int = 12
    smooth_control: int = 3
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_sessions: int = 8
    patience: int = 40
    grad_clip: float = 1.0
    huber_delta: float = 1.0
    keep_min: float = 0.35          # neuron dropout: smallest fraction kept
    keep_floor: int = 6             # never drop below this many neurons
    ctrl_jitter: float = 1.0        # multiples of the control standard error
    min_type_trials: int = 8        # trials of each type needed for a selectivity profile
    device: str = "cuda"
    seed: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the stereotype: a group average over animals, reusable across folds
# ---------------------------------------------------------------------------
class Stereotype:
    """Mean perturbation response across animals, as a function of dose.

    Built once from every animal's sessions, then evaluated with any set of animals
    excluded. Leave-one-animal-out therefore costs a mean over a handful of cached
    curves instead of a rebuild.
    """

    def __init__(self, sets, measured):
        self.by_animal: dict[str, dict[float, list]] = {}
        for t in sets:
            cs = [int(c) for c in np.unique(t.cond[t.perturbed])]
            for c, D in measured(t, cs).items():
                amp = round(float(t.meta["cond_amp"][c]), 3)
                self.by_animal.setdefault(t.animal, {}).setdefault(amp, []).append(
                    np.nanmean(D, 1))
        self._cache: dict[frozenset, tuple] = {}

    def curves(self, exclude) -> tuple[np.ndarray, np.ndarray] | None:
        key = frozenset(exclude)
        if key in self._cache:
            return self._cache[key]
        by: dict[float, list] = {}
        for a, tab in self.by_animal.items():
            if a in key:
                continue
            for amp, cur in tab.items():
                by.setdefault(amp, []).append(np.nanmean(cur, 0))
        if not by:
            self._cache[key] = None
            return None
        amps = np.array(sorted(by))
        stack = np.stack([np.nanmean(by[a], 0) for a in amps])
        self._cache[key] = (amps, stack)
        return self._cache[key]

    def predict(self, s, conds, exclude) -> dict[int, np.ndarray]:
        got = self.curves(exclude)
        if got is None:
            return {}
        amps, stack = got
        out = {}
        for c in conds:
            a = float(s.meta["cond_amp"][c])
            cv = (np.stack([np.interp(a, amps, stack[:, t])
                            for t in range(stack.shape[1])])
                  if len(amps) > 1 else stack[0])
            out[int(c)] = np.tile(cv[:, None], (1, s.n_obs))
        return out


# ---------------------------------------------------------------------------
def _smooth(x, k):
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, x)


def coupling_summary(s, n_comp: int, feat_idx) -> np.ndarray:
    y = s.y[feat_idx]
    flat = y.reshape(-1, s.n_obs).astype(np.float64)
    flat = flat - flat.mean(0, keepdims=True)
    cov = (flat.T @ flat) / max(len(flat) - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:n_comp]
    load = v[:, order] * np.sqrt(np.clip(w[order], 0, None))[None, :]
    for j in range(load.shape[1]):
        if load[np.argmax(np.abs(load[:, j])), j] < 0:
            load[:, j] *= -1
    out = np.zeros((s.n_obs, n_comp), dtype=np.float32)
    out[:, : load.shape[1]] = load
    return out / (np.abs(out).mean() + 1e-9)


def session_scale(yc, n_obs) -> float:
    v = float(np.nanstd(yc.reshape(-1, n_obs), axis=0).mean())
    return max(v, 1e-3)


def profiles(s, cfg: Operator2Config, yc, feat_idx) -> np.ndarray:
    """The activity patterns a perturbation can act on, all from control trials.

    Three channels per neuron, each a time course over the scored window:

      * its mean firing on unperturbed trials,
      * how much that firing has grown since the alignment point, which in a delay
        task is the neuron's preparatory ramp,
      * the difference between the two trial types, which is the neuron's selectivity.

    A perturbation that suppresses preparatory activity should act most strongly on
    neurons with a large ramp, and a perturbation that biases the upcoming choice
    should act along the selectivity axis. Giving the model these channels lets a
    single shared operator produce a different prediction for every neuron.
    """
    ctrl = _smooth(np.nanmean(yc, 0), cfg.smooth_control)           # (T, n_obs)
    pre = np.nanmean(s.y[feat_idx][:, : max(s.t0, 1)], axis=(0, 1))
    ramp = ctrl - pre[None, :]
    sel = np.zeros_like(ctrl)
    if s.behavior is not None:
        ch = s.behavior[feat_idx][:, 0, 0]
        fin = np.isfinite(ch)
        binary = fin.any() and np.all(np.isin(ch[fin], (0.0, 1.0)))
        L, R = (ch == 0) & binary, (ch == 1) & binary
        if L.sum() >= cfg.min_type_trials and R.sum() >= cfg.min_type_trials:
            sel = _smooth(np.nanmean(yc[R], 0) - np.nanmean(yc[L], 0),
                          cfg.smooth_control)
    return np.nan_to_num(np.stack([ctrl, ramp, sel], 0))            # (P, T, n_obs)


def session_tensors(s, cfg: Operator2Config, measured, base: dict, cache=None):
    """Per-neuron features, per-condition stimulus features, stereotype and target.

    Only the stereotype changes between folds and between seeds, so everything else is
    computed once per recording and reused. Without this the eigendecompositions and
    trial averages are repeated once per fold per seed, which dominates the runtime
    long before any of the training does.
    """
    if cache is not None and s.key in cache:
        st = cache[s.key]
        if st is None:
            return None
        conds = [c for c in st["conds"] if c in base]
        if not conds:
            return None
        keep = [st["conds"].index(c) for c in conds]
        return dict(neu=st["neu"], stim=st["stim"][keep], rel=st["rel"][keep],
                    tgt=st["tgt"][keep], prof=st["prof"], se=st["se"],
                    conds=conds, scale=st["scale"],
                    base=np.stack([(np.nan_to_num(base[c]).T / st["scale"]
                                    ).astype(np.float32) for c in conds]))
    # one half of the unperturbed trials builds every feature, the other half defines
    # the effect, so nothing the model reads shares a trial with its target
    feat_idx, _ = IND.control_split(s)
    yc = s.y[feat_idx][:, s.t0 :]
    n_ctrl = max(len(yc), 1)
    prof = profiles(s, cfg, yc, feat_idx)                           # (P, T, n_obs)
    ctrl = prof[0]
    ctrl_sd = np.nanstd(yc, 0).mean(0) + 1e-6                       # (n_obs,)
    ctrl_se = ctrl_sd / np.sqrt(n_ctrl)
    stat = s.unit_features if s.unit_features is not None else np.zeros((s.n_obs, 1))
    coup = coupling_summary(s, cfg.n_coupling, feat_idx)
    depth = np.asarray(s.meta["unit_y_um"], float)
    dscale = max(np.nanmax(np.abs(depth)), 1.0)

    conds = [int(c) for c in np.unique(s.cond[s.perturbed])]
    dl = measured(s, conds)
    conds = [c for c in conds if c in dl]
    if not conds:
        if cache is not None:
            cache[s.key] = None
        return None

    scale = session_scale(yc, s.n_obs)
    prof = prof / scale
    ctrl = prof[0]
    se = ctrl_se / scale
    neu = np.nan_to_num(np.concatenate([
        np.concatenate([p.T for p in prof], axis=1),   # the P profile channels
        np.log1p(np.abs(ctrl.T)) * np.sign(ctrl.T),
        stat,
        coup,
        (depth / dscale)[:, None],
        np.log1p(ctrl_sd)[:, None],
        np.full((s.n_obs, 1), np.log1p(n_ctrl)),
        np.full((s.n_obs, 1), np.log1p(s.n_obs)),
    ], axis=1).astype(np.float32))

    stims, targets, rel = [], [], []
    for c in conds:
        p = float(s.meta["cond_amp"][c])
        if "cond_galvo" in s.meta:
            gx, gy = (float(v) for v in s.meta["cond_galvo"][c])
        else:
            gx, gy = 0.0, float(s.meta["cond_depth_um"][c])
        dz = (depth - gy) / dscale
        n_st = int((s.cond == c).sum())
        stims.append(np.array([p, p ** 2, np.sqrt(max(p, 0.0)), np.log1p(p), gx,
                               gy / dscale, np.log1p(n_st), np.log1p(n_ctrl),
                               np.log1p(s.n_obs)], np.float32))
        rel.append(np.stack([dz, np.abs(dz),
                             np.exp(-((dz / 0.1) ** 2)),
                             np.exp(-((dz / 0.3) ** 2)),
                             np.exp(-((dz / 0.8) ** 2))], 1).astype(np.float32))
        targets.append((np.nan_to_num(dl[c]).T / scale).astype(np.float32))
    T = targets[0].shape[1]
    st = dict(neu=neu, stim=np.stack(stims), rel=np.stack(rel),
              tgt=np.stack(targets),
              prof=np.stack([p.T[:, :T] for p in prof], 1).astype(np.float32),
              se=se.astype(np.float32), conds=conds, scale=scale)
    if cache is not None:
        cache[s.key] = st
    keep = [i for i, c in enumerate(conds) if c in base]
    if not keep:
        return None
    out = dict(st)
    out["conds"] = [conds[i] for i in keep]
    for k in ("stim", "rel", "tgt"):
        out[k] = st[k][keep]
    out["base"] = np.stack([(np.nan_to_num(base[c]).T / scale).astype(np.float32)
                            for c in out["conds"]])
    return out


def pack(sets, cfg, measured, base_of, cache=None):
    ex = []
    for s in sets:
        got = session_tensors(s, cfg, measured, base_of(s), cache)
        if got is None:
            continue
        for i, c in enumerate(got["conds"]):
            ex.append(dict(key=s.key, animal=s.animal, cond=c,
                           n=got["neu"].shape[0], scale=got["scale"],
                           neu=torch.as_tensor(got["neu"]),
                           stim=torch.as_tensor(got["stim"][i]),
                           rel=torch.as_tensor(got["rel"][i]),
                           tgt=torch.as_tensor(got["tgt"][i]),
                           base=torch.as_tensor(got["base"][i]),
                           prof=torch.as_tensor(got["prof"]),
                           se=torch.as_tensor(got["se"])))
    return ex


def collate(batch, device, rng=None, cfg: Operator2Config | None = None):
    """Pad to the largest population in the batch.

    When an rng is supplied the batch is augmented: a random subset of neurons is
    kept and the control profile is jittered by its own standard error. Both make
    the model usable on recordings much smaller than the ones it was trained on.
    """
    keep_idx = []
    for b in batch:
        n = b["n"]
        if rng is None or cfg is None:
            keep_idx.append(np.arange(n))
            continue
        lo = max(cfg.keep_floor, int(np.ceil(cfg.keep_min * n)))
        k = n if lo >= n else int(rng.integers(lo, n + 1))
        keep_idx.append(rng.permutation(n)[:k] if k < n else np.arange(n))

    N = max(len(k) for k in keep_idx)
    B = len(batch)
    Fn = batch[0]["neu"].shape[1]
    R = batch[0]["rel"].shape[1]
    T = batch[0]["tgt"].shape[1]
    P = batch[0]["prof"].shape[1]
    neu = torch.zeros(B, N, Fn); rel = torch.zeros(B, N, R)
    tgt = torch.zeros(B, N, T); prof = torch.zeros(B, N, P, T)
    base = torch.zeros(B, N, T)
    mask = torch.zeros(B, N, dtype=torch.bool)
    stim = torch.stack([b["stim"] for b in batch])
    for i, b in enumerate(batch):
        idx = torch.as_tensor(np.asarray(keep_idx[i]), dtype=torch.long)
        k = len(idx)
        nb = b["neu"][idx].clone()
        pb = b["prof"][idx].clone()
        if rng is not None and cfg is not None and cfg.ctrl_jitter > 0:
            se = b["se"][idx][:, None]
            noise = torch.as_tensor(
                rng.standard_normal((k, T)).astype(np.float32)) * se * cfg.ctrl_jitter
            # jitter the measured profiles by their own standard error, and keep the
            # feature block and the multiplicative channels consistent
            pb[:, 0] += noise
            nb[:, :T] = pb[:, 0]
            nb[:, P * T : (P + 1) * T] = (torch.log1p(pb[:, 0].abs())
                                          * torch.sign(pb[:, 0]))
        neu[i, :k] = nb
        rel[i, :k] = b["rel"][idx]
        tgt[i, :k] = b["tgt"][idx]
        base[i, :k] = b["base"][idx]
        prof[i, :k] = pb
        mask[i, :k] = True
    return (neu.to(device), stim.to(device), rel.to(device), prof.to(device),
            base.to(device), tgt.to(device), mask.to(device))


# ---------------------------------------------------------------------------
class Operator2(nn.Module):
    def __init__(self, cfg: Operator2Config, n_neu: int, n_stim: int, n_rel: int,
                 T: int, n_prof: int = 3):
        super().__init__()
        self.cfg = cfg
        self.T, self.P = T, n_prof
        d = cfg.d_model
        self.neu_in = nn.Sequential(nn.LayerNorm(n_neu), nn.Linear(n_neu, d),
                                    nn.GELU(), nn.Linear(d, d))
        self.stim_in = nn.Sequential(nn.Linear(n_stim + n_rel, d), nn.GELU(),
                                     nn.Linear(d, d))
        self.base_in = nn.Sequential(nn.LayerNorm(T), nn.Linear(T, d), nn.GELU(),
                                     nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.add = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.d_ff), nn.GELU(),
                                 nn.Linear(cfg.d_ff, T))
        self.gain = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_prof * T))
        self.bscale = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, T))
        # start as the stereotype exactly: corrections are zero at initialisation
        for m in (self.add[-1], self.gain[-1], self.bscale[-1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, neu, stim, rel, prof, base, mask):
        """prof is (B, N, P, T): the activity channels the operator acts on."""
        B, N, _ = neu.shape
        x = (self.neu_in(neu)
             + self.stim_in(torch.cat([stim[:, None, :].expand(-1, N, -1), rel], -1))
             + self.base_in(base))
        x = self.enc(x, src_key_padding_mask=~mask)
        g = torch.tanh(self.gain(x)).view(B, N, self.P, self.T)
        # the stereotype rescaled per neuron, plus a gain applied to each of that
        # neuron's own activity channels, plus a free additive term
        return (base * (1.0 + torch.tanh(self.bscale(x)))
                + (g * prof).sum(2)
                + self.add(x))


def masked_loss(pred, tgt, mask, delta: float, weight=None, eps: float = 1e-3):
    """One minus the score, as a loss.

    The error on each example is divided by that example's own effect energy, which
    is the same normalisation the score uses, and examples are then weighted so that
    every animal counts equally.
    """
    m = mask[..., None].float()
    err = F.huber_loss(pred * m, tgt * m, delta=delta, reduction="none")
    per = err.sum(dim=(1, 2))
    energy = ((tgt * m) ** 2).sum(dim=(1, 2)) + eps * m.sum(dim=(1, 2)) * tgt.shape[-1]
    r = per / energy.clamp_min(1e-6)
    if weight is None:
        return r.mean()
    w = weight / weight.sum().clamp_min(1e-9)
    return (r * w).sum()
