"""A learned cross-animal perturbation operator.

Every earlier model was linear in hand-built features, and each one landed in the
same place: it could predict the average neuron, so a group average over other
animals matched it. The missing ingredient is a model that reads what an individual
neuron is doing, in the context of the population it sits in, and maps that to how
the stimulus will move it.

The model is a set transformer over neurons. One token per recorded neuron carries

  * that neuron's own mean firing profile on unperturbed trials,
  * its static properties (depth on the probe, cell class, firing statistics),
  * how it couples to the rest of the population during spontaneous activity,
  * where it sits relative to the stimulus, and the stimulus settings themselves.

Attention runs across the neurons of one recording, so a neuron's predicted response
can depend on the population it is embedded in. The decoder emits the whole
time-resolved effect for that neuron.

Everything the model sees about a held-out animal comes from unperturbed trials and
from the stimulus parameters, which the experimenter chooses. No stimulation trial
from the held-out animal is read at any point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OperatorConfig:
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.1
    n_coupling: int = 16        # rank of the spontaneous-coupling summary
    smooth_control: int = 3
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 260
    batch_sessions: int = 8
    patience: int = 45
    grad_clip: float = 1.0
    huber_delta: float = 1.0
    device: str = "cuda"
    seed: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# per-recording feature construction, from unperturbed trials only
# ---------------------------------------------------------------------------
def _smooth(x, k):
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, x)


def coupling_summary(s, n_comp: int) -> np.ndarray:
    """Each neuron's loading on the leading modes of spontaneous covariance.

    Signs are made deterministic so that the summary is comparable across animals
    even though the modes themselves are not.
    """
    y = s.y[~s.perturbed]
    flat = y.reshape(-1, s.n_obs).astype(np.float64)
    flat = flat - flat.mean(0, keepdims=True)
    cov = (flat.T @ flat) / max(len(flat) - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:n_comp]
    load = v[:, order] * np.sqrt(np.clip(w[order], 0, None))[None, :]
    # fix the sign of each mode by its dominant entry, then sort by |loading| so
    # the columns mean the same thing in every animal
    for j in range(load.shape[1]):
        if load[np.argmax(np.abs(load[:, j])), j] < 0:
            load[:, j] *= -1
    out = np.zeros((s.n_obs, n_comp), dtype=np.float32)
    out[:, : load.shape[1]] = load
    sd = np.abs(out).mean() + 1e-9
    return out / sd


def session_scale(s) -> float:
    """A per-recording activity scale, from unperturbed trials only.

    Firing rates differ by an order of magnitude between animals and probes. The
    model is asked to predict the effect *in units of that animal's own activity*,
    and the prediction is multiplied back by this scale afterwards. Because the
    scale is measured on control trials it costs nothing from the protocol, and it
    stops a few high-rate recordings from dominating the loss.
    """
    yc = s.y[~s.perturbed][:, s.t0 :]
    v = float(np.nanstd(yc.reshape(-1, s.n_obs), axis=0).mean())
    return max(v, 1e-3)


def session_tensors(s, cfg: OperatorConfig, measured):
    """Returns (neuron features, stimulus features per condition, targets)."""
    T = s.T - s.t0
    yc = s.y[~s.perturbed][:, s.t0 :]
    ctrl = _smooth(np.nanmean(yc, 0), cfg.smooth_control)             # (T, n_obs)
    ctrl_sd = np.nanstd(yc, 0).mean(0) + 1e-6                          # (n_obs,)
    stat = s.unit_features if s.unit_features is not None else np.zeros((s.n_obs, 1))
    coup = coupling_summary(s, cfg.n_coupling)
    depth = np.asarray(s.meta["unit_y_um"], float)
    dscale = max(np.nanmax(np.abs(depth)), 1.0)

    conds = [int(c) for c in np.unique(s.cond[s.perturbed])]
    dl = measured(s, conds)
    conds = [c for c in conds if c in dl]
    if not conds:
        return None

    scale = session_scale(s)
    ctrl = ctrl / scale
    neu = np.concatenate([
        ctrl.T,                                    # (n_obs, T) own profile
        np.log1p(np.abs(ctrl.T)) * np.sign(ctrl.T),
        stat,
        coup,
        (depth / dscale)[:, None],
        np.log1p(ctrl_sd)[:, None],
    ], axis=1).astype(np.float32)
    neu = np.nan_to_num(neu)

    stims, targets, rel = [], [], []
    for c in conds:
        p = float(s.meta["cond_amp"][c])
        if "cond_galvo" in s.meta:
            gx, gy = (float(v) for v in s.meta["cond_galvo"][c])
        else:
            gx, gy = 0.0, float(s.meta["cond_depth_um"][c])
        dz = (depth - gy) / dscale
        stims.append(np.array([p, p**2, np.sqrt(max(p, 0.0)), np.log1p(p), gx,
                               gy / dscale], np.float32))
        rel.append(np.stack([dz, np.abs(dz),
                             np.exp(-((dz / 0.1) ** 2)),
                             np.exp(-((dz / 0.3) ** 2)),
                             np.exp(-((dz / 0.8) ** 2))], 1).astype(np.float32))
        targets.append((np.nan_to_num(dl[c]).T / scale).astype(np.float32))
    return neu, np.stack(stims), np.stack(rel), np.stack(targets), conds, scale


# ---------------------------------------------------------------------------
class NeuralOperator(nn.Module):
    def __init__(self, cfg: OperatorConfig, n_neu_feat: int, n_stim: int,
                 n_rel: int, T: int):
        super().__init__()
        self.cfg = cfg
        self.T = T
        d = cfg.d_model
        self.neu_in = nn.Sequential(
            nn.LayerNorm(n_neu_feat), nn.Linear(n_neu_feat, d), nn.GELU(),
            nn.Linear(d, d))
        self.stim_in = nn.Sequential(nn.Linear(n_stim + n_rel, d), nn.GELU(),
                                     nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.d_ff), nn.GELU(),
                                  nn.Linear(cfg.d_ff, T))
        # a multiplicative path, so the model can express "scale what this neuron
        # was already doing" without having to learn it from scratch
        self.gain = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, T))

    def forward(self, neu, stim, rel, ctrl, mask):
        """neu (B,N,F) stim (B,S) rel (B,N,R) ctrl (B,N,T) mask (B,N) -> (B,N,T)."""
        B, N, _ = neu.shape
        h = self.neu_in(neu)
        s = self.stim_in(torch.cat([stim[:, None, :].expand(-1, N, -1), rel], -1))
        x = h + s
        x = self.enc(x, src_key_padding_mask=~mask)
        add = self.head(x)
        g = torch.tanh(self.gain(x))
        return add + g * ctrl


# ---------------------------------------------------------------------------
def pack_fold(sets, cfg: OperatorConfig, measured, device):
    """One example per (session, condition)."""
    ex = []
    for s in sets:
        got = session_tensors(s, cfg, measured)
        if got is None:
            continue
        neu, stims, rel, tgt, conds, scale = got
        T = tgt.shape[-1]
        ctrl = neu[:, :T]
        for i, c in enumerate(conds):
            ex.append(dict(
                key=s.key, animal=s.animal, cond=c, n=neu.shape[0], scale=scale,
                neu=torch.as_tensor(neu), stim=torch.as_tensor(stims[i]),
                rel=torch.as_tensor(rel[i]), tgt=torch.as_tensor(tgt[i]),
                ctrl=torch.as_tensor(ctrl)))
    return ex


def collate(batch, device):
    N = max(b["n"] for b in batch)
    B = len(batch)
    F_ = batch[0]["neu"].shape[1]
    R = batch[0]["rel"].shape[1]
    T = batch[0]["tgt"].shape[1]
    neu = torch.zeros(B, N, F_); rel = torch.zeros(B, N, R)
    tgt = torch.zeros(B, N, T); ctrl = torch.zeros(B, N, T)
    mask = torch.zeros(B, N, dtype=torch.bool)
    stim = torch.stack([b["stim"] for b in batch])
    for i, b in enumerate(batch):
        n = b["n"]
        neu[i, :n] = b["neu"]; rel[i, :n] = b["rel"]
        tgt[i, :n] = b["tgt"]; ctrl[i, :n] = b["ctrl"]
        mask[i, :n] = True
    return (neu.to(device), stim.to(device), rel.to(device), ctrl.to(device),
            tgt.to(device), mask.to(device))


def masked_loss(pred, tgt, mask, delta: float, normalise: bool = True,
                eps: float = 1e-3):
    """Loss that matches the metric.

    The score divides each recording's error by that recording's own effect energy,
    so a plain mean-squared error lets the model ignore recordings with small
    effects and then over-predict them badly at test time. Dividing each example by
    its own target energy makes the training objective proportional to one minus
    the score, which is what we are actually judged on.
    """
    m = mask[..., None].float()
    err = F.huber_loss(pred * m, tgt * m, delta=delta, reduction="none")
    per = err.sum(dim=(1, 2))
    if not normalise:
        return per.sum() / m.sum().clamp_min(1.0) / tgt.shape[-1]
    energy = ((tgt * m) ** 2).sum(dim=(1, 2)) + eps * m.sum(dim=(1, 2)) * tgt.shape[-1]
    return (per / energy.clamp_min(1e-6)).mean()
