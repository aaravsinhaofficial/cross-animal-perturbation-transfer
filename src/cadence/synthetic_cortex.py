"""A synthetic cortex with a stimulating electrode, built to test the explanation
we give for the real data.

The real finding is that a shared causal rule predicts the shape of every neuron's
response but not its amplitude, and that the amplitude is unpredictable because
low-current stimulation grabs a sparse, scattered set of cells that depends on where
that particular electrode landed. This simulator lets us switch that mechanism on and
off, and vary how many neurons we record, so we can see whether it really produces
the pattern we measured.

The model is a linear-threshold network on a one-dimensional cortical depth axis:

    x_{t+1} = A x_t + a * r * on(t) + noise
    A       = (1 - leak) I + W,   W_ij = k(|z_i - z_j|) shared + animal-specific part
    y_t     = Poisson(softplus(gain * x_t + b))    for a random subset of neurons

Two recruitment modes decide who the electrode actually drives.

``local``   r_i is a smooth function of the distance from the contact. Every animal
            has the same rule, so which neurons respond is predictable from depth.
``sparse``  r_i is a sparse random draw, wide in depth, redrawn for every animal and
            contact. The rule for *how much total drive* is injected is still shared,
            but which cells receive it is private to that implant.

Knobs worth sweeping: ``n_obs`` (how many neurons are recorded), ``recruit`` (the mode
above), ``sparsity`` and ``animal_het`` (how different animals' circuits are).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.containers import AnimalTrials, Dataset
from .data.features import unit_features

MAX_D = 1900.0


@dataclass
class CortexConfig:
    n_neurons: int = 200
    n_obs: int = 24                       # simultaneously recorded neurons
    n_animals: int = 8
    depths_um: tuple[float, float] = (0.0, MAX_D)
    amplitudes: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)
    contacts_um: tuple[float, ...] = (300.0, 900.0, 1500.0)
    trials_per_cond: int = 60
    unperturbed_trials: int = 320
    T: int = 80
    t0: int = 20
    stim_bins: int = 28                   # 0.7 s at 25 ms
    bin_s: float = 0.025

    leak: float = 0.35
    w_scale: float = 0.55
    w_sigma_um: float = 260.0
    animal_het: float = 0.25              # size of the animal-specific circuit part
    recruit: str = "sparse"               # 'local' | 'sparse'
    recruit_sigma_um: float = 220.0       # width for 'local'
    sparse_sigma_um: float = 900.0        # width for 'sparse' (deliberately wide)
    sparsity: float = 0.12                # fraction of neurons the electrode drives
    drive_scale: float = 0.9
    saturation: float = 6.0               # amplitude at which the drive saturates
    obs_gain: float = 1.1
    obs_bias: float = -0.15
    rate_scale: float = 2.2
    noise: float = 0.25
    animal_gain_cv: float = 0.25          # between-animal responsiveness spread
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _kernel(d1, d2, sigma):
    return np.exp(-((d1[:, None] - d2[None, :]) / sigma) ** 2)


def _build_animal(cfg: CortexConfig, rng: np.random.Generator):
    z = np.sort(rng.uniform(*cfg.depths_um, size=cfg.n_neurons))
    shared_rng = np.random.default_rng(12345)          # same circuit rule for all
    K = _kernel(z, z, cfg.w_sigma_um)
    sign = np.where(shared_rng.random(cfg.n_neurons) < 0.8, 1.0, -1.6)
    W = cfg.w_scale * (K * sign[None, :]) / cfg.n_neurons ** 0.5
    W += cfg.animal_het * rng.normal(0, 1, W.shape) / cfg.n_neurons ** 0.5
    A = (1.0 - cfg.leak) * np.eye(cfg.n_neurons) + W
    ev = np.max(np.abs(np.linalg.eigvals(A)))
    if ev > 0.97:
        A *= 0.97 / ev
    return z, A


def _recruitment(cfg: CortexConfig, z, contact, rng: np.random.Generator):
    if cfg.recruit == "local":
        r = np.exp(-((z - contact) / cfg.recruit_sigma_um) ** 2)
    elif cfg.recruit == "sparse":
        env = np.exp(-((z - contact) / cfg.sparse_sigma_um) ** 2)
        hit = rng.random(len(z)) < cfg.sparsity
        r = hit * env * rng.gamma(2.0, 0.5, len(z))
    else:
        raise ValueError(cfg.recruit)
    n = np.linalg.norm(r)
    return r / n * np.sqrt(len(z)) if n > 0 else r


def build_cortex_dataset(cfg: CortexConfig) -> Dataset:
    sets: list[AnimalTrials] = []
    for i in range(cfg.n_animals):
        rng = np.random.default_rng(cfg.seed * 1009 + i)
        z, A = _build_animal(cfg, rng)
        # animal-specific overall responsiveness, correlated with its own dynamics
        g_animal = float(np.exp(rng.normal(0, cfg.animal_gain_cv)))
        sel = np.sort(rng.choice(cfg.n_neurons, size=min(cfg.n_obs, cfg.n_neurons),
                                 replace=False))
        gains = cfg.obs_gain * np.exp(rng.normal(0, 0.3, len(sel)))
        bias = rng.normal(cfg.obs_bias, 0.1, len(sel))
        recruit = {c: _recruitment(cfg, z, c, rng) for c in cfg.contacts_um}

        conds = [(0.0, None)] + [(a, c) for c in cfg.contacts_um for a in cfg.amplitudes]
        ys, raws, ons, perts, cids, behs = [], [], [], [], [], []
        cond_amp, cond_dep = {}, {}
        cid = 0
        for a, contact in conds:
            n_tr = cfg.unperturbed_trials if contact is None else cfg.trials_per_cond
            drive = np.zeros((cfg.T, cfg.n_neurons))
            if contact is not None:
                # saturating dose, shared across animals; scaled by this animal's gain
                dose = cfg.drive_scale * g_animal * a / (1.0 + a / cfg.saturation)
                drive[cfg.t0 : cfg.t0 + cfg.stim_bins] = dose * recruit[contact]
            x = np.zeros((n_tr, cfg.n_neurons))
            X = np.zeros((n_tr, cfg.T, cfg.n_neurons))
            for t in range(cfg.T):
                x = x @ A.T + drive[t] + cfg.noise * rng.normal(0, 1, x.shape)
                X[:, t] = x
            r = np.log1p(np.exp(X[:, :, sel] * gains + bias))
            y = rng.poisson(np.clip(r * cfg.rate_scale, 1e-4, None)).astype(np.float32)
            ys.append(y)
            raw = np.zeros((n_tr, cfg.T, 4), dtype=np.float32)
            on = np.zeros((n_tr, cfg.T), dtype=np.float32)
            if contact is not None:
                raw[:, cfg.t0 : cfg.t0 + cfg.stim_bins, 0] = a / 12.0
                raw[:, cfg.t0 : cfg.t0 + cfg.stim_bins, 1] = contact / MAX_D
                on[:, cfg.t0 : cfg.t0 + cfg.stim_bins] = 1.0
            raws.append(raw); ons.append(on)
            perts.append(np.full(n_tr, contact is not None, dtype=bool))
            if contact is None:
                cids.append(np.zeros(n_tr, dtype=np.int64))
            else:
                cid += 1
                cids.append(np.full(n_tr, cid, dtype=np.int64))
                cond_amp[cid] = float(a)
                cond_dep[cid] = float(contact)
            # behaviour: a shared saturating readout of total drive received
            beh = np.zeros((n_tr, cfg.T, 1), dtype=np.float32)
            if contact is not None:
                p = 1.0 / (1.0 + np.exp(-(1.6 * a / cfg.saturation - 1.0)))
                hit = rng.random(n_tr) < p
                onset = cfg.t0 + rng.integers(3, 12, n_tr)
                for k in range(n_tr):
                    if hit[k] and onset[k] < cfg.T:
                        beh[k, onset[k] :, 0] = 1.0
            behs.append(beh)

        y = np.concatenate(ys)
        pert = np.concatenate(perts)
        feats = unit_features(y[~pert], depth_um=z[sel], cell_type=None)
        sets.append(AnimalTrials(
            key=f"cx{i:02d}/s0", animal=f"cx{i:02d}",
            y=y, u=None,
            interv_raw=np.concatenate(raws), interv_on=np.concatenate(ons),
            behavior=np.concatenate(behs), perturbed=pert,
            t0=cfg.t0, bin_s=cfg.bin_s, cond=np.concatenate(cids),
            unit_features=feats,
            meta={"cond_amp": cond_amp, "cond_depth_um": cond_dep,
                  "unit_y_um": z[sel].tolist(),
                  "cell_type": ["pyramidal"] * len(sel),
                  "recruit": cfg.recruit, "n_obs": len(sel),
                  "animal_gain": g_animal},
        ))
    return Dataset(name=f"cortex-{cfg.recruit}-n{cfg.n_obs}", sets=sets,
                   n_u=0, n_raw=4, n_beh=1, bin_s=cfg.bin_s,
                   interv_names=("amplitude", "depth", "", ""),
                   behavior_names=("detection_prob",))
