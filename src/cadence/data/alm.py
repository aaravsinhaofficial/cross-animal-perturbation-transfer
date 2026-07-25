"""Loader for DANDI:000009, optogenetic photoinhibition of mouse frontal cortex.

Twenty mice, silicon-probe recordings in anterior lateral motor cortex during a
delayed-response task, with light delivered on a subset of trials through a
scanning galvo at a named power and location. The trials table carries the
stimulus parameters explicitly, so a condition is defined by the power and the
galvo coordinates, and an unseen intervention is a power or a location that never
appears in training.

Why this dataset matters here. Microstimulation drives a sparse scattered set of
cells that belongs to the implant. Photoinhibition of an inhibitory-opsin line
drives a dense local volume, so which cells are affected follows a rule that is
the same in every animal. That is the contrast the simulated cortex predicted
would matter, and this is the dataset that tests it.

Alignment is on the start of the delay epoch, which is when the light comes on.
Unperturbed trials are the ones the task delivered with the light off.
"""

from __future__ import annotations

import glob
import os
import zlib
from dataclasses import dataclass

import h5py
import numpy as np

from .containers import AnimalTrials, Dataset
from .features import unit_features


def _dec(v):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v])


def _num(v, default=np.nan):
    out = np.full(len(v), default, dtype=float)
    for i, x in enumerate(v):
        try:
            out[i] = float(x)
        except (TypeError, ValueError):
            pass
    return out


@dataclass
class AlmConfig:
    root: str = "data/raw/dandi000009"
    bin_s: float = 0.05
    pre_s: float = 1.00            # before delay onset
    post_s: float = 2.50           # covers the delay and the start of the response
    min_units: int = 5
    min_trials_per_cond: int = 8
    min_unperturbed: int = 30
    require_good: bool = True
    drop_early_lick: bool = True
    max_power_mw: float = 50.0
    seed: int = 0


def _bin_spikes(spikes, starts, n_bins, bin_s):
    out = np.zeros((len(starts), n_bins, len(spikes)), dtype=np.float32)
    edges = np.arange(n_bins + 1) * bin_s
    for ui, st in enumerate(spikes):
        if len(st) == 0:
            continue
        for wi, s0 in enumerate(starts):
            lo = np.searchsorted(st, s0)
            hi = np.searchsorted(st, s0 + n_bins * bin_s)
            if hi > lo:
                out[wi, :, ui] = np.histogram(st[lo:hi] - s0, bins=edges)[0]
    return out


def load_session(path: str, cfg: AlmConfig) -> AnimalTrials | None:
    subject = os.path.basename(os.path.dirname(path))
    session = os.path.basename(path).split("_ses-")[1][:15]
    n_pre = int(round(cfg.pre_s / cfg.bin_s))
    n_post = int(round(cfg.post_s / cfg.bin_s))
    n_bins = n_pre + n_post

    with h5py.File(path, "r") as f:
        if "units" not in f or "intervals/trials" not in f:
            return None
        u = f["units"]
        sti = u["spike_times_index"][:]
        if len(sti) < cfg.min_units:
            return None
        allspk = u["spike_times"][:]
        bounds = np.concatenate([[0], sti])
        spikes = [np.sort(allspk[bounds[i] : bounds[i + 1]]) for i in range(len(sti))]
        depth = u["unit_y"][:] if "unit_y" in u else np.zeros(len(sti))
        ctype = _dec(u["cell_type"][:]) if "cell_type" in u else np.array(["?"] * len(sti))

        tr = f["intervals/trials"]
        need = ("start_time", "pole_out_time", "cue_start_time", "photo_stim_power",
                "photo_stim_period", "stim_present", "response", "type")
        if any(k not in tr for k in need):
            return None
        t_start = _num(tr["start_time"][:])
        t_delay = _num(tr["pole_out_time"][:])
        t_go = _num(tr["cue_start_time"][:])
        power = _num(tr["photo_stim_power"][:], 0.0)
        period = _dec(tr["photo_stim_period"][:])
        present = _num(tr["stim_present"][:], 0.0)
        resp = _dec(tr["response"][:])
        ttype = _dec(tr["type"][:])
        good = _num(tr["is_good"][:], 1.0) if "is_good" in tr else np.ones(len(t_start))
        gx = _num(tr["photo_loc_galvo_x"][:], 0.0) if "photo_loc_galvo_x" in tr else np.zeros(len(t_start))
        gy = _num(tr["photo_loc_galvo_y"][:], 0.0) if "photo_loc_galvo_y" in tr else np.zeros(len(t_start))

        # the light comes on at the start of the delay, so align there
        align = np.where(np.isfinite(t_delay), t_delay, t_start)
        ok = np.isfinite(align) & np.isfinite(t_start)
        if cfg.require_good:
            ok &= good > 0
        if cfg.drop_early_lick:
            ok &= resp != "early lick"
        ok &= ttype != "non-performing"
        if ok.sum() < cfg.min_unperturbed:
            return None

        idx = np.where(ok)[0]
        starts = align[idx] - cfg.pre_s
        y = _bin_spikes(spikes, starts, n_bins, cfg.bin_s)

        is_stim = (present[idx] > 0) & (power[idx] > 0) & (period[idx] == "delay")
        pw = power[idx]
        gxs, gys = gx[idx], gy[idx]

        # a condition is a (power, galvo location) setting
        keys = [
            (round(float(p), 2), round(float(a), 2), round(float(b), 2))
            for p, a, b in zip(pw, gxs, gys)
        ]
        uniq = sorted({k for k, s in zip(keys, is_stim) if s})
        cmap = {k: i + 1 for i, k in enumerate(uniq)}
        cond = np.array([cmap.get(k, 0) if s else 0 for k, s in zip(keys, is_stim)],
                        dtype=np.int64)

        keep = np.ones(len(idx), dtype=bool)
        for k, ci in cmap.items():
            if (cond == ci).sum() < cfg.min_trials_per_cond:
                keep &= cond != ci
        if (~is_stim[keep]).sum() < cfg.min_unperturbed:
            return None
        present_conds = sorted({int(c) for c in cond[keep] if c > 0})
        if len(present_conds) < 1:
            return None
        remap = {c: i + 1 for i, c in enumerate(present_conds)}
        inv = {v: k for k, v in cmap.items()}

        y = y[keep]
        cond = np.array([remap.get(int(c), 0) for c in cond[keep]], dtype=np.int64)
        pert = cond > 0
        sub_resp, sub_type = resp[idx][keep], ttype[idx][keep]
        pw_k, gx_k, gy_k = pw[keep], gxs[keep], gys[keep]

        # behavioural readout: which side the animal actually chose
        chose_right = np.where(
            sub_type == "lick right", sub_resp == "correct", sub_resp == "incorrect"
        ).astype(np.float32)
        valid_choice = np.isin(sub_resp, ["correct", "incorrect"])
        beh = np.zeros((len(y), n_bins, 2), dtype=np.float32)
        # held flat in time: the choice is reported after the go cue, so the
        # informative quantity is the trial-level probability
        beh[:, :, 0] = chose_right[:, None]
        beh[:, :, 1] = (sub_resp == "correct").astype(np.float32)[:, None]
        beh[~valid_choice] = np.nan

        n_tr = len(y)
        raw = np.zeros((n_tr, n_bins, 4), dtype=np.float32)
        on = np.zeros((n_tr, n_bins), dtype=np.float32)
        # the light is on for the delay epoch, from alignment to the go cue
        delay_bins = int(round(np.nanmedian(t_go[idx][keep] - align[idx][keep]) / cfg.bin_s))
        delay_bins = int(np.clip(delay_bins, 4, n_post))
        for k in range(n_tr):
            if not pert[k]:
                continue
            raw[k, n_pre : n_pre + delay_bins, 0] = pw_k[k] / cfg.max_power_mw
            raw[k, n_pre : n_pre + delay_bins, 1] = gx_k[k]
            raw[k, n_pre : n_pre + delay_bins, 2] = gy_k[k]
            raw[k, n_pre : n_pre + delay_bins, 3] = 1.0
            on[k, n_pre : n_pre + delay_bins] = 1.0

        feats = unit_features(y[~pert], depth_um=depth * 1000.0, cell_type=list(ctype),
                              max_depth_um=6000.0)
        cond_info = {remap[c]: inv[c] for c in present_conds}
        return AnimalTrials(
            key=f"{subject}/{session}", animal=subject,
            y=y, u=None, interv_raw=raw, interv_on=on, behavior=beh,
            perturbed=pert, t0=n_pre, bin_s=cfg.bin_s, cond=cond,
            unit_features=feats,
            meta={
                "cond_info": cond_info,
                "cond_amp": {c: float(cond_info[c][0]) for c in cond_info},
                "cond_depth_um": {c: float(cond_info[c][2]) * 1000.0 for c in cond_info},
                "cond_galvo": {c: (cond_info[c][1], cond_info[c][2]) for c in cond_info},
                "unit_y_um": (depth * 1000.0).tolist(),
                "cell_type": list(ctype),
                "session": session,
                "delay_bins": int(delay_bins),
            },
        )


def load_alm(cfg: AlmConfig | None = None, verbose: bool = True) -> Dataset:
    cfg = cfg or AlmConfig()
    sets = []
    for p in sorted(glob.glob(os.path.join(cfg.root, "*", "*.nwb"))):
        try:
            s = load_session(p, cfg)
        except Exception as exc:
            if verbose:
                print(f"  ! {os.path.basename(p)}: {exc!r}"[:110], flush=True)
            continue
        if s is None:
            continue
        sets.append(s)
        if verbose:
            print(f"  + {s.key:38s} units={s.n_obs:3d} trials={s.n_trials:4d} "
                  f"pert={int(s.perturbed.sum()):4d} conds={len(set(s.cond[s.perturbed]))}",
                  flush=True)
    return Dataset(
        name="alm-photoinhibition-dandi000009", sets=sets, n_u=0, n_raw=4,
        n_beh=2, bin_s=cfg.bin_s,
        interv_names=("power_mW", "galvo_x", "galvo_y", "on"),
        behavior_names=("chose_right", "correct"),
    )


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF
