"""Loader for the two larger ALM photoinhibition dandisets, 000010 and 000011.

Same laboratory, same delayed response task and the same perturbation as 000009, but
the files describe the light differently. There are no per trial photostimulation
columns. Instead the recording carries

  * ``photostim_start_time`` and ``photostim_stop_time`` event series, whose data
    field holds the laser power in mW,
  * one ``<site>_laser_power`` series per site that was ever used, each covering only
    the stretches of time when that site was actually driven,
  * ``sample``, ``delay`` and ``go`` event series giving the task epochs.

So a trial is perturbed if a photostimulation onset falls inside it, its dose is the
power recorded at that onset, and its site is whichever site's trace covers that
moment. Everything else follows the 000009 loader: align to delay onset, because that
is when the light comes on, and keep the animal's choice as the behavioural readout.

Only sites in ALM are kept. Some sessions also silence the pontine nuclei, which is a
different structure and would pollute a stereotype built by pooling over dose.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import h5py
import numpy as np

from .containers import AnimalTrials, Dataset
from .features import unit_features


def _dec(v):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


@dataclass
class AlmWideConfig:
    roots: tuple[str, ...] = ("data/raw/dandi000010", "data/raw/dandi000011")
    bin_s: float = 0.05
    pre_s: float = 1.00
    post_s: float = 2.50
    min_units: int = 5
    min_trials_per_cond: int = 8
    min_unperturbed: int = 30
    require_good: bool = True
    drop_early_lick: bool = True
    max_power_mw: float = 50.0
    site_match_s: float = 0.05     # tolerance when attributing an onset to a site
    keep_sites: tuple[str, ...] = ("ALM",)
    # Keep only light that arrives at the start of the delay, which is the epoch the
    # first cohort restricts to. About a ninth of onsets fall in another epoch and
    # would be mixed into an average that means something else.
    max_onset_offset_s: float = 0.4
    # Control and light trials must not already differ before the light arrives, or
    # the measured effect is partly a difference that was there anyway.
    max_pre_rate_mismatch: float = 1.25


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


def _event(f, name):
    g = f.get(f"acquisition/BehavioralEvents/{name}")
    if g is None or "timestamps" not in g:
        return None, None
    t = g["timestamps"][:]
    d = g["data"][:] if "data" in g else np.full(len(t), np.nan)
    return t, d


def _assign_to_trials(t_ev, t_start, t_stop):
    """Index of the trial containing each event, or -1."""
    out = np.full(len(t_ev), -1, dtype=np.int64)
    order = np.argsort(t_start)
    s, e = t_start[order], t_stop[order]
    j = np.searchsorted(s, t_ev, side="right") - 1
    ok = (j >= 0) & (j < len(s))
    good = ok.copy()
    good[ok] &= t_ev[ok] <= e[j[ok]]
    out[good] = order[j[good]]
    return out


def _site_of(f, sites, t_on, tol):
    """Which site's laser trace covers each onset.

    Each site's series only spans the periods when that site was driven, so
    membership in its timestamps is enough and no sample values need reading.
    """
    best = np.full(len(t_on), -1, dtype=np.int64)
    for si, name in enumerate(sites):
        ts = f[f"stimulus/presentation/{name}_laser_power/timestamps"]
        # timestamps are large; read only the neighbourhood of each onset
        n = ts.shape[0]
        head, tail = ts[0], ts[n - 1]
        cand = np.flatnonzero((best < 0) & (t_on >= head - tol) & (t_on <= tail + tol))
        if not len(cand):
            continue
        arr = ts[:]
        for i in cand:
            k = int(np.searchsorted(arr, t_on[i]))
            for kk in (k - 1, k):
                if 0 <= kk < n and abs(arr[kk] - t_on[i]) <= tol:
                    best[i] = si
                    break
    return best


def load_session(path: str, cfg: AlmWideConfig) -> AnimalTrials | None:
    subject = os.path.basename(os.path.dirname(path))
    # a few filenames do not carry a ``_ses-`` field, so build the session name from
    # whatever follows the subject and keep it unique within the subject
    stem = os.path.basename(path).rsplit(".", 1)[0]
    session = stem.replace(f"{subject}_", "").replace("_behavior+ecephys+ogen", "")
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
        # ``posy`` carries a large and arbitrary offset in some sessions, so only the
        # spread within a recording is meaningful. Referencing it to the shallowest
        # unit keeps that spread and throws the offset away.
        depth = np.asarray(u["posy"][:] if "posy" in u else np.zeros(len(sti)), float)
        depth = depth - np.nanmin(depth) if np.isfinite(depth).any() else depth
        ctype = _dec(u["cell_type"][:]) if "cell_type" in u else np.array(["?"] * len(sti))
        qual = _dec(u["quality"][:]) if "quality" in u else np.array(["good"] * len(sti))

        tr = f["intervals/trials"]
        t_start, t_stop = tr["start_time"][:], tr["stop_time"][:]
        instr = _dec(tr["trial_instruction"][:])
        outcome = _dec(tr["outcome"][:])
        early = _dec(tr["early_lick"][:]) if "early_lick" in tr else np.array(["no early"] * len(t_start))

        t_delay, _ = _event(f, "delay")
        t_go, _ = _event(f, "go")
        t_on, pw_on = _event(f, "photostim_start_time")
        if t_delay is None or t_on is None or not len(t_on):
            return None

        # one alignment time per trial: the delay onset
        d_tr = _assign_to_trials(t_delay, t_start, t_stop)
        align = np.full(len(t_start), np.nan)
        align[d_tr[d_tr >= 0]] = t_delay[d_tr >= 0]
        g_tr = _assign_to_trials(t_go, t_start, t_stop) if t_go is not None else None
        gotime = np.full(len(t_start), np.nan)
        if g_tr is not None:
            gotime[g_tr[g_tr >= 0]] = t_go[g_tr >= 0]

        # which trials were perturbed, at what dose and which site
        sites = sorted(k[: -len("_laser_power")] for k in f["stimulus/presentation"]
                       if k.endswith("_laser_power"))
        keep_site = [s for s in sites if any(w.lower() in s.lower() for w in cfg.keep_sites)]
        if not keep_site:
            return None
        s_of = _site_of(f, sites, t_on, cfg.site_match_s)
        p_tr = _assign_to_trials(t_on, t_start, t_stop)

        power = np.zeros(len(t_start))
        site = np.full(len(t_start), "", dtype=object)
        for i, ti in enumerate(p_tr):
            if ti < 0 or s_of[i] < 0:
                continue
            # only light delivered at the start of the delay
            if not (np.isfinite(align[ti])
                    and abs(t_on[i] - align[ti]) <= cfg.max_onset_offset_s):
                site[ti] = "__drop__"
                continue
            nm = sites[s_of[i]]
            if nm not in keep_site:
                site[ti] = "__drop__"
                continue
            p = float(pw_on[i]) if np.isfinite(pw_on[i]) else np.nan
            if not np.isfinite(p) or p <= 0 or p > cfg.max_power_mw:
                site[ti] = "__drop__"
                continue
            power[ti], site[ti] = p, nm

        ok = np.isfinite(align) & (site != "__drop__")
        if cfg.drop_early_lick:
            ok &= early == "no early"
        ok &= instr != "non-performing"
        if cfg.require_good:
            ok &= np.isin(outcome, ["hit", "miss", "ignore"])
        if ok.sum() < cfg.min_unperturbed:
            return None

        idx = np.where(ok)[0]
        starts = align[idx] - cfg.pre_s
        y = _bin_spikes(spikes, starts, n_bins, cfg.bin_s)
        pw, st = power[idx], site[idx]
        is_stim = (pw > 0) & (st != "")

        keys = [(round(float(p), 1), str(s)) for p, s in zip(pw, st)]
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
        present = sorted({int(c) for c in cond[keep] if c > 0})
        if not present:
            return None
        remap = {c: i + 1 for i, c in enumerate(present)}
        inv = {v: k for k, v in cmap.items()}

        y = y[keep]
        cond = np.array([remap.get(int(c), 0) for c in cond[keep]], dtype=np.int64)
        pert = cond > 0
        sub_instr, sub_out = instr[idx][keep], outcome[idx][keep]
        pw_k = pw[keep]

        # which side the animal chose: a hit means it licked the instructed side
        chose_right = np.where(sub_instr == "right", sub_out == "hit",
                               sub_out == "miss").astype(np.float32)
        licked = np.isin(sub_out, ["hit", "miss"])
        beh = np.zeros((len(y), n_bins, 2), dtype=np.float32)
        beh[:, :, 0] = chose_right[:, None]
        beh[:, :, 1] = (sub_out == "hit").astype(np.float32)[:, None]
        # a trial with no lick has no choice to report, but it is still a failure, so
        # only the choice channel is undefined there and the performance channel is not
        beh[~licked, :, 0] = np.nan

        # reject the recording if the two trial groups differ before the light
        pre = y[:, :n_pre].mean(axis=(1, 2))
        if pert.any() and (~pert).any():
            r = float(np.nanmean(pre[~pert])) / max(float(np.nanmean(pre[pert])), 1e-9)
            if not np.isfinite(r) or not (1.0 / cfg.max_pre_rate_mismatch <= r
                                          <= cfg.max_pre_rate_mismatch):
                return None

        raw = np.zeros((len(y), n_bins, 4), dtype=np.float32)
        on = np.zeros((len(y), n_bins), dtype=np.float32)
        dly = gotime[idx][keep] - align[idx][keep]
        delay_bins = int(np.clip(round(float(np.nanmedian(dly)) / cfg.bin_s)
                                 if np.isfinite(np.nanmedian(dly)) else 26, 4, n_post))
        for k in range(len(y)):
            if not pert[k]:
                continue
            raw[k, n_pre : n_pre + delay_bins, 0] = pw_k[k] / cfg.max_power_mw
            raw[k, n_pre : n_pre + delay_bins, 3] = 1.0
            on[k, n_pre : n_pre + delay_bins] = 1.0

        # ALM sits at the front of the brain; the recorded depth is what varies within
        # a session, and the site tells us which hemisphere the light went to
        feats = unit_features(y[~pert], depth_um=depth, cell_type=list(ctype),
                              max_depth_um=6000.0)
        cinfo = {remap[c]: inv[c] for c in present}
        side = {"left": -1.0, "right": 1.0, "bilateral": 0.0}
        return AnimalTrials(
            key=f"{subject}/{session}", animal=subject,
            y=y, u=None, interv_raw=raw, interv_on=on, behavior=beh,
            perturbed=pert, t0=n_pre, bin_s=cfg.bin_s, cond=cond,
            unit_features=feats,
            meta={
                "cond_info": cinfo,
                "cond_amp": {c: float(cinfo[c][0]) for c in cinfo},
                "cond_depth_um": {c: 0.0 for c in cinfo},
                "cond_galvo": {
                    c: (next((v for k, v in side.items() if k in cinfo[c][1].lower()), 0.0),
                        0.0)
                    for c in cinfo},
                "cond_site": {c: cinfo[c][1] for c in cinfo},
                "unit_y_um": np.asarray(depth, float).tolist(),
                "cell_type": list(ctype),
                "quality": list(qual),
                "session": session,
                "delay_bins": int(delay_bins),
            },
        )


def load_alm_wide(cfg: AlmWideConfig | None = None, verbose: bool = True) -> Dataset:
    cfg = cfg or AlmWideConfig()
    sets = []
    for root in cfg.roots:
        for p in sorted(glob.glob(os.path.join(root, "*", "*.nwb"))):
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
                print(f"  + {s.key:34s} units={s.n_obs:3d} trials={s.n_trials:4d} "
                      f"pert={int(s.perturbed.sum()):4d} "
                      f"conds={len(set(s.cond[s.perturbed]))}", flush=True)
    return Dataset(
        name="alm-photoinhibition-dandi000010-000011", sets=sets, n_u=0, n_raw=4,
        n_beh=2, bin_s=cfg.bin_s,
        interv_names=("power_mW", "unused", "unused", "on"),
        behavior_names=("chose_right", "correct"),
    )
