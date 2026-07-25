"""Loader for DANDI:001868 -- intracortical microstimulation (ICMS) of mouse
S1-trunk during an ICMS-cued wheel-turn detection task.

Six chronically implanted, task-trained mice (ICMS83/92/93/98/100/101), 55
sessions. The intervention is a fully parameterised electrical stimulus train
(amplitude in uA, 100 Hz, 70 biphasic pulses, 167 us pulse width) delivered on a
chosen contact of a 32-channel linear NET probe, so the intervention descriptor
is genuinely physical: **amplitude and cortical depth of the stimulating
contact**. That is what makes "an intervention never seen in training"
well-defined: a new amplitude, or a new depth, or both.

Readouts
--------
neural     spike counts of sorted single units (animal- and session-specific)
behaviour  wheel velocity (1 kHz encoder, decimated into the analysis bins) and
           the trial's detection outcome

Unperturbed data
----------------
Two sources, both free of stimulation:
  * *catch trials* -- trials the task delivered with 0 uA;
  * *inter-trial windows* -- randomly placed windows in the inter-trial interval
    that are guaranteed to be at least ``iti_guard_s`` from any stimulation.
Both are used to calibrate a held-out animal, and nothing else is.
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


def stable_seed(*parts) -> int:
    """Process-independent seed (``hash`` is salted per interpreter)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF

BEHAVIOUR_SUBJECTS = (
    "sub-ICMS83", "sub-ICMS92", "sub-ICMS93", "sub-ICMS98", "sub-ICMS100", "sub-ICMS101",
)


@dataclass
class IcmsConfig:
    root: str = "data/raw/dandi001868"
    bin_s: float = 0.025
    pre_s: float = 0.50            # window before stimulation onset
    post_s: float = 1.50           # window after stimulation onset
    iti_guard_s: float = 1.20      # keep unperturbed windows this far from any stim
    n_iti_windows: int = 240       # extra unperturbed pseudo-trials per session
    iti_pool_factor: int = 8       # candidates generated per kept window (for matching)
    # A session is dropped if, after covariate matching, its pre-stimulus
    # population rate still differs between stimulation trials and unperturbed
    # windows by more than this factor. Predicting from unperturbed initial
    # conditions is only valid when those distributions agree.
    max_pre_rate_mismatch: float = 1.25
    min_units: int = 8
    min_trials_per_cond: int = 12
    min_unperturbed: int = 30
    max_depth_um: float = 1900.0
    max_amp_ua: float = 12.0
    require_good_trial: bool = True
    behavior: bool = True
    seed: int = 0
    subjects: tuple[str, ...] = BEHAVIOUR_SUBJECTS


def _bin_spikes(spike_times, starts, n_bins, bin_s):
    """(n_units, n_windows, n_bins) -> returned as (n_windows, n_bins, n_units)."""
    n_u = len(spike_times)
    out = np.zeros((len(starts), n_bins, n_u), dtype=np.float32)
    edges_rel = np.arange(n_bins + 1) * bin_s
    for ui, st in enumerate(spike_times):
        if len(st) == 0:
            continue
        for wi, s0 in enumerate(starts):
            lo = np.searchsorted(st, s0)
            hi = np.searchsorted(st, s0 + n_bins * bin_s)
            if hi <= lo:
                continue
            rel = st[lo:hi] - s0
            cnt, _ = np.histogram(rel, bins=edges_rel)
            out[wi, :, ui] = cnt
    return out


def _wheel_velocity(data, rate, starts, n_bins, bin_s):
    """Mean absolute and signed wheel velocity in each analysis bin."""
    step = int(round(bin_s * rate))
    n_samp = n_bins * step
    out = np.zeros((len(starts), n_bins, 2), dtype=np.float32)
    for wi, s0 in enumerate(starts):
        i0 = int(round(s0 * rate))
        if i0 < 1 or i0 + n_samp + 1 > len(data):
            continue
        seg = data[i0 - 1 : i0 + n_samp + 1].astype(np.float64)
        vel = np.diff(seg) * rate                      # encoder units / s
        vel = vel[:n_samp].reshape(n_bins, step)
        out[wi, :, 0] = vel.mean(1)
        out[wi, :, 1] = np.abs(vel).mean(1)
    return out


def _match_pre_window(
    y_all: np.ndarray,
    wv_all: np.ndarray | None,
    n_pre: int,
    n_stim: int,
    cand_offset: int,
    n_want: int,
    rng: np.random.Generator,
    n_quantiles: int = 5,
) -> np.ndarray:
    """Choose inter-trial windows whose *pre-window* state matches stimulation trials.

    Matching is stratified on a joint quantile grid of two covariates measured
    strictly before stimulation onset: mean population firing rate and mean wheel
    speed. Candidates are drawn so that their joint distribution over the grid
    reproduces the stimulation trials' distribution, which removes the
    quiescence-criterion bias without ever looking at post-onset activity.

    Returns indices into the candidate block (0-based within that block).
    """
    n_cand = y_all.shape[0] - cand_offset
    if n_cand <= 0 or n_want <= 0:
        return np.zeros(0, dtype=int)

    def feats(sl):
        r = y_all[sl, :n_pre].mean(axis=(1, 2))
        if wv_all is None:
            return np.stack([r, np.zeros_like(r)], 1)
        s = wv_all[sl, :n_pre, 1].mean(1)
        return np.stack([r, s], 1)

    f_stim = feats(slice(0, n_stim))
    f_cand = feats(slice(cand_offset, None))

    # quantile edges from the stimulation trials define the strata
    def bins_of(x, edges):
        return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)

    keep_all = []
    codes_stim = np.zeros(len(f_stim), dtype=int)
    codes_cand = np.zeros(len(f_cand), dtype=int)
    mult = 1
    for j in range(f_stim.shape[1]):
        qs = np.quantile(f_stim[:, j], np.linspace(0, 1, n_quantiles + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        qs = np.unique(qs)
        if len(qs) < 3:
            continue
        codes_stim += mult * bins_of(f_stim[:, j], qs)
        codes_cand += mult * bins_of(f_cand[:, j], qs)
        mult *= len(qs) - 1

    cells, counts = np.unique(codes_stim, return_counts=True)
    target = counts / counts.sum()
    for cell, frac in zip(cells, target):
        avail = np.where(codes_cand == cell)[0]
        if len(avail) == 0:
            continue
        take = min(len(avail), max(1, int(round(frac * n_want))))
        keep_all.append(rng.choice(avail, size=take, replace=False))
    if not keep_all:
        return rng.choice(n_cand, size=min(n_want, n_cand), replace=False)
    keep = np.concatenate(keep_all)
    # top up from anywhere if strata were short, preferring unused candidates
    if len(keep) < n_want:
        rest = np.setdiff1d(np.arange(n_cand), keep)
        if len(rest):
            extra = rng.choice(rest, size=min(n_want - len(keep), len(rest)), replace=False)
            keep = np.concatenate([keep, extra])
    keep = np.sort(keep[:n_want])

    # Final greedy trim so the *mean* pre-window population rate matches the
    # stimulation trials to within `tol`. Stratification alone can leave a
    # residual offset when some strata are underpopulated.
    tol, max_drop = 0.02, int(0.3 * len(keep))
    target = float(f_stim[:, 0].mean())
    vals = f_cand[keep, 0]
    dropped = 0
    while dropped < max_drop and len(keep) > 8:
        cur = float(vals.mean())
        if abs(cur / (target + 1e-12) - 1.0) <= tol:
            break
        j = int(np.argmin(vals)) if cur < target else int(np.argmax(vals))
        keep = np.delete(keep, j)
        vals = np.delete(vals, j)
        dropped += 1
    return keep


def load_session(path: str, cfg: IcmsConfig) -> AnimalTrials | None:
    subject = os.path.basename(os.path.dirname(path))
    session = os.path.basename(path).split("_ses-")[1][:10]
    rng = np.random.default_rng(stable_seed(subject, session, cfg.seed))
    n_pre = int(round(cfg.pre_s / cfg.bin_s))
    n_post = int(round(cfg.post_s / cfg.bin_s))
    n_bins = n_pre + n_post

    with h5py.File(path, "r") as f:
        if "units" not in f or "intervals/trials" not in f:
            return None
        u = f["units"]
        sti = u["spike_times_index"][:]
        allspk = u["spike_times"][:]
        bounds = np.concatenate([[0], sti])
        accepted = u["accepted"][:] if "accepted" in u else np.ones(len(sti), bool)
        spikes = [np.sort(allspk[bounds[i] : bounds[i + 1]]) for i in range(len(sti))]
        keep_u = np.where(accepted)[0]
        if len(keep_u) < cfg.min_units:
            return None
        spikes = [spikes[i] for i in keep_u]
        unit_y = u["unit_y_um"][:][keep_u] if "unit_y_um" in u else np.zeros(len(keep_u))
        cell_type = (
            [x.decode() if isinstance(x, bytes) else x for x in u["cell_type"][:][keep_u]]
            if "cell_type" in u
            else ["?"] * len(keep_u)
        )

        tr = f["intervals/trials"]
        t_start = tr["start_time"][:]
        t_cur = tr["current_uA"][:]
        t_idx = tr["trial_index"][:]
        t_good = tr["is_good_trial"][:] if "is_good_trial" in tr else np.ones(len(t_start), bool)
        t_hit = tr["is_hit"][:] if "is_hit" in tr else np.zeros(len(t_start), bool)
        t_rt = tr["response_time"][:] if "response_time" in tr else np.full(len(t_start), np.nan)

        depths = f["general/extracellular_ephys/electrodes/rel_y"][:]

        # ---- stimulation events ----
        stim = {}
        if "intervals/electrical_stimulation" in f:
            s = f["intervals/electrical_stimulation"]
            for i, ti in enumerate(s["trial_index"][:]):
                stim[int(ti)] = {
                    "start": float(s["start_time"][i]),
                    "stop": float(s["stop_time"][i]),
                    "amp": float(s["current_uA"][i]),
                    "freq": float(s["frequency_hz"][i]),
                    "pc": int(s["pulse_count"][i]),
                    "pw": float(s["pulse_width_us"][i]),
                    "ch": int(s["stim_channel"][i]),
                }
        if not stim:
            return None

        # ---- behaviour ----
        wheel = None
        if cfg.behavior and "processing/behavior/wheel/wheel_position_processed" in f:
            g = f["processing/behavior/wheel/wheel_position_processed"]
            rate = float(g["starting_time"].attrs.get("rate", 1000.0))
            wheel = (g["data"][:], rate)

        # ---- build stimulation trials ----
        all_stim_starts = np.array(sorted(v["start"] for v in stim.values()))
        all_stim_stops = np.array([stim[k]["stop"] for k in sorted(stim)])
        rows = []
        for i in range(len(t_start)):
            ti = int(t_idx[i])
            if ti not in stim:
                continue
            if cfg.require_good_trial and not t_good[i]:
                continue
            ev = stim[ti]
            w0 = ev["start"] - cfg.pre_s
            w1 = ev["start"] + cfg.post_s
            # the window must contain exactly this stimulation train
            others = all_stim_starts[
                (all_stim_starts > w0 - 1e-9) & (all_stim_starts < w1 + 1e-9)
            ]
            if len(others) != 1:
                continue
            rows.append((w0, ev, bool(t_hit[i]), ti, float(t_rt[i])))
        if not rows:
            return None

        # ---- unperturbed: catch trials ----
        unp_starts = []
        for i in range(len(t_start)):
            ti = int(t_idx[i])
            if ti in stim or t_cur[i] != 0:
                continue
            w0 = t_start[i] - cfg.pre_s
            w1 = t_start[i] + cfg.post_s
            if np.any((all_stim_stops > w0 - cfg.iti_guard_s) & (all_stim_starts < w1 + cfg.iti_guard_s)):
                continue
            unp_starts.append(w0)

        # ---- unperturbed: candidate inter-trial windows ----
        # Stimulation trials are delivered by the task under a quiescence
        # criterion, so uniformly sampled inter-trial windows have a *different*
        # pre-window state distribution (up to 3x the population rate). Because
        # the prediction protocol draws initial conditions from unperturbed
        # trials, that mismatch would bias every measured effect. A pool of
        # candidates is therefore generated and then matched, below, to the
        # stimulation trials' pre-window covariates.
        n_catch = len(unp_starts)
        # Candidates are drawn from the gaps *between* consecutive task trials, so
        # they are interleaved in time with the stimulation trials and therefore
        # matched for slow drift in state, arousal and unit yield.
        win_len = cfg.pre_s + cfg.post_s
        srt = np.argsort(all_stim_starts)
        starts_s = all_stim_starts[srt]
        stops_s = np.sort(all_stim_stops)
        gaps = []
        for i in range(len(starts_s) - 1):
            lo = stops_s[i] + cfg.iti_guard_s
            hi = starts_s[i + 1] - cfg.iti_guard_s - win_len
            if hi > lo:
                gaps.append((lo, hi))
        pool: list[float] = []
        n_pool = cfg.iti_pool_factor * cfg.n_iti_windows
        if gaps:
            lens = np.array([h - lo for lo, h in gaps])
            p = lens / lens.sum()
            picks = rng.choice(len(gaps), size=n_pool, p=p)
            for gi in picks:
                lo, hi = gaps[gi]
                pool.append(float(rng.uniform(lo, hi)))

        # ---- bin everything (stim, catch, candidate pool) ----
        stim_w0 = np.array([r[0] for r in rows])
        catch_w0 = np.array(unp_starts, dtype=float)
        cand_w0 = np.array(pool, dtype=float)
        starts_all = np.concatenate([stim_w0, catch_w0, cand_w0])
        y_all = _bin_spikes(spikes, starts_all, n_bins, cfg.bin_s)
        wv_all = (
            _wheel_velocity(wheel[0], wheel[1], starts_all, n_bins, cfg.bin_s)
            if wheel is not None
            else None
        )
        n_stim = len(rows)
        n_catch_kept = len(catch_w0)

        # covariate matching on pre-window population rate and wheel speed
        keep_cand = _match_pre_window(
            y_all, wv_all, n_pre, n_stim, n_stim + n_catch_kept, cfg.n_iti_windows, rng
        )
        sel = np.concatenate(
            [
                np.arange(n_stim + n_catch_kept),
                (n_stim + n_catch_kept) + keep_cand,
            ]
        )
        starts = starts_all[sel]
        y = y_all[sel]
        wv_sel = None if wv_all is None else wv_all[sel]
        unp_starts = list(starts[n_stim:])
        if len(unp_starts) < cfg.min_unperturbed:
            return None
        n_unp_all = len(unp_starts)
        beh = None
        if wv_sel is not None:
            # normalise to a comparable scale across animals (encoder units differ
            # only by gain, so a robust per-session scale keeps behaviour in the
            # same physical units of "fraction of typical movement speed")
            scale = np.percentile(np.abs(wv_sel[:, :, 1]), 95) + 1e-6
            wv = wv_sel / scale
            # third channel: cumulative detection indicator. Condition-averaging
            # it gives the time-resolved probability that the animal has reported
            # detection by time t -- a low-variance behavioural response curve
            # with a strong dose dependence.
            det = np.zeros((n_stim + n_unp_all, n_bins, 1), dtype=np.float32)
            for k, (_, ev, hit, _, rt) in enumerate(rows):
                if not hit or not np.isfinite(rt):
                    continue
                b = n_pre + int(round(rt / cfg.bin_s))
                if b < n_bins:
                    det[k, max(b, 0) :, 0] = 1.0
            beh = np.concatenate([wv, det], axis=2).astype(np.float32)

        n_unp = len(unp_starts)
        n_tot = n_stim + n_unp
        raw = np.zeros((n_tot, n_bins, 4), dtype=np.float32)
        on = np.zeros((n_tot, n_bins), dtype=np.float32)
        perturbed = np.zeros(n_tot, dtype=bool)
        perturbed[:n_stim] = True
        cond_key = []
        hits = np.zeros(n_tot, dtype=np.float32)
        for k, (w0, ev, hit, ti, _rt) in enumerate(rows):
            a = ev["amp"] / cfg.max_amp_ua
            dep = depths[ev["ch"]] / cfg.max_depth_um if ev["ch"] < len(depths) else 0.0
            b0 = int(round((ev["start"] - w0) / cfg.bin_s))
            b1 = int(round((ev["stop"] - w0) / cfg.bin_s))
            b0 = max(0, min(b0, n_bins)); b1 = max(b0 + 1, min(b1, n_bins))
            raw[k, b0:b1, 0] = a
            raw[k, b0:b1, 1] = dep
            raw[k, b0:b1, 2] = ev["pw"] / 400.0
            raw[k, b0:b1, 3] = ev["pc"] / 100.0
            on[k, b0:b1] = 1.0
            cond_key.append((round(ev["amp"], 3), int(ev["ch"])))
            hits[k] = float(hit)

        uniq = sorted(set(cond_key))
        cmap = {c: i + 1 for i, c in enumerate(uniq)}
        cond = np.zeros(n_tot, dtype=np.int64)
        for k, c in enumerate(cond_key):
            cond[k] = cmap[c]

        # drop conditions with too few trials
        keep = np.ones(n_tot, dtype=bool)
        for c, ci in cmap.items():
            m = cond == ci
            if m.sum() < cfg.min_trials_per_cond:
                keep &= ~m
        if keep.sum() < n_unp + cfg.min_trials_per_cond:
            return None
        y, raw, on, perturbed, cond, hits = (
            y[keep], raw[keep], on[keep], perturbed[keep], cond[keep], hits[keep]
        )
        beh = None if beh is None else beh[keep]
        # renumber conditions contiguously
        present = sorted({int(c) for c in cond[perturbed]})
        remap = {c: i + 1 for i, c in enumerate(present)}
        cond = np.array([remap.get(int(c), 0) for c in cond], dtype=np.int64)
        inv = {v: k for k, v in cmap.items()}
        cond_info = {remap[c]: inv[c] for c in present}
        if len(present) < 2:
            return None

        # final leakage/matching gate
        pre_p = float(y[perturbed, :n_pre].mean())
        pre_u = float(y[~perturbed, :n_pre].mean())
        ratio = pre_p / (pre_u + 1e-12)
        lim = cfg.max_pre_rate_mismatch
        if not (1.0 / lim < ratio < lim):
            return None

        feats = unit_features(
            y[~perturbed], depth_um=unit_y, cell_type=cell_type, max_depth_um=cfg.max_depth_um
        )
        return AnimalTrials(
            unit_features=feats,
            key=f"{subject}/{session}",
            animal=subject,
            y=y,
            u=None,
            interv_raw=raw,
            interv_on=on,
            behavior=beh,
            perturbed=perturbed,
            t0=n_pre,
            bin_s=cfg.bin_s,
            cond=cond,
            meta={
                "cond_info": cond_info,
                "cond_amp": {int(ci): float(cond_info[ci][0]) for ci in cond_info},
                "cond_channel": {int(ci): int(cond_info[ci][1]) for ci in cond_info},
                "cond_depth_um": {
                    int(ci): float(depths[cond_info[ci][1]])
                    if cond_info[ci][1] < len(depths)
                    else float("nan")
                    for ci in cond_info
                },
                "unit_y_um": unit_y.tolist(),
                "cell_type": cell_type,
                "hits": hits.tolist(),
                "session": session,
                "n_catch": int(n_catch),
                "n_iti": int(len(unp_starts) - n_catch),
                "electrode_depths_um": depths.tolist(),
            },
        )


def load_icms(cfg: IcmsConfig | None = None, verbose: bool = True) -> Dataset:
    cfg = cfg or IcmsConfig()
    files = []
    for sub in cfg.subjects:
        files += sorted(glob.glob(os.path.join(cfg.root, sub, "*.nwb")))
    sets = []
    for p in files:
        try:
            s = load_session(p, cfg)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"  ! {os.path.basename(p)}: {exc!r}", flush=True)
            continue
        if s is None:
            if verbose:
                print(f"  - skipped {os.path.basename(p)}", flush=True)
            continue
        sets.append(s)
        if verbose:
            print(
                f"  + {s.key:30s} units={s.n_obs:3d} trials={s.n_trials:5d} "
                f"pert={int(s.perturbed.sum()):5d} conds={len(set(s.cond[s.perturbed]))}",
                flush=True,
            )
    return Dataset(
        name="icms-s1-dandi001868",
        sets=sets,
        n_u=0,
        n_raw=4,
        n_beh=3 if cfg.behavior else 0,
        bin_s=cfg.bin_s,
        interv_names=("amplitude_uA", "depth_um", "pulse_width_us", "pulse_count"),
        behavior_names=("wheel_velocity", "wheel_speed", "detection_prob"),
    )
