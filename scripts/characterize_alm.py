"""Characterise DANDI:000011 (Li, Daie, Svoboda & Druckmann 2016, Nature 17643):
ALM extracellular recordings with optogenetic photoinhibition during an
audio/tactile delayed-response task.

Prints, per animal: sessions, units, trials, photostim epochs, sites, powers and
the epoch (sample / delay / response) in which stimulation was delivered.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


def dec(x):
    return x.decode() if isinstance(x, bytes) else x


def photostim_epochs(f, site_key: str, thresh_frac: float = 0.15):
    """Recover (start, stop, mean_power) epochs from the continuous laser trace."""
    g = f["stimulus/presentation"][site_key]
    p = np.asarray(g["data"][:], dtype=np.float64)
    ts = np.asarray(g["timestamps"][:], dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    if p.size == 0:
        return []
    hi = np.nanmax(p)
    if hi <= 0:
        return []
    on = p > thresh_frac * hi
    if not on.any():
        return []
    d = np.diff(on.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    stops = np.where(d == -1)[0] + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        stops = np.r_[stops, len(on) - 1]
    n = min(len(starts), len(stops))
    out = []
    # merge epochs separated by < 30 ms (the 40 Hz sinusoid / pulse train dips)
    merged = []
    for s, e in zip(starts[:n], stops[:n]):
        if merged and ts[s] - merged[-1][1] < 0.030:
            merged[-1][1] = ts[e]
            merged[-1][2].append((s, e))
        else:
            merged.append([ts[s], ts[e], [(s, e)]])
    for s_t, e_t, segs in merged:
        pw = np.concatenate([p[a : b + 1] for a, b in segs])
        out.append((float(s_t), float(e_t), float(np.mean(pw)), float(np.max(pw))))
    return out


def describe(path: Path) -> dict:
    r: dict = {"path": path.name, "subject": path.parent.name}
    with h5py.File(path, "r") as f:
        r["genotype"] = dec(f["general/subject/genotype"][()]) if "general/subject/genotype" in f else None
        t = f["intervals/trials"]
        r["n_trials"] = int(len(t["id"]))
        r["trial_cols"] = sorted(t.keys())
        for col in ("outcome", "trial_instruction", "early_lick", "task"):
            if col in t:
                r[f"{col}_counts"] = dict(Counter(dec(x) for x in t[col][:]))
        if "task_protocol" in t:
            r["task_protocol"] = sorted({int(x) for x in t["task_protocol"][:]})
        u = f["units"]
        r["n_units"] = int(len(u["id"]))
        if "quality" in u:
            r["unit_quality"] = dict(Counter(dec(x) for x in u["quality"][:]))
        if "cell_type" in u:
            r["cell_type"] = dict(Counter(dec(x) for x in u["cell_type"][:]))
        st = u["spike_times"][:]
        r["n_spikes"] = int(len(st))
        r["session_dur_s"] = float(st.max() - st.min()) if len(st) else None
        if "general/extracellular_ephys/electrodes/location" in f:
            r["electrode_locations"] = sorted(
                {dec(x) for x in f["general/extracellular_ephys/electrodes/location"][:]}
            )
        # behavioural events
        ev = {}
        if "acquisition/BehavioralEvents" in f:
            for k in f["acquisition/BehavioralEvents"].keys():
                ev[k] = int(len(f["acquisition/BehavioralEvents"][k]["data"]))
        r["behavioral_events"] = ev
        if "acquisition/BehavioralTimeSeries" in f:
            bts = {}
            for k in f["acquisition/BehavioralTimeSeries"].keys():
                g = f["acquisition/BehavioralTimeSeries"][k]
                bts[k] = int(g["data"].shape[0])
            r["behavior_timeseries"] = bts
        # optogenetics sites
        r["ogen_sites"] = sorted(f["general/optogenetics"].keys()) if "general/optogenetics" in f else []
        sp = sorted(f["stimulus/presentation"].keys()) if "stimulus/presentation" in f else []
        r["stim_presentation"] = sp
        # photostim epochs from laser power traces
        eps = {}
        for key in sp:
            if not key.endswith("_laser_power"):
                continue
            try:
                e = photostim_epochs(f, key)
            except Exception as exc:
                e = []
                r.setdefault("errors", []).append(f"{key}: {exc!r}")
            eps[key] = e
        r["n_photostim_epochs"] = {k: len(v) for k, v in eps.items()}
        allep = [e for v in eps.values() for e in v]
        if allep:
            durs = np.array([e[1] - e[0] for e in allep])
            pws = np.array([e[3] for e in allep])
            r["photostim_dur_s"] = [float(durs.min()), float(np.median(durs)), float(durs.max())]
            r["photostim_peak_power_mW"] = [float(pws.min()), float(np.median(pws)), float(pws.max())]
            # align to trials: which epoch of the trial?
            tstart = t["start_time"][:]
            ev_names = {}
            for nm in ("sample", "delay", "go"):
                key = f"acquisition/BehavioralEvents/{nm}"
                if key in f:
                    ev_names[nm] = f[key]["timestamps"][:]
            aligned = Counter()
            rel = []
            for s_t, e_t, _, _ in allep:
                idx = np.searchsorted(tstart, s_t) - 1
                if idx < 0 or idx >= len(tstart):
                    aligned["no-trial"] += 1
                    continue
                rel.append(s_t - tstart[idx])
                lab = "other"
                if "go" in ev_names and idx < len(ev_names["go"]) and s_t >= ev_names["go"][idx]:
                    lab = "response"
                elif "delay" in ev_names and idx < len(ev_names["delay"]) and s_t >= ev_names["delay"][idx]:
                    lab = "delay"
                elif "sample" in ev_names and idx < len(ev_names["sample"]) and s_t >= ev_names["sample"][idx]:
                    lab = "sample"
                aligned[lab] += 1
            r["photostim_trial_epoch"] = dict(aligned)
            if rel:
                r["photostim_rel_to_trial_start_s"] = [
                    float(np.min(rel)), float(np.median(rel)), float(np.max(rel))
                ]
            r["n_photostim_total"] = len(allep)
        else:
            r["n_photostim_total"] = 0
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/dandi000011", type=Path)
    ap.add_argument("--out", default="results/tables/dandi000011_characterization.json", type=Path)
    args = ap.parse_args()
    files = sorted(args.root.rglob("*.nwb"))
    recs = []
    for i, p in enumerate(files):
        try:
            recs.append(describe(p))
        except Exception as exc:
            recs.append({"path": p.name, "subject": p.parent.name, "error": repr(exc)})
        print(f"[{i+1}/{len(files)}] {p.name}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recs, indent=1, default=str))

    by = defaultdict(list)
    for r in recs:
        by[r["subject"]].append(r)
    hdr = (
        f"{'subject':16s} {'ses':>4s} {'trials':>7s} {'units':>6s} {'stimEp':>7s} "
        f"{'sites':>34s} {'peakP_mW':>18s} {'epochs':>34s}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    tot_ses = tot_tr = tot_u = tot_ep = 0
    for sub in sorted(by):
        rs = by[sub]
        ntr = sum(r.get("n_trials", 0) for r in rs)
        nu = sum(r.get("n_units", 0) for r in rs)
        nep = sum(r.get("n_photostim_total", 0) for r in rs)
        sites = sorted({s for r in rs for s in (r.get("ogen_sites") or [])})
        pw = [r["photostim_peak_power_mW"] for r in rs if r.get("photostim_peak_power_mW")]
        pwr = (
            f"[{min(p[0] for p in pw):.2f},{max(p[2] for p in pw):.2f}]" if pw else "-"
        )
        ep = Counter()
        for r in rs:
            ep.update(r.get("photostim_trial_epoch") or {})
        print(
            f"{sub:16s} {len(rs):4d} {ntr:7d} {nu:6d} {nep:7d} "
            f"{str(sites)[:34]:>34s} {pwr:>18s} {str(dict(ep))[:34]:>34s}"
        )
        tot_ses += len(rs); tot_tr += ntr; tot_u += nu; tot_ep += nep
    print("-" * len(hdr))
    print(f"{'TOTAL':16s} {tot_ses:4d} {tot_tr:7d} {tot_u:6d} {tot_ep:7d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
