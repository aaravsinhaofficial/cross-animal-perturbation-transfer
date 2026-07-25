"""Characterise DANDI:001868: what interventions exist, in which animals, with
which readouts. This drives the experimental design, so it prints everything
that could constrain the cross-animal / cross-intervention splits.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def scalar(f, key, default=None):
    try:
        v = f[key][()]
    except Exception:
        return default
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype == object:
        return [x.decode() if isinstance(x, bytes) else x for x in v.tolist()]
    return v


def rate_of(f, group):
    g = f[group]
    if "rate" in g.attrs:
        return float(g.attrs["rate"])
    if "starting_time" in g and "rate" in g["starting_time"].attrs:
        return float(g["starting_time"].attrs["rate"])
    return None


def describe(path: Path) -> dict:
    out: dict = {"path": str(path.name), "subject": path.parent.name}
    with h5py.File(path, "r") as f:
        out["session_id"] = scalar(f, "general/session_id")
        out["session_description"] = scalar(f, "general/session_description")
        out["genotype"] = scalar(f, "general/subject/genotype")
        out["sex"] = scalar(f, "general/subject/sex")
        out["strain"] = scalar(f, "general/subject/strain")
        out["subject_desc"] = scalar(f, "general/subject/description")

        # ---- trials ----
        if "intervals/trials" in f:
            t = f["intervals/trials"]
            cur = t["current_uA"][:]
            out["n_trials"] = int(len(cur))
            out["trial_currents"] = sorted(np.unique(cur).tolist())
            out["n_catch"] = int(np.sum(cur == 0))
            out["trial_channels"] = sorted(np.unique(t["stim_channel"][:]).tolist())
            if "is_hit" in t:
                hit = t["is_hit"][:]
                out["hit_rate_overall"] = float(hit.mean())
                out["hit_by_current"] = {
                    float(c): [float(hit[cur == c].mean()), int((cur == c).sum())]
                    for c in np.unique(cur)
                }
            if "is_good_trial" in t:
                out["n_good_trials"] = int(t["is_good_trial"][:].sum())
            if "response_time" in t:
                rt = t["response_time"][:]
                out["response_time_finite_frac"] = float(np.isfinite(rt).mean())
            dur = t["stop_time"][:] - t["start_time"][:]
            out["trial_dur_s"] = [float(np.min(dur)), float(np.median(dur)), float(np.max(dur))]
            out["trial_cols"] = sorted(t.keys())

        # ---- stimulation table ----
        if "intervals/electrical_stimulation" in f:
            s = f["intervals/electrical_stimulation"]
            out["n_stim_events"] = int(len(s["current_uA"]))
            out["stim_currents"] = sorted(np.unique(s["current_uA"][:]).tolist())
            out["stim_freqs"] = sorted(np.unique(s["frequency_hz"][:]).tolist())
            out["stim_pulse_counts"] = sorted(np.unique(s["pulse_count"][:]).tolist())
            out["stim_pulse_widths"] = sorted(np.unique(s["pulse_width_us"][:]).tolist())
            out["stim_channels"] = sorted(np.unique(s["stim_channel"][:]).tolist())
            sdur = s["stop_time"][:] - s["start_time"][:]
            out["stim_dur_s"] = [float(np.min(sdur)), float(np.median(sdur)), float(np.max(sdur))]
            out["stim_cols"] = sorted(s.keys())
            # unique full parameter combinations
            combos = defaultdict(int)
            for c, fr, pc, pw, ch in zip(
                s["current_uA"][:],
                s["frequency_hz"][:],
                s["pulse_count"][:],
                s["pulse_width_us"][:],
                s["stim_channel"][:],
            ):
                combos[(float(c), float(fr), int(pc), float(pw), int(ch))] += 1
            out["n_stim_combos"] = len(combos)
            out["stim_combos"] = {str(k): v for k, v in sorted(combos.items())}

        # ---- units ----
        if "units" in f:
            u = f["units"]
            out["n_units"] = int(len(u["id"]))
            if "accepted" in u:
                out["n_units_accepted"] = int(u["accepted"][:].sum())
            if "cell_type" in u:
                ct = [x.decode() if isinstance(x, bytes) else x for x in u["cell_type"][:]]
                vals, cnts = np.unique(ct, return_counts=True)
                out["cell_types"] = dict(zip(vals.tolist(), cnts.tolist()))
            out["unit_cols"] = sorted(u.keys())
            st = u["spike_times"][:]
            out["spike_time_range"] = [float(st.min()), float(st.max())] if len(st) else None
            out["n_spikes"] = int(len(st))

        # ---- ophys ----
        if "processing/ophys/DfOverF" in f:
            g = f["processing/ophys/DfOverF"]
            name = list(g.keys())[0]
            d = g[name]["data"]
            out["ophys_shape"] = list(d.shape)
            out["ophys_rate"] = rate_of(f, f"processing/ophys/DfOverF/{name}")
            out["ophys_name"] = name
        # imaging planes
        if "general/optophysiology" in f:
            planes = list(f["general/optophysiology"].keys())
            out["n_imaging_planes"] = len(planes)
            p0 = planes[0]
            out["imaging_rate"] = float(scalar(f, f"general/optophysiology/{p0}/imaging_rate", 0))
            out["indicator"] = scalar(f, f"general/optophysiology/{p0}/indicator")
            out["imaging_location"] = scalar(f, f"general/optophysiology/{p0}/location")

        # ---- behaviour ----
        if "processing/behavior" in f:
            beh = {}
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("data"):
                    beh[name] = [list(obj.shape), str(obj.dtype)]
            f["processing/behavior"].visititems(visit)
            out["behavior_series"] = beh
            for k in list(beh):
                grp = "processing/behavior/" + k.rsplit("/", 1)[0]
                out.setdefault("behavior_rates", {})[k] = rate_of(f, grp)

        # ---- electrodes ----
        if "general/extracellular_ephys/electrodes" in f:
            e = f["general/extracellular_ephys/electrodes"]
            out["n_electrodes"] = int(len(e["id"]))
            locs = [x.decode() if isinstance(x, bytes) else x for x in e["location"][:]]
            out["electrode_locations"] = sorted(set(locs))
            out["electrode_xyz_ranges"] = {
                ax: [float(e[f"rel_{ax}"][:].min()), float(e[f"rel_{ax}"][:].max())]
                for ax in ("x", "y", "z")
            }
        out["protocol"] = scalar(f, "general/protocol")
        out["notes"] = scalar(f, "general/notes")
        out["experiment_description"] = scalar(f, "general/experiment_description")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/dandi001868", type=Path)
    ap.add_argument("--out", default="results/tables/dandi001868_characterization.json", type=Path)
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.nwb"))
    recs = []
    for p in files:
        try:
            recs.append(describe(p))
        except Exception as exc:  # pragma: no cover
            recs.append({"path": p.name, "subject": p.parent.name, "error": repr(exc)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recs, indent=1, default=str))

    # ---- console summary ----
    print(f"{len(recs)} sessions\n")
    ex = next(r for r in recs if "error" not in r)
    for k in (
        "experiment_description",
        "protocol",
        "notes",
        "session_description",
        "indicator",
        "imaging_location",
        "genotype",
        "strain",
    ):
        print(f"{k}: {str(ex.get(k))[:400]}")
    print()
    by_sub = defaultdict(list)
    for r in recs:
        by_sub[r["subject"]].append(r)
    hdr = (
        f"{'subject':14s} {'ses':4s} {'trials':>7s} {'catch':>6s} {'stim':>6s} "
        f"{'units':>6s} {'rois':>6s} {'combos':>7s} {'currents':>34s} {'freqs':>18s} {'chans':>16s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for sub in sorted(by_sub, key=lambda s: int(s.replace("sub-ICMS", ""))):
        rs = by_sub[sub]
        cur, frq, chn = set(), set(), set()
        tr = ca = st = 0
        un, roi, cmb = [], [], 0
        for r in rs:
            cur |= set(r.get("stim_currents") or [])
            frq |= set(r.get("stim_freqs") or [])
            chn |= set(r.get("stim_channels") or [])
            tr += r.get("n_trials", 0)
            ca += r.get("n_catch", 0)
            st += r.get("n_stim_events", 0)
            un.append(r.get("n_units", 0))
            roi.append((r.get("ophys_shape") or [0, 0])[1])
            cmb += r.get("n_stim_combos", 0)
        print(
            f"{sub:14s} {len(rs):4d} {tr:7d} {ca:6d} {st:6d} "
            f"{int(np.mean(un)):6d} {int(np.mean(roi)):6d} {cmb:7d} "
            f"{str(sorted(cur))[:34]:>34s} {str(sorted(frq))[:18]:>18s} {str(sorted(chn))[:16]:>16s}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
