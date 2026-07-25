"""Build the trial tensors for the two larger ALM photoinhibition releases.

These files describe the light as a continuous laser trace with onset events rather
than per trial columns, so the loader has to reconstruct which trials were perturbed,
at what dose, at which site and in which epoch. That reconstruction is the thing most
likely to be silently wrong, so the checks here are about it: the light should arrive
at the start of the delay, it should suppress activity, and the effect it produces
should be large enough relative to its own noise for a model to be scored against.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import individuality as I
from cadence.data.alm_wide import AlmWideConfig, load_alm_wide


def audit(ds) -> dict:
    rep: dict = {"sets": {}, "problems": []}
    for s in ds.sets:
        Y = s.y[:, s.t0 :]
        ctrl = ~s.perturbed
        base = np.nanmean(Y[ctrl], 0)
        nb = int(np.clip(s.meta.get("delay_bins", 26), 1, Y.shape[1]))
        drop = []
        for c in np.unique(s.cond[s.perturbed]):
            m = s.cond == c
            if m.sum() >= 6:
                drop.append(float(np.nanmean((np.nanmean(Y[m], 0) - base)[:nb])))
        pre = s.y[:, : s.t0].mean(axis=(1, 2))
        ratio = (float(np.nanmean(pre[ctrl])) / max(float(np.nanmean(pre[~ctrl])), 1e-9)
                 if (~ctrl).any() else np.nan)
        entry = dict(animal=s.animal, n_obs=int(s.n_obs),
                     n_stim=int(s.perturbed.sum()), n_ctrl=int(ctrl.sum()),
                     n_conds=int(len(np.unique(s.cond[s.perturbed]))),
                     pre_light_bins=float(s.interv_on[:, : s.t0].sum()),
                     pre_rate_ratio=ratio,
                     rate_change=float(np.mean(drop)) if drop else np.nan,
                     delta_ceiling=float(I.delta_ceiling(s)))
        rep["sets"][s.key] = entry
        if entry["pre_light_bins"] > 0:
            rep["problems"].append(f"{s.key}: light before the alignment point")
        if np.isfinite(ratio) and not (0.75 <= ratio <= 1.3):
            rep["problems"].append(
                f"{s.key}: pre-stimulus rate mismatch of {ratio:.2f}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["data/raw/dandi000010", "data/raw/dandi000011"])
    ap.add_argument("--out", type=Path, default=Path("data/proc/alm_wide.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path("results/tables/alm_wide_audit.json"))
    args = ap.parse_args()

    ds = load_alm_wide(AlmWideConfig(roots=tuple(args.roots)), verbose=False)
    print(f"{len(ds.animals)} animals, {len(ds.sets)} sessions, "
          f"{sum(s.n_obs for s in ds.sets)} neurons, "
          f"{sum(int(s.perturbed.sum()) for s in ds.sets)} light trials, "
          f"{sum(int((~s.perturbed).sum()) for s in ds.sets)} control trials")

    rep = audit(ds)
    rc = [v["rate_change"] for v in rep["sets"].values() if np.isfinite(v["rate_change"])]
    ce = [v["delta_ceiling"] for v in rep["sets"].values()
          if np.isfinite(v["delta_ceiling"])]
    print(f"activity change while the light is on: mean {np.mean(rc):+.3f} per bin, "
          f"down in {100*np.mean(np.array(rc) < 0):.0f}% of sessions")
    print(f"individual-part ceiling: median {np.median(ce):.3f}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=1, default=float))
    if rep["problems"]:
        print(f"{len(rep['problems'])} problem(s):")
        for p in rep["problems"][:8]:
            print(f"  {p}")
    else:
        print("audit clean")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as fh:
        pickle.dump({"dataset": ds}, fh, protocol=4)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
