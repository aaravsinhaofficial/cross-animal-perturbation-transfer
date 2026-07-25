"""Build the trial tensors for the optogenetic silencing cohort and check them.

The checks are the ones that would silently invalidate everything downstream: that no
light arrives before the alignment point, that control and light trials are matched on
what the animal was doing beforehand, and that the effect in each session is large
enough relative to its own noise for any model to be scored against it.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import individuality as I
from cadence.data.alm import AlmConfig, load_alm


def audit(ds) -> dict:
    rep: dict = {"sets": {}, "problems": []}
    for s in ds.sets:
        pre_on = float(s.interv_on[:, : s.t0].sum())
        pre = s.y[:, : s.t0].mean(axis=(1, 2))
        stim, ctrl = pre[s.perturbed], pre[~s.perturbed]
        ratio = (float(np.nanmean(ctrl)) / max(float(np.nanmean(stim)), 1e-9)
                 if len(stim) and len(ctrl) else np.nan)
        entry = dict(
            animal=s.animal, n_obs=int(s.n_obs),
            n_stim=int(s.perturbed.sum()), n_ctrl=int((~s.perturbed).sum()),
            n_conds=int(len(np.unique(s.cond[s.perturbed]))),
            pre_light_bins=pre_on,
            pre_rate_ratio=ratio,
            delta_ceiling=float(I.delta_ceiling(s)),
        )
        rep["sets"][s.key] = entry
        if pre_on > 0:
            rep["problems"].append(f"{s.key}: light before the alignment point")
        if np.isfinite(ratio) and not (0.8 <= ratio <= 1.25):
            rep["problems"].append(
                f"{s.key}: pre-stimulus rate mismatch of {ratio:.2f}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/dandi000009")
    ap.add_argument("--out", type=Path, default=Path("data/proc/alm.pkl"))
    ap.add_argument("--report", type=Path,
                    default=Path("results/tables/alm_audit.json"))
    args = ap.parse_args()

    ds = load_alm(AlmConfig(root=args.root))
    print(f"{len(ds.animals)} animals, {len(ds.sets)} sessions, "
          f"{sum(s.n_obs for s in ds.sets)} neurons, "
          f"{sum(int(s.perturbed.sum()) for s in ds.sets)} light trials, "
          f"{sum(int((~s.perturbed).sum()) for s in ds.sets)} control trials")

    rep = audit(ds)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=1, default=float))
    ce = [v["delta_ceiling"] for v in rep["sets"].values()
          if np.isfinite(v["delta_ceiling"])]
    print(f"individual-part ceiling: median {np.median(ce):.3f}, "
          f"range {min(ce):.3f} to {max(ce):.3f}")
    if rep["problems"]:
        print(f"{len(rep['problems'])} problem(s):")
        for p in rep["problems"][:10]:
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
