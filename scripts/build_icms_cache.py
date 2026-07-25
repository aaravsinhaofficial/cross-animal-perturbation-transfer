"""Build (and cache) the ICMS trial tensors, then audit the split for leakage."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cadence import metrics as M
from cadence.data.icms import IcmsConfig, load_icms


def audit(ds) -> dict:
    """Checks that must pass before any modelling happens."""
    rep: dict = {"sets": {}, "problems": []}
    for s in ds.sets:
        pre_on = float(s.interv_on[:, : s.t0].sum())
        unp_on = float(s.interv_on[~s.perturbed].sum())
        unp_raw = float(np.abs(s.interv_raw[~s.perturbed]).sum())
        y_post = s.y[:, s.t0 :]
        ce = M.noise_ceiling(y_post, s.cond, s.perturbed, n_splits=150)
        ceb = (
            M.noise_ceiling(s.behavior[:, s.t0 :], s.cond, s.perturbed, n_splits=150)
            if s.behavior is not None
            else {}
        )
        # pre-stimulus activity must not differ between stim and unperturbed
        # trials, otherwise "predict from unperturbed initial conditions" would
        # be biased
        pre_p = s.y[s.perturbed, : s.t0].mean()
        pre_u = s.y[~s.perturbed, : s.t0].mean()
        info = {
            "n_obs": s.n_obs,
            "n_trials": s.n_trials,
            "n_perturbed": int(s.perturbed.sum()),
            "n_unperturbed": int((~s.perturbed).sum()),
            "n_catch": s.meta.get("n_catch"),
            "n_iti": s.meta.get("n_iti"),
            "n_conditions": len(set(s.cond[s.perturbed].tolist())),
            "interv_on_in_pre_window": pre_on,
            "interv_on_in_unperturbed": unp_on,
            "interv_raw_in_unperturbed": unp_raw,
            "pre_rate_perturbed": float(pre_p),
            "pre_rate_unperturbed": float(pre_u),
            "pre_rate_ratio": float(pre_p / (pre_u + 1e-9)),
            "ceiling": ce.get("delta_r2_ceiling"),
            "ceiling_behavior": ceb.get("delta_r2_ceiling"),
            "cond_info": {str(k): v for k, v in (s.meta.get("cond_info") or {}).items()},
        }
        rep["sets"][s.key] = info
        if pre_on != 0:
            rep["problems"].append(f"{s.key}: intervention gate active before t0")
        if unp_on != 0 or unp_raw != 0:
            rep["problems"].append(f"{s.key}: intervention present on unperturbed trials")
        if not (0.8 < info["pre_rate_ratio"] < 1.25):
            rep["problems"].append(
                f"{s.key}: pre-stimulus rate mismatch ratio={info['pre_rate_ratio']:.3f}"
            )
    # every animal must have at least 2 sessions worth of conditions
    by_animal: dict[str, list[str]] = {}
    for s in ds.sets:
        by_animal.setdefault(s.animal, []).append(s.key)
    rep["animals"] = {k: len(v) for k, v in by_animal.items()}
    for a, ks in by_animal.items():
        if len(ks) < 2:
            rep["problems"].append(f"{a}: only {len(ks)} observation set(s)")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--report", type=Path, default=Path("results/tables/icms_audit.json"))
    ap.add_argument("--bin-s", type=float, default=0.025)
    ap.add_argument("--pre-s", type=float, default=0.5)
    ap.add_argument("--post-s", type=float, default=1.5)
    ap.add_argument("--iti-windows", type=int, default=240)
    args = ap.parse_args()

    cfg = IcmsConfig(
        bin_s=args.bin_s, pre_s=args.pre_s, post_s=args.post_s,
        n_iti_windows=args.iti_windows,
    )
    ds = load_icms(cfg)
    print(f"\n{ds.summary()}\n")
    print(f"animals: {ds.animals}")

    rep = audit(ds)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nAUDIT: {len(rep['problems'])} problem(s)")
    for p in rep["problems"]:
        print("  !", p)
    ceils = [v["ceiling"] for v in rep["sets"].values() if v["ceiling"] is not None]
    ceilb = [v["ceiling_behavior"] for v in rep["sets"].values() if v.get("ceiling_behavior")]
    print(f"neural ceiling:   median {np.median(ceils):.3f}  range [{min(ceils):.3f}, {max(ceils):.3f}]")
    if ceilb:
        print(f"behaviour ceiling: median {np.median(ceilb):.3f} range [{min(ceilb):.3f}, {max(ceilb):.3f}]")
    tot_pert = sum(v["n_perturbed"] for v in rep["sets"].values())
    tot_unp = sum(v["n_unperturbed"] for v in rep["sets"].values())
    tot_units = sum(v["n_obs"] for v in rep["sets"].values())
    print(f"totals: {len(ds.sets)} sets, {tot_units} units, {tot_pert} stim trials, {tot_unp} unperturbed windows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as fh:
        pickle.dump({"dataset": ds, "config": cfg}, fh, protocol=4)
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
