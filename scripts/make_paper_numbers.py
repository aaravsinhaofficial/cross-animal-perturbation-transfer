"""Emit every number the paper quotes as a LaTeX macro, straight from the result
files, so the text cannot drift away from the experiments."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def fmt(x, nd=3):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{nd}f}" if abs(x) < 100 else f"{x:.1f}"


def pval(p):
    if p is None or not np.isfinite(p):
        return "--"
    if p < 1e-12:
        return r"<10^{-12}"
    e = int(np.floor(np.log10(p)))
    m = p / 10**e
    return rf"{m:.1f}\times 10^{{{e}}}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=Path, default=Path("results/icms_ladder.json"))
    ap.add_argument("--audit", type=Path, default=Path("results/tables/icms_audit.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--teacher", type=Path, nargs="*", default=[
        Path("results/teacher_shared.json"),
        Path("results/teacher_heterogeneous.json"),
        Path("results/teacher_degenerate.json"),
    ])
    ap.add_argument("--levels", type=Path, default=Path("results/tables/transfer_levels_fr.json"))
    ap.add_argument("--out", type=Path, default=Path("paper/numbers.tex"))
    args = ap.parse_args()

    M: dict[str, str] = {}

    # ---- dataset ----
    if args.cache.exists():
        with args.cache.open("rb") as fh:
            ds = pickle.load(fh)["dataset"]
        M["NAnimals"] = str(len(ds.animals))
        M["NSessions"] = str(len(ds.sets))
        M["NUnits"] = str(sum(s.n_obs for s in ds.sets))
        M["NStimTrials"] = f"{sum(int(s.perturbed.sum()) for s in ds.sets):,}"
        M["NUnpTrials"] = f"{sum(int((~s.perturbed).sum()) for s in ds.sets):,}"
        M["BinMs"] = f"{ds.bin_s*1000:.0f}"
        amps = sorted({round(float(v), 1) for s in ds.sets
                       for v in (s.meta["cond_amp"].values())})
        M["AmpMin"] = f"{min(amps):g}"
        M["AmpMax"] = f"{max(amps):g}"
        M["NAmps"] = str(len(amps))
        nconds = [len({int(c) for c in s.cond[s.perturbed]}) for s in ds.sets]
        M["MedConds"] = f"{int(np.median(nconds))}"
        M["MinUnits"] = str(min(s.n_obs for s in ds.sets))
        M["MaxUnits"] = str(max(s.n_obs for s in ds.sets))

    if args.audit.exists():
        aud = json.loads(args.audit.read_text())
        ce = [v["ceiling"] for v in aud["sets"].values() if v.get("ceiling")]
        cb = [v["ceiling_behavior"] for v in aud["sets"].values() if v.get("ceiling_behavior")]
        M["CeilUnitMed"] = f"{np.median(ce):.3f}"
        M["CeilBehMed"] = f"{np.median(cb):.3f}"
        M["NAuditProblems"] = str(len(aud["problems"]))

    # ---- ladder ----
    if args.ladder.exists():
        lad = json.loads(args.ladder.read_text())
        res, tests = lad["results"], lad["tests"]
        name = {"unit": "Unit", "depth_band": "Band", "population": "Pop",
                "wheel_speed": "Wheel", "detection_prob": "Det"}
        gname = {"in_sample": "In", "cross_session": "XSess",
                 "cross_animal": "XAni", "cross_animal_unseen_amp": "XAniUnseen"}
        for lv, ln in name.items():
            for g, gn in gname.items():
                r = res.get(f"{lv}|{g}|shared_operator")
                if not r:
                    continue
                M[f"{ln}{gn}"] = fmt(r["delta_r2"])
                M[f"{ln}{gn}Lo"] = fmt(r["ci"][0])
                M[f"{ln}{gn}Hi"] = fmt(r["ci"][1])
                M[f"{ln}{gn}R"] = fmt(r["delta_corr"])
                M[f"{ln}{gn}Ceil"] = f"{r['ceiling']:.3f}"
                M[f"{ln}{gn}Frac"] = f"{r['frac_of_ceiling']:.2f}"
                M[f"{ln}{gn}Pos"] = f"{r['sessions_above_zero']}/{r['n_sessions']}"
            r = res.get(f"{lv}|cross_animal|physical_ridge")
            if r:
                M[f"{ln}XAniRidge"] = fmt(r["delta_r2"])
        for lv, ln in name.items():
            t = tests.get(lv)
            if t:
                M[f"{ln}P"] = pval(t["p_perm"])
                M[f"{ln}PW"] = pval(t["wilcoxon_p"])
        # per-animal detection numbers
        r = res.get("detection_prob|cross_animal|shared_operator")
        if r:
            pa = r["per_animal"]
            M["DetPerAnimalMin"] = fmt(min(pa.values()))
            M["DetPerAnimalMax"] = fmt(max(pa.values()))
            M["DetNAnimalsPos"] = f"{sum(v > 0 for v in pa.values())}/{len(pa)}"
            M["DetPerAnimalList"] = ", ".join(
                f"{k.replace('sub-ICMS','m')}: {v:+.2f}" for k, v in pa.items()
            )

    # ---- generalisation levels probe (fluctuation-response features) ----
    if args.levels.exists():
        lv = json.loads(args.levels.read_text())
        for k, tag in [("within_session", "FRWithin"), ("cross_session", "FRXSess"),
                       ("cross_animal", "FRXAni")]:
            if k in lv:
                M[tag] = fmt(lv[k]["delta_r2"])
                M[tag + "R"] = fmt(lv[k]["delta_corr"])

    # ---- teacher ----
    for p in args.teacher:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if not d.get("summary"):
            continue
        regime = p.stem.replace("teacher_", "")
        tag = {"shared": "TShared", "heterogeneous": "THetero",
               "degenerate": "TDegen"}.get(regime, "T" + regime.title())
        s = d["summary"]["methods"]
        for meth, mt in [("cadence", ""), ("oracle", "Oracle"), ("ma_latent", "MaLat"),
                         ("ma_cca", "MaCca"), ("unit_ridge", "Ridge"),
                         ("ctrl_permuted_obs", "CtrlPerm"),
                         ("ctrl_scrambled_interv", "CtrlScr")]:
            v = s.get(meth, {}).get("neural.delta_r2")
            if v:
                M[tag + mt] = fmt(v["mean"])
                M[tag + mt + "Lo"] = fmt(v["ci_lo"])
                M[tag + mt + "Hi"] = fmt(v["ci_hi"])
        c = s.get("cadence", {}).get("ceiling.delta_r2_ceiling")
        if c:
            M[tag + "Ceil"] = f"{c['mean']:.3f}"
        b = s.get("cadence", {}).get("behavior.delta_r2")
        if b:
            M[tag + "Beh"] = fmt(b["mean"])
        for gk, gt in [("group.group:conserved", "Cons"),
                       ("group.group:idiosyncratic", "Idio")]:
            v = s.get("cadence", {}).get(gk)
            if v:
                M[tag + gt] = fmt(v["mean"])
        pt = d["summary"].get("paired_tests_vs_cadence", {})
        for meth, mt in [("no_effect", "VsNull"), ("ma_latent", "VsMaLat")]:
            if meth in pt:
                M[tag + mt + "P"] = pval(pt[meth]["p_perm"])
        M[tag + "NFolds"] = str(d["summary"].get("n_folds", 0))
        M[tag + "Pos"] = str(d["summary"].get("cadence_folds_above_zero", "--"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Auto-generated by scripts/make_paper_numbers.py -- do not edit.",
             "% Every number quoted in the paper resolves to a result file."]
    # \def (not \newcommand) so these override the fallbacks declared in main.tex
    for k, v in sorted(M.items()):
        lines.append(rf"\expandafter\def\csname {k}\endcsname{{{v}}}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} with {len(M)} macros")
    missing = [k for k in ("DetXAni", "UnitXAni") if k not in M]
    if missing:
        print(f"  warning: missing {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
