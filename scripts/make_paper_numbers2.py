"""Emit the paper's numbers as LaTeX macros, straight from the result files.

Everything the text quotes resolves to a file under results/, so the prose cannot
drift away from the experiments. Animal-level values are the default; session-level
values carry an explicit ``Sess`` suffix.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def f3(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def pv(p):
    if p is None or not np.isfinite(p):
        return "--"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, default=Path("results/final_analysis.json"))
    ap.add_argument("--readout", type=Path, default=Path("results/tables/readout_oracle.json"))
    ap.add_argument("--unitgain", type=Path, default=Path("results/tables/unit_gain.json"))
    ap.add_argument("--cortex", type=Path, default=Path("results/cortex_sweep.json"))
    ap.add_argument("--audit", type=Path, default=Path("results/tables/icms_audit.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--out", type=Path, default=Path("paper/numbers.tex"))
    args = ap.parse_args()
    M: dict[str, str] = {}

    # ---------- dataset ----------
    if args.cache.exists():
        with args.cache.open("rb") as fh:
            ds = pickle.load(fh)["dataset"]
        M["NAnimals"] = str(len(ds.animals))
        M["NSessions"] = str(len(ds.sets))
        M["NUnits"] = str(sum(s.n_obs for s in ds.sets))
        M["NStimTrials"] = f"{sum(int(s.perturbed.sum()) for s in ds.sets):,}"
        M["NUnpTrials"] = f"{sum(int((~s.perturbed).sum()) for s in ds.sets):,}"
        M["BinMs"] = f"{ds.bin_s*1000:.0f}"
        amps = sorted({round(float(v), 1) for s in ds.sets for v in s.meta["cond_amp"].values()})
        M["AmpMin"], M["AmpMax"], M["NAmps"] = f"{min(amps):g}", f"{max(amps):g}", str(len(amps))
        M["MinUnits"] = str(min(s.n_obs for s in ds.sets))
        M["MaxUnits"] = str(max(s.n_obs for s in ds.sets))
        M["MedUnits"] = str(int(np.median([s.n_obs for s in ds.sets])))
        M["PFloor"] = f"{2.0 ** -(len(ds.animals) - 1):.3f}"
    if args.audit.exists():
        aud = json.loads(args.audit.read_text())
        ce = [v["ceiling"] for v in aud["sets"].values() if v.get("ceiling")]
        cb = [v["ceiling_behavior"] for v in aud["sets"].values() if v.get("ceiling_behavior")]
        M["CeilUnitMed"] = f2(float(np.median(ce)))
        M["CeilBehMed"] = f2(float(np.median(cb)))
        M["NAuditProblems"] = str(len(aud["problems"]))

    # ---------- the main table ----------
    if args.final.exists():
        fin = json.loads(args.final.read_text())
        res, tests = fin["results"], fin.get("tests", {})
        lv = {"detection": "Beh", "wheel": "Wheel", "population": "Pop",
              "depth_band": "Band", "unit": "Unit"}
        ho = {"none": "", "amp_interior": "AmpIn", "amp_block": "AmpBlock",
              "amp_high_extrap": "AmpHigh", "amp_low_extrap": "AmpLow",
              "depth_superficial": "DepthSup", "depth_deep": "DepthDeep",
              "amp_x_depth": "AmpDepth"}
        me = {"zero": "Zero", "group_mean": "Group", "group_interp": "GroupI",
              "dose_gam": "Gam", "dose_physical": "Dose", "dose_plus_spont": "DoseSp",
              "linear_response": "LR", "lr_plus_gain": "LRG"}
        for k, r in res.items():
            a, b, c = k.split("|")
            if a not in lv or b not in ho or c not in me:
                continue
            tag = lv[a] + ho[b] + me[c]
            M[tag] = f3(r["animal_mean"])
            M[tag + "Lo"] = f3(r["ci_lo"])
            M[tag + "Hi"] = f3(r["ci_hi"])
            M[tag + "Pos"] = f"{r['sign_test']['n_positive']}/{r['sign_test']['n']}"
            M[tag + "P"] = pv(r["permutation"]["p"])
            M[tag + "R"] = f3(r["delta_corr"])
            M[tag + "Ceil"] = f2(r["ceiling"])
            M[tag + "Sess"] = f3(r["session_mean"])
        for k, t in tests.items():
            a, b = k.split("|")
            if a in lv and b in ho:
                tag = "Cmp" + lv[a] + ho[b]
                M[tag + "Diff"] = f3(t["mean_diff"])
                M[tag + "P"] = pv(t["p"])
                M[tag + "Base"] = t["best_simple"].replace("_", " ")

    # ---------- readout decomposition ----------
    if args.readout.exists():
        ro = json.loads(args.readout.read_text())
        for k, t in (("none", "RoNone"), ("gain", "RoGain"),
                     ("gain+offset", "RoGainOff"), ("timecourse", "RoTime")):
            if k in ro:
                M[t] = f3(ro[k].get("animal_mean"))
                M[t + "Lo"] = f3(ro[k].get("animal_ci", [np.nan, np.nan])[0])
                M[t + "Hi"] = f3(ro[k].get("animal_ci", [np.nan, np.nan])[1])
                M[t + "Pos"] = str(ro[k].get("animals_positive", "--"))
                M[t + "P"] = pv(ro[k].get("animal_p"))
        t = ro.get("test_gain_vs_none", {}).get("animal")
        if t:
            M["RoGainDiff"] = f3(t["mean_diff"])
            M["RoGainDiffP"] = pv(t["p"])

    # ---------- per-unit gain prediction ----------
    if args.unitgain.exists():
        ug = json.loads(args.unitgain.read_text())
        for k, t in (("none", "UgNone"), ("predicted", "UgPred"), ("oracle", "UgOracle")):
            if k in ug:
                M[t] = f3(ug[k].get("animal_mean"))
                M[t + "Pos"] = str(ug[k].get("animals_positive", "--"))
                M[t + "P"] = pv(ug[k].get("animal_p"))
        if "unit_gain_corr" in ug:
            M["UgR"] = f3(ug["unit_gain_corr"])
        t = ug.get("test_predicted_vs_none", {}).get("animal")
        if t:
            M["UgDiff"] = f3(t["mean_diff"])
            M["UgDiffP"] = pv(t["p"])

    # ---------- simulated cortex ----------
    if args.cortex.exists():
        cx = json.loads(args.cortex.read_text())
        for row in cx:
            tag = f"Cx{row['recruit'].title()}N{row['n_obs']}"
            M[tag] = f3(row["unit"]["animal_mean"])
            M[tag + "Lo"] = f3(row["unit"]["ci_lo"])
            M[tag + "Hi"] = f3(row["unit"]["ci_hi"])
            M[tag + "Pos"] = (f"{row['unit']['sign_test']['n_positive']}"
                              f"/{row['unit']['sign_test']['n']}")
            M[tag + "Pop"] = f3(row["population"]["animal_mean"])
        for rec in {r["recruit"] for r in cx}:
            sub = sorted([r for r in cx if r["recruit"] == rec], key=lambda r: r["n_obs"])
            vals = [r["unit"]["animal_mean"] for r in sub]
            wid = [r["unit"]["ci_hi"] - r["unit"]["ci_lo"] for r in sub]
            ns = [r["n_obs"] for r in sub]
            M[f"Cx{rec.title()}Mean"] = f3(float(np.mean(vals)))
            M[f"Cx{rec.title()}Width"] = f2(float(np.mean(wid)))
            if len(ns) > 2:
                M[f"Cx{rec.title()}Trend"] = f3(
                    float(np.corrcoef(np.log(ns), vals)[0, 1]))
        loc = {r["n_obs"]: r["unit"]["animal_mean"] for r in cx if r["recruit"] == "local"}
        spa = {r["n_obs"]: r["unit"]["animal_mean"] for r in cx if r["recruit"] == "sparse"}
        both = sorted(set(loc) & set(spa))
        if both:
            M["CxGap"] = f3(float(np.mean([loc[n] - spa[n] for n in both])))
        obs = sorted({r["n_obs"] for r in cx})
        M["CxNObsMin"], M["CxNObsMax"] = str(min(obs)), str(max(obs))
        for rec in {r["recruit"] for r in cx}:
            sub = sorted([r for r in cx if r["recruit"] == rec], key=lambda r: r["n_obs"])
            if sub:
                M[f"Cx{rec.title()}Lowest"] = f3(sub[0]["unit"]["animal_mean"])
                M[f"Cx{rec.title()}Highest"] = f3(sub[-1]["unit"]["animal_mean"])
                M[f"Cx{rec.title()}LowestN"] = str(sub[0]["n_obs"])
                M[f"Cx{rec.title()}HighestN"] = str(sub[-1]["n_obs"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Auto-generated by scripts/make_paper_numbers2.py. Do not edit.",
             "% Every number in the paper resolves to a file under results/."]
    for k, v in sorted(M.items()):
        lines.append(rf"\expandafter\def\csname {k}\endcsname{{{v}}}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} with {len(M)} macros")

    # ---------- fallbacks, so a missing experiment cannot break the build ----------
    import re

    tex = (args.out.parent / "main.tex")
    used = set()
    if tex.exists():
        body = tex.read_text()
        for name in re.findall(r"\\([A-Z][A-Za-z]{2,})\b", body):
            used.add(name)
    builtin = {
        "IfFileExists", "Delta", "R", "Poisson", "NoteWidth", "LaTeX", "TeX",
        "Huge", "Large", "Big", "Bigg", "Roman", "Alph", "Alpha", "Beta", "Gamma",
        "Sigma", "Omega", "Theta", "Lambda", "Phi", "Psi", "Pi", "Xi", "Upsilon",
    }
    miss = sorted((used - set(M)) - builtin)
    # \providecommand cannot take a name containing a digit, so define through
    # \csname, which accepts any name
    fb = ["% Auto-generated. Placeholders for numbers whose experiment has not run.",
          r"\makeatletter"]
    fb += [rf"\@ifundefined{{{k}}}{{\expandafter\def\csname {k}\endcsname{{--}}}}{{}}"
           for k in sorted(set(M) | set(miss))]
    fb.append(r"\makeatother")
    (args.out.parent / "fallbacks.tex").write_text("\n".join(fb) + "\n")
    if miss:
        print(f"  note: {len(miss)} macro(s) used in main.tex but not produced: "
              f"{miss[:8]}{' ...' if len(miss) > 8 else ''}")

    # ---------- tables ----------
    if args.final.exists():
        fin = json.loads(args.final.read_text())
        res = fin["results"]

        def row(level, hold, meth, label):
            r = res.get(f"{level}|{hold}|{meth}")
            if not r:
                return None
            return (f"{label} & {f3(r['animal_mean'])} & "
                    f"[{f3(r['ci_lo'])}, {f3(r['ci_hi'])}] & "
                    f"{f3(r['delta_corr'])} & "
                    f"{r['sign_test']['n_positive']}/{r['sign_test']['n']} & "
                    f"{pv(r['permutation']['p'])} \\\\")

        t = [r"\begin{table}[h]\centering\small",
             r"\caption{What transfers to a held-out animal. Scores are averaged over "
             r"animals, with a bootstrap over animals for the interval and an exact "
             r"sign-flip permutation over animals for the p value. A score of zero is "
             r"the model that says the stimulus does nothing. With six animals the "
             r"smallest attainable p value is 0.031.}",
             r"\label{tab:main}",
             r"\begin{tabular}{llccccc}", r"\toprule",
             r"readout & method & $\Delta R^2$ & 95\% CI & $r$ & animals $>0$ & $p$ \\",
             r"\midrule"]
        blocks = [
            ("behaviour", "detection",
             [("group_mean", "average of the other mice"),
              ("dose_gam", "smooth curve in current and time"),
              ("dose_physical", "shared rule (this paper)"),
              ("dose_plus_spont", "shared rule $+$ resting activity")]),
            ("population", "population",
             [("group_mean", "average of the other mice"),
              ("dose_physical", "shared rule"),
              ("linear_response", "shared rule through own dynamics"),
              ("lr_plus_gain", "the same $+$ predicted responsiveness")]),
            ("single neurons", "unit",
             [("group_mean", "average of the other mice"),
              ("dose_physical", "shared rule"),
              ("linear_response", "shared rule through own dynamics"),
              ("lr_plus_gain", "the same $+$ predicted responsiveness")]),
        ]
        for i, (name, level, ms) in enumerate(blocks):
            rows = [row(level, "none", m, lab) for m, lab in ms]
            rows = [x for x in rows if x]
            if not rows:
                continue
            if i:
                t.append(r"\midrule")
            t.append(rf"\multirow{{{len(rows)}}}{{*}}{{{name}}}")
            for j, rr in enumerate(rows):
                t.append(("& " if j else "& ") + rr)
        t += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

        # holdout table for behaviour
        t += [r"\begin{table}[h]\centering\small",
              r"\caption{Behavioural transfer when the stimulus setting is also "
              r"removed from every training animal. Deleting a middle current is close "
              r"to interpolating between its neighbours; the other rows are harder.}",
              r"\label{tab:holdout}",
              r"\begin{tabular}{lcccc}", r"\toprule",
              r"held out from all training animals & $\Delta R^2$ & 95\% CI & "
              r"animals $>0$ & $p$ \\", r"\midrule"]
        for h, lab in (("none", "nothing"),
                       ("amp_interior", "one middle current"),
                       ("amp_block", "a block of three currents"),
                       ("amp_high_extrap", "the top of the current range"),
                       ("amp_low_extrap", "the bottom of the current range"),
                       ("depth_superficial", "all superficial contacts"),
                       ("depth_deep", "all deep contacts"),
                       ("amp_x_depth", "high current at superficial contacts")):
            r = res.get(f"detection|{h}|dose_physical")
            if not r:
                continue
            t.append(f"{lab} & {f3(r['animal_mean'])} & "
                     f"[{f3(r['ci_lo'])}, {f3(r['ci_hi'])}] & "
                     f"{r['sign_test']['n_positive']}/{r['sign_test']['n']} & "
                     f"{pv(r['permutation']['p'])} \\\\")
        t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (args.out.parent / "tables.tex").write_text("\n".join(t) + "\n")
        print(f"wrote {args.out.parent / 'tables.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
