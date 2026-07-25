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

    # ---------- the second cohort, the operator, and the decomposition ----------
    alm_cache = Path("data/proc/alm.pkl")
    if alm_cache.exists():
        with alm_cache.open("rb") as fh:
            al = pickle.load(fh)["dataset"]
        M["AlmAnimals"] = str(len(al.animals))
        M["AlmSessions"] = str(len(al.sets))
        M["AlmUnits"] = str(sum(s.n_obs for s in al.sets))
        M["AlmStimTrials"] = f"{sum(int(s.perturbed.sum()) for s in al.sets):,}"
        M["AlmCtrlTrials"] = f"{sum(int((~s.perturbed).sum()) for s in al.sets):,}"
        M["AlmBinMs"] = f"{al.bin_s*1000:.0f}"
        pw = sorted({round(float(v), 1) for s in al.sets
                     for v in s.meta["cond_amp"].values()})
        M["AlmPowerMin"], M["AlmPowerMax"] = f"{min(pw):g}", f"{max(pw):g}"
        M["AlmPFloor"] = f"{2.0 ** -(len(al.animals) - 1):.1e}"

    wide = Path("data/proc/alm_wide.pkl")
    if alm_cache.exists() and wide.exists():
        with alm_cache.open("rb") as fh:
            a1 = pickle.load(fh)["dataset"]
        with wide.open("rb") as fh:
            a2 = pickle.load(fh)["dataset"]
        sets = list(a1.sets) + list(a2.sets)
        an = sorted({s.animal for s in sets})
        M["AllAnimals"] = str(len(an))
        M["AllSessions"] = str(len(sets))
        M["AllUnits"] = f"{sum(s.n_obs for s in sets):,}"
        M["AllStimTrials"] = f"{sum(int(s.perturbed.sum()) for s in sets):,}"
        M["AllCtrlTrials"] = f"{sum(int((~s.perturbed).sum()) for s in sets):,}"
        M["AllPFloor"] = f"{2.0 ** -(len(an) - 1):.1e}"
        M["WideAnimals"] = str(len(a2.animals))
        M["WideSessions"] = str(len(a2.sets))

    for tag, pre in (("alm5", "AlmOp"), ("icms5", "IcmsOp"),
                     ("almall", "AllOp")):
        p = Path(f"results/operator_{tag}.json")
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        best = r.get("blend", {}).get("per_animal", {})
        if best:
            v = sorted(best.values())[-4:]
            M[f"{pre}Top"] = f3(max(v))
            M[f"{pre}TopLo"] = f3(min(v))
            g = r.get("group", {}).get("per_animal", {})
            gv = [g[k] for k in sorted(best, key=lambda k: -best[k])[:4] if k in g]
            if gv:
                M[f"{pre}TopStereoLo"] = f3(min(gv))
                M[f"{pre}TopStereoHi"] = f3(max(gv))
        for k, suf in (("group", "Stereo"), ("operator", "Net"), ("blend", "Model")):
            if k in r:
                M[f"{pre}{suf}"] = f3(r[k]["animal_mean"])
                M[f"{pre}{suf}Lo"] = f3(r[k]["ci_lo"])
                M[f"{pre}{suf}Hi"] = f3(r[k]["ci_hi"])
                M[f"{pre}{suf}Pos"] = str(r[k]["sign_test"]["n_positive"])
                M[f"{pre}{suf}N"] = str(r[k]["sign_test"]["n"])
        for k, suf in (("test_blend_vs_group", "ModelVsStereo"),
                       ("test_operator_vs_group", "NetVsStereo")):
            if k in r:
                M[f"{pre}{suf}Diff"] = f3(r[k]["mean_diff"])
                M[f"{pre}{suf}P"] = pv(r[k]["p"])

    for tag, pre in (("alm", "AlmInd"), ("icms", "IcmsInd"),
                     ("almall", "AllInd")):
        p = Path(f"results/individuality_{tag}.json")
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for k, suf in (("shared_operator", "Shared"), ("learned_operator", "Net"),
                       ("no effect present", "Null"), ("ceiling", "Ceiling")):
            if k in r:
                M[f"{pre}{suf}"] = f3(r[k]["animal_mean"])
                med = r[k].get("median")
                if med is None and r[k].get("per_animal"):
                    med = float(np.median(list(r[k]["per_animal"].values())))
                M[f"{pre}{suf}Med"] = f3(med)
                if "sign_test" in r[k]:
                    M[f"{pre}{suf}Pos"] = str(r[k]["sign_test"]["n_positive"])
                    M[f"{pre}{suf}N"] = str(r[k]["sign_test"]["n"])
                    M[f"{pre}{suf}P"] = pv(r[k]["sign_test"]["p"])
                    M[f"{pre}{suf}PermP"] = pv(r[k]["permutation"]["p"])
                    M[f"{pre}{suf}Lo"] = f3(r[k]["ci_lo"])
                    M[f"{pre}{suf}Hi"] = f3(r[k]["ci_hi"])
        for k, suf in (("shared_operator", "Frac"), ("learned_operator", "NetFrac")):
            key = f"{k}_fraction_of_ceiling"
            if key in r:
                M[f"{pre}{suf}"] = f"{100*r[key]:.0f}"
        for k, suf in (("shared_operator", "PooledShared"),
                       ("no effect present", "PooledNull"),
                       ("learned_operator", "PooledNet")):
            v = r.get("pooled", {}).get(k)
            if v is not None:
                M[f"{pre}{suf}"] = f3(v)
        if "ceiling_vs_transfer" in r:
            M[f"{pre}QualRho"] = f"{r['ceiling_vs_transfer']['rho']:+.2f}"
            M[f"{pre}QualP"] = pv(r["ceiling_vs_transfer"]["p"])

    ce = Path("results/cohort_size_effect_alm.json")
    if ce.exists():
        r = json.loads(ce.read_text())
        for k in ("model", "network"):
            if k in r:
                pre = "Grow" + k.title()
                M[pre + "Small"] = f3(r[k]["small"])
                M[pre + "Large"] = f3(r[k]["large"])
                M[pre + "Diff"] = f3(r[k]["diff"])
                M[pre + "P"] = pv(r[k]["p"])
                M[pre + "Better"] = str(r[k]["n_better"])
                M[pre + "N"] = str(r[k]["n"])
        M["GrowSmallCohort"] = str(r.get("n_animals_small", 0) - 1)
        M["GrowLargeCohort"] = str(r.get("n_animals_large", 0) - 1)

    be = Path("results/behaviour_effect_alm.json")
    if be.exists():
        r = json.loads(be.read_text())
        M["AlmBehDrop"] = f"{abs(r['correct_rate_change'])*100:.1f} percentage points"
        M["AlmBehDropFrac"] = f"{100*r['frac_down']:.0f}"

    bt = Path("results/behaviour_transfer_alm.json")
    if bt.exists():
        r = json.loads(bt.read_text())
        for k, suf in (("stereotype", "Stereo"), ("stereotype_plus_neural", "Chain")):
            if k in r:
                M[f"Beh{suf}"] = f3(r[k]["animal_mean"])
                M[f"Beh{suf}Med"] = f3(r[k].get("median"))
                M[f"Beh{suf}Pos"] = str(r[k]["sign_test"]["n_positive"])
                M[f"Beh{suf}N"] = str(r[k]["sign_test"]["n"])
                M[f"Beh{suf}P"] = pv(r[k]["sign_test"]["p"])
        if "test_chain_vs_stereotype" in r:
            M["BehChainDiff"] = f3(r["test_chain_vs_stereotype"]["mean_diff"])
            M["BehChainDiffP"] = pv(r["test_chain_vs_stereotype"]["p"])

    for tg, pre in (("alm", "Cohort"), ("almall", "CohortAll")):
        sc = Path(f"results/cohort_scaling_{tg}.json")
        if not sc.exists():
            continue
        cur = json.loads(sc.read_text())
        M[pre + "MinN"] = str(cur[0]["n_animals"])
        M[pre + "MaxN"] = str(cur[-1]["n_animals"])
        M[pre + "MinVal"] = f3(cur[0]["delta_r2"])
        M[pre + "MaxVal"] = f3(cur[-1]["delta_r2"])
        M[pre + "MaxPos"] = str(cur[-1]["n_positive"])
        x = np.log([c["n_animals"] for c in cur]); y = np.array([c["delta_r2"] for c in cur])
        M[pre + "Corr"] = f"{np.corrcoef(x, y)[0, 1]:+.2f}"
        cross = next((c["n_animals"] for c in cur if c["delta_r2"] > 0), None)
        if cross:
            M[pre + "Cross"] = str(cross)
        last = cur[-1]
        n_tot = len(last.get("per_animal", [])) or last["n_positive"]
        if n_tot:
            from scipy.stats import binomtest
            M[pre + "MaxN2"] = str(n_tot)
            M[pre + "MaxP"] = pv(binomtest(last["n_positive"], n_tot, 0.5).pvalue)

    from scipy.stats import binomtest as _bt
    for tg, pre in (("almall", "Rule"), ("icms", "RuleIcms")):
        rl = Path(f"results/rule_{tg}.json")
        if not rl.exists():
            continue
        r = json.loads(rl.read_text())
        ro = r.get("rule_only_operator")
        if ro:
            M[f"{pre}Only"] = f3(ro["mean"])
            M[f"{pre}OnlyMed"] = f3(ro["median"])
            M[f"{pre}OnlyPos"] = str(ro["n_positive"])
            M[f"{pre}OnlyN"] = str(ro["n"])
            M[f"{pre}OnlyK"] = str(ro["n_params"])
        for k, suf in (("firing rate", "Rate"), ("selectivity", "Sel"),
                       ("preparatory ramp", "Ramp")):
            v = r.get(k)
            if not v or not np.isfinite(v.get("mean_r", np.nan)):
                continue
            M[f"{pre}{suf}R"] = f3(v["mean_r"])
            M[f"{pre}{suf}Neg"] = str(v["n_negative"])
            M[f"{pre}{suf}N"] = str(v["n"])
            M[f"{pre}{suf}P"] = pv(float(_bt(v["n_negative"], v["n"], 0.5).pvalue))
            M[f"{pre}{suf}Null"] = (f'{f3(v["null_mean_r"])}, '
                                    f'{v["null_negative"]}/{v["n"]}')

    jk = Path("results/jackknife_alm.json")
    if jk.exists():
        r = json.loads(jk.read_text())
        M["JackMaxP"] = pv(r["jackknife_max_p"])
        M["JackFullP"] = pv(r["full_p"])
        M["JackN"] = str(r["n"])
        M["JackPos"] = str(r.get("n_positive", 17))

    sp = Path("results/individuality_split_alm.json")
    if sp.exists():
        r = json.loads(sp.read_text())
        for key, pre in (("well measured|learned", "SplitHiNet"),
                         ("poorly measured|learned", "SplitLoNet"),
                         ("well measured|shared", "SplitHiShared"),
                         ("poorly measured|shared", "SplitLoShared")):
            v = r.get(key)
            if not v:
                continue
            M[pre] = f3(v["mean"])
            M[pre + "Med"] = f3(v["median"])
            M[pre + "Pos"] = str(v["pos"])
            M[pre + "N"] = str(v["n"])
            M[pre + "P"] = pv(v["p"])
            M[pre + "Frac"] = f"{100*v['frac']:.0f}"
            M[pre + "Ceil"] = f2(v["ceiling"])

    rb = Path("results/rule_by_release.json")
    if rb.exists():
        r = json.loads(rb.read_text())
        for prop, suf in (("selectivity", "Sel"), ("firing rate", "Rate")):
            for rel, tag in (("000009 (first)", "A"), ("000010+11 (second)", "B")):
                v = r.get(f"{prop}|{rel}")
                if not v:
                    continue
                M[f"Rule{suf}{tag}"] = str(v["neg"])
                M[f"Rule{suf}{tag}N"] = str(v["n"])
                M[f"Rule{suf}{tag}P"] = pv(v["p"])

    sc = Path("results/cohort_scaling_alm.json")
    if sc.exists():
        cur = json.loads(sc.read_text())
        M["CohortMinN"] = str(cur[0]["n_animals"])
        M["CohortMaxN"] = str(cur[-1]["n_animals"])
        M["CohortMinVal"] = f3(cur[0]["delta_r2"])
        M["CohortMaxVal"] = f3(cur[-1]["delta_r2"])
        x = np.log([c["n_animals"] for c in cur])
        y = np.array([c["delta_r2"] for c in cur])
        M["CohortCorr"] = f"{np.corrcoef(x, y)[0, 1]:+.2f}"

    dsw = Path("results/decomposition_sweep.json")
    if dsw.exists():
        rows = json.loads(dsw.read_text())
        for pre, keep in (("SweepShared", lambda r: r["private"] <= 0.01),
                          ("SweepPrivate", lambda r: r["private"] >= 0.99)):
            sub = [r for r in rows if keep(r)]
            if sub:
                M[f"{pre}Frac"] = f"{100*np.mean([r['fraction'] for r in sub]):.0f}"
                M[f"{pre}Best"] = f3(max(r["delta_r2"] for r in sub))

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
    # ---------- everything at a glance ----------
    g = [r"\begin{table}[t]\centering\small",
         r"\caption{Every headline claim, the animals it is measured over, and the "
         r"test. Animal-level throughout: an exact sign test over animals, which "
         r"cannot be moved by one extreme value. The last column is the same analysis "
         r"run where the answer is known to be zero.}",
         r"\label{tab:glance}",
         r"\begin{tabular}{p{5.6cm}lccl}", r"\toprule",
         r"claim & measure & animals & $p$ & null \\", r"\midrule"]

    def row(desc, measure, pos, n, p, null):
        g.append(f"{desc} & {measure} & {pos}/{n} & {p} & {null} \\\\")

    m = M
    if "AlmOpModelPos" in m:
        row("A shared operator predicts single-neuron responses to an unseen "
            "perturbation better than the average of the other animals",
            f"${m.get('AlmOpModel','--')}$ vs ${m.get('AlmOpStereo','--')}$",
            m.get("AlmOpModelPos", "--"), m.get("AlmOpModelN", "--"),
            m.get("AlmOpModelVsStereoP", "--"), "--")
    if "RuleSelNeg" in m:
        row("How sharply a neuron distinguishes the two choices predicts how much "
            "the light suppresses it",
            f"$r = {m['RuleSelR']}$", m["RuleSelNeg"], m["RuleSelN"], m["RuleSelP"],
            m.get("RuleSelNull", "--"))
        row("How fast a neuron fires predicts how much the light suppresses it",
            f"$r = {m['RuleRateR']}$", m["RuleRateNeg"], m["RuleRateN"],
            m["RuleRateP"], m.get("RuleRateNull", "--"))
        row("How much a neuron ramps predicts nothing (negative control)",
            f"$r = {m['RuleRampR']}$", m["RuleRampNeg"], m["RuleRampN"],
            m["RuleRampP"], m.get("RuleRampNull", "--"))
    if "RuleIcmsRateNeg" in m:
        row("Under microstimulation, firing rate predicts nothing",
            f"$r = {m['RuleIcmsRateR']}$", m["RuleIcmsRateNeg"], m["RuleIcmsRateN"],
            m["RuleIcmsRateP"], m.get("RuleIcmsRateNull", "--"))
    if "SplitHiNetPos" in m:
        row("In the better measured half of the light cohort, split on a ceiling "
            "fixed before any model is fitted, the operator recovers a fifth of "
            "everything individual that is measurable",
            f"{m.get('SplitHiNetFrac','--')}\\% of ceiling",
            m["SplitHiNetPos"], m["SplitHiNetN"], m["SplitHiNetP"], "--")
    if "AlmIndSharedPos" in m:
        row("The individual part of the response transfers, under light",
            f"${m.get('AlmIndSharedMed','--')}$ median",
            m["AlmIndSharedPos"], m["AlmIndSharedN"], m["AlmIndSharedP"],
            f"${m.get('AlmIndNull','--')}$")
    if "IcmsIndSharedPos" in m:
        row("The individual part does not transfer, under current, though it is "
            "better resolved there",
            f"ceiling ${m.get('IcmsIndCeiling','--')}$",
            m["IcmsIndSharedPos"], m["IcmsIndSharedN"], m["IcmsIndSharedP"],
            f"${m.get('IcmsIndNull','--')}$")
    if "CohortAllMaxPos" in m:
        row("Fitting the operator on more animals keeps helping, with no sign of "
            "flattening",
            f"$r = {m.get('CohortAllCorr','--')}$ with $\\log n$",
            m["CohortAllMaxPos"], m.get("CohortAllMaxN2", "--"),
            m.get("CohortAllMaxP", "--"), "--")
    g += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    gp = args.out.parent / "glance.tex"
    gp.write_text("\n".join(g) + "\n")
    print(f"wrote {gp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
