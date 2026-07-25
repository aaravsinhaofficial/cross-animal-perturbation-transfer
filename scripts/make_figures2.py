"""Figures for the shared-operator results.

Each panel is drawn from a results file written by one of the analysis scripts, so
the figures cannot drift away from the numbers in the text.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

BLUE, ORANGE, GREY, GREEN = "#2a6ebb", "#e07b39", "#8a8a8a", "#3f9a53"
plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.dpi": 200,
})


def save(fig, out: Path, name: str) -> None:
    """Both formats: the paper uses the vector version, the web page the raster."""
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def panel_per_animal(ax, res, key_a, key_b, label_a, label_b, title):
    """Every animal as one point: baseline on x, model on y."""
    pa = res[key_a]["per_animal"]
    pb = res[key_b]["per_animal"]
    ks = [k for k in pa if k in pb]
    x = np.array([pb[k] for k in ks])
    y = np.array([pa[k] for k in ks])
    lo = min(x.min(), y.min()) - 0.05
    hi = max(x.max(), y.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color=GREY, lw=0.5, zorder=0)
    ax.axvline(0, color=GREY, lw=0.5, zorder=0)
    win = y > x
    ax.scatter(x[win], y[win], s=16, color=BLUE, zorder=3, label="operator better")
    ax.scatter(x[~win], y[~win], s=16, color=ORANGE, zorder=3, label="baseline better")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(label_b); ax.set_ylabel(label_a)
    ax.set_title(f"{title}\n{int(win.sum())} of {len(ks)} animals above the line",
                 fontsize=8)
    ax.legend(loc="lower right", fontsize=6)


def panel_delta(ax, res, title, xlim=None):
    """One dot per animal, for each model, on the individual part of the response.

    The stereotype is at exactly zero by construction, and the null row is the same
    analysis with no effect present, so the question the figure asks is whether the
    operator row sits to the right of both.
    """
    rows = [("stereotype", "stereotype", GREY),
            ("no effect present", "no effect present", GREY),
            ("shared_operator", "shared operator", BLUE),
            ("learned_operator", "learned operator", GREEN)]
    rows = [r for r in rows if r[0] in res]
    for i, (k, lab, col) in enumerate(rows):
        v = np.array(list(res[k]["per_animal"].values()), float)
        v = v[np.isfinite(v)]
        jit = (np.arange(len(v)) % 5 - 2) * 0.045
        ax.scatter(v, np.full(len(v), i) + jit, s=11, color=col, alpha=0.75,
                   zorder=3, linewidths=0)
        med = float(np.median(v))
        ax.plot([med, med], [i - 0.3, i + 0.3], color=col, lw=2.2, zorder=4)
        st = res[k].get("sign_test", {})
        if st and k not in ("stereotype",):
            ax.text(0.985, i + 0.3,
                    f"{st['n_positive']}/{st['n']}, p = {st['p']:.3f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=6, color=col)
    ax.axvline(0, color="k", lw=0.7, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("$\\Delta R^2$ on the individual part")
    ceil = res.get("ceiling", {}).get("animal_mean")
    ax.set_title(f"{title}\nmeasurable maximum {ceil:.2f}" if ceil else title,
                 fontsize=8)


def panel_scaling(ax, curve, title):
    n = np.array([c["n_animals"] for c in curve])
    m = np.array([c["delta_r2"] for c in curve])
    s = np.array([c["sem"] for c in curve])
    ax.fill_between(n, m - s, m + s, color=BLUE, alpha=0.2, lw=0)
    ax.plot(n, m, "-o", color=BLUE, ms=3, lw=1.2)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xscale("log")
    show = [v for v in n if v in (1, 2, 3, 5, 8, 12, 19, 20)]
    ax.set_xticks(show); ax.set_xticklabels([str(v) for v in show])
    ax.minorticks_off()
    ax.set_xlabel("animals the operator was fitted on")
    ax.set_ylabel("$\\Delta R^2$ on the individual part")
    ax.set_title(title, fontsize=8)


def panel_sweep(ax, rows, title, real=None):
    """The simulated cortex, with the two real cohorts placed on the same axis."""
    for gcv, col, lab in sorted({r.get("gain_cv", 0.3) for r in rows},
                                reverse=True) and [
            (g, c, l) for g, c, l in
            ((0.3, BLUE, "cells differ in responsiveness"),
             (0.0, GREEN, "cells equally responsive"))
            if any(abs(r.get("gain_cv", 0.3) - g) < 1e-9 for r in rows)]:
        r = sorted([x for x in rows if abs(x.get("gain_cv", 0.3) - gcv) < 1e-9],
                   key=lambda x: x["private"])
        ax.plot([x["private"] for x in r], [x["fraction"] for x in r],
                "-o", color=col, ms=3, lw=1.2, label=lab)
    for name, val, col in (real or []):
        ax.axhline(val, color=col, lw=1.0, ls="--")
        ax.text(0.02, val, f" {name}", color=col, fontsize=6, va="bottom")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlabel("how much of the recruitment belongs to one implant")
    ax.set_ylabel("fraction of the individual part recovered")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="best")


def panel_rule(ax, rule, rule_icms, title):
    """What predicts how much a perturbation moves a neuron, one dot per animal."""
    names = [("selectivity", "how choice\nselective it is"),
             ("firing rate", "how fast\nit fires"),
             ("preparatory ramp", "how much\nit ramps")]
    names = [(k, lab) for k, lab in names if k in rule]
    for i, (k, lab) in enumerate(names):
        v = np.array(list(rule[k]["per_animal"].values()), float)
        v = v[np.isfinite(v)]
        jit = (np.arange(len(v)) % 5 - 2) * 0.05
        col = BLUE if rule[k]["p"] < 0.05 else GREY
        ax.scatter(v, np.full(len(v), i) + jit, s=11, color=col, alpha=0.75,
                   zorder=3, linewidths=0)
        m = float(np.median(v))
        ax.plot([m, m], [i - 0.3, i + 0.3], color=col, lw=2.2, zorder=4)
        ax.text(0.985, i + 0.33, f"{rule[k]['n_negative']}/{rule[k]['n']}, "
                f"p = {rule[k]['p']:.1g}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6, color=col)
    if rule_icms and "firing rate" in rule_icms:
        v = np.array(list(rule_icms["firing rate"]["per_animal"].values()), float)
        ax.scatter(v, np.full(len(v), len(names)), s=13, color=ORANGE, zorder=3,
                   linewidths=0)
        m = float(np.median(v))
        ax.plot([m, m], [len(names) - 0.3, len(names) + 0.3], color=ORANGE, lw=2.2)
        ax.text(0.985, len(names) + 0.33,
                f"{rule_icms['firing rate']['n_negative']}/"
                f"{rule_icms['firing rate']['n']}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6, color=ORANGE)
        names = names + [("icms", "how fast it fires\n(under current)")]
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([lab for _, lab in names], fontsize=7)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("correlation with how much the perturbation moves the neuron")
    ax.set_title(title, fontsize=8)


def panel_quality(ax, detail, title):
    """Does what transfers track how well the animal was measured?"""
    # the shared operator, so the figure and the number quoted in the text are the
    # same model
    ks = list(detail)
    c = np.array([detail[k]["ceiling"] for k in ks])
    v = np.array([detail[k]["shared"] for k in ks], float)
    ok = np.isfinite(c) & np.isfinite(v)
    c, v = c[ok], v[ok]
    ax.scatter(c, v, s=16, color=BLUE)
    if len(c) > 3:
        from scipy.stats import spearmanr
        r, p = spearmanr(c, v)
        b = np.polyfit(c, v, 1)
        xs = np.linspace(c.min(), c.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.0)
        pt = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
        ax.set_title(f"{title}\nrank correlation {r:+.2f}, p = {pt}", fontsize=8)
    else:
        ax.set_title(title, fontsize=8)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlabel("how measurable the individual part is")
    ax.set_ylabel("$\\Delta R^2$ on the individual part")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    R = Path("results")
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- the main zero-shot result -------------------------------------------
    alm = load(R / "operator_alm5.json")
    icms = load(R / "operator_icms5.json")
    if alm:
        fig, axes = plt.subplots(1, 2 if icms else 1, figsize=(6.8 if icms else 3.4, 3.0))
        axes = np.atleast_1d(axes)
        key = "blend" if "blend" in alm else "operator"
        panel_per_animal(axes[0], alm, key, "group",
                         "shared operator", "stereotype from other animals",
                         "photoinhibition of frontal cortex, 20 mice")
        if icms:
            k2 = "blend" if "blend" in icms else "operator"
            panel_per_animal(axes[1], icms, k2, "group",
                             "shared operator", "stereotype from other animals",
                             "microstimulation of somatosensory cortex, 6 mice")
        fig.tight_layout()
        save(fig, args.out, "fig8_zeroshot")
        plt.close(fig)
        print("wrote fig8_zeroshot.png")

    # ---- the decomposition ----------------------------------------------------
    ia = load(R / "individuality_alm.json")
    ii = load(R / "individuality_icms.json")
    scal = load(R / "cohort_scaling_alm.json")
    if ia:
        fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.4))
        lim = (-0.5, 0.42)
        panel_delta(axes[0, 0], ia, "light in frontal cortex, 20 mice", lim)
        if ii:
            panel_delta(axes[0, 1], ii, "current in somatosensory cortex, 6 mice",
                        lim)
        panel_quality(axes[1, 0], ia["per_animal_detail"],
                      "what transfers versus what is measurable")
        if scal:
            panel_scaling(axes[1, 1], scal, "more animals, better operator")
        fig.tight_layout()
        save(fig, args.out, "fig9_decomposition")
        plt.close(fig)
        print("wrote fig9_decomposition.png")

    # ---- what the shared rule is ----------------------------------------------
    rule = load(R / "rule_almall.json")
    if rule:
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        panel_rule(ax, rule, load(R / "rule_icms.json"),
                   "what predicts how much a perturbation moves a neuron")
        fig.tight_layout()
        save(fig, args.out, "fig11_rule")
        plt.close(fig)
        print("wrote fig11_rule.png")

    # ---- the simulation -------------------------------------------------------
    sw = load(R / "decomposition_sweep.json")
    if sw:
        real = []
        for tag, lab, col in (("alm", "under light", BLUE),
                              ("icms", "under current", ORANGE)):
            r = load(R / f"individuality_{tag}.json")
            if r and "shared_operator" in r:
                f = r["shared_operator"]["animal_mean"] / max(
                    r["ceiling"]["animal_mean"], 1e-9)
                real.append((lab, f, col))
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        panel_sweep(ax, sw, "what the split means, checked against truth", real)
        fig.tight_layout()
        save(fig, args.out, "fig10_sweep")
        plt.close(fig)
        print("wrote fig10_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
