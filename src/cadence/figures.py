"""Publication figures. One function per figure; all take result JSONs / the
cached dataset and write a PDF plus a PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ---------------------------------------------------------------------------
# house style
# ---------------------------------------------------------------------------
C = {
    "cadence": "#1b4f9c",
    "shared": "#2f6fb5",
    "ridge": "#e08a1e",
    "ma": "#8c8c8c",
    "null": "#c0c0c0",
    "oracle": "#3f9b52",
    "ctrl": "#b03030",
    "ceiling": "#444444",
    "accent": "#8e44ad",
    "warm": "#c0392b",
}
ANIMAL_COLORS = ["#1b4f9c", "#2f8fb5", "#3f9b52", "#e08a1e", "#b03030", "#8e44ad"]


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / (name + '.pdf')}")


def panel_label(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Figure 1 -- concept and protocol
# ---------------------------------------------------------------------------
def fig_concept(out: Path):
    use_style()
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                          hspace=0.55, wspace=0.28)

    # --- model schematic ---
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    panel_label(ax, "a", dx=-0.02, dy=0.99)
    ax.set_title("A shared causal operator with animal-specific observation maps",
                 loc="left", pad=6)

    def box(x, y, w, h, label, color, fs=7.5, alpha=0.18):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    linewidth=1.0, edgecolor=color,
                                    facecolor=color, alpha=alpha))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#333333", style="-|>", lw=1.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, linewidth=lw,
                                     color=color, linestyle=ls,
                                     shrinkA=1, shrinkB=1))

    # latent chain
    box(1.1, 4.4, 1.5, 0.85, "$z_t$", C["cadence"], fs=10)
    box(4.3, 4.4, 1.5, 0.85, "$z_{t+1}$", C["cadence"], fs=10)
    box(7.5, 4.4, 1.5, 0.85, "$z_{t+2}$", C["cadence"], fs=10)
    arrow(2.6, 4.82, 4.3, 4.82)
    arrow(5.8, 4.82, 7.5, 4.82)
    ax.text(3.45, 5.55, "$F_{\\mathrm{shared}}$", ha="center", fontsize=8,
            color=C["cadence"])
    ax.text(6.65, 5.55, "$F_{\\mathrm{shared}}$", ha="center", fontsize=8,
            color=C["cadence"])

    # intervention
    box(3.0, 2.85, 2.0, 0.75, "intervention $a_t$", C["warm"], fs=7.5)
    arrow(4.0, 3.60, 4.6, 4.40, color=C["warm"], lw=1.3)
    ax.text(4.95, 3.95, "$G_{\\mathrm{shared}}(z_t)\\,a_t$", fontsize=8, color=C["warm"])

    # residual
    box(0.5, 2.85, 2.0, 0.75, "$F^{\\mathrm{res}}_i(z_t)$", "#7f8c8d", fs=7.5)
    arrow(1.5, 3.60, 1.75, 4.40, color="#7f8c8d", ls=(0, (2, 1.6)))

    # observation maps
    for k, (x, name) in enumerate([(0.6, "animal 1"), (3.6, "animal 2"), (6.6, "animal $i$")]):
        col = ANIMAL_COLORS[k]
        box(x, 0.5, 2.6, 1.05, f"$y_t^{{({name[-1]})}} = H_i(z_t)$\n{name}", col, fs=7)
        arrow(x + 1.3, 1.55, x + 1.3, 2.75, color=col, ls=(0, (2, 1.6)))
    ax.text(5.0, 2.28, "animal-specific observation maps", ha="center", fontsize=7,
            color="#555555", style="italic")

    # --- protocol ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 4.2)
    panel_label(ax2, "b", dx=-0.06, dy=1.0)
    ax2.set_title("Zero-shot protocol", loc="left", pad=6)
    steps = [
        ("train", "shared operator from\nperturbations in other animals", C["cadence"]),
        ("calibrate", "new animal, **unperturbed\nactivity only**", C["oracle"]),
        ("predict", "time-resolved response to an\nunseen intervention", C["warm"]),
    ]
    for i, (name, desc, col) in enumerate(steps):
        y = 3.2 - i * 1.25
        ax2.add_patch(FancyBboxPatch((0.2, y - 0.42), 9.4, 0.86,
                                     boxstyle="round,pad=0.08", linewidth=1.0,
                                     edgecolor=col, facecolor=col, alpha=0.13))
        ax2.text(0.6, y, f"{i+1}. {name}", fontsize=8, fontweight="bold",
                 va="center", color=col)
        ax2.text(3.1, y, desc.replace("**", ""), fontsize=7, va="center")

    # --- identifiability ---
    ax3 = fig.add_subplot(gs[1, 1])
    panel_label(ax3, "c", dx=-0.20, dy=1.05)
    ax3.set_title("Identifiability requires an\nasymmetric shared flow", loc="left", pad=6)
    th = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(th), np.sin(th), color=C["ctrl"], lw=1.2)
    ax3.plot(1.9 + 0.75 * np.cos(th) + 0.25 * np.cos(2 * th),
             0.55 * np.sin(th) + 0.18 * np.sin(3 * th), color=C["oracle"], lw=1.2)
    ax3.annotate("", xy=(0.35, 0.94), xytext=(-0.35, 0.94),
                 arrowprops=dict(arrowstyle="-|>", color=C["ctrl"], lw=1.0))
    ax3.text(0, -1.45, "symmetric flow\n$T$ free → transfer fails", ha="center",
             fontsize=7, color=C["ctrl"])
    ax3.text(1.9, -1.45, "generic flow\n$T=I$ → transfer identified", ha="center",
             fontsize=7, color=C["oracle"])
    ax3.set_xlim(-1.35, 3.0); ax3.set_ylim(-1.75, 1.3)
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_visible(False)
    save(fig, out, "fig1_concept")


# ---------------------------------------------------------------------------
# Figure 2 -- teacher-RNN benchmark
# ---------------------------------------------------------------------------
def _method_series(summary, key="neural.delta_r2"):
    out = {}
    for m, s in summary["methods"].items():
        if key in s:
            out[m] = (s[key]["mean"], s[key]["ci_lo"], s[key]["ci_hi"])
    return out


def fig_teacher(paths: dict[str, Path], out: Path, ident: dict[str, Path] | None = None):
    use_style()
    loaded = {}
    for regime, p in paths.items():
        if p.exists():
            loaded[regime] = json.loads(p.read_text())
    if not loaded:
        print("  (no teacher results yet, skipping fig2)")
        return
    ncol = 3
    fig, axes = plt.subplots(1, ncol, figsize=(7.2, 2.5))

    # panel a: transfer per regime
    ax = axes[0]
    panel_label(ax, "a")
    order = [r for r in ("shared", "heterogeneous", "degenerate") if r in loaded]
    labels, vals, los, his, ceils = [], [], [], [], []
    for r in order:
        s = loaded[r]["summary"]
        d = s["methods"].get("cadence", {}).get("neural.delta_r2")
        c = s["methods"].get("cadence", {}).get("ceiling.delta_r2_ceiling")
        if d is None:
            continue
        labels.append(r.replace("heterogeneous", "hetero."))
        vals.append(d["mean"]); los.append(d["mean"] - d["ci_lo"]); his.append(d["ci_hi"] - d["mean"])
        ceils.append(c["mean"] if c else np.nan)
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=[los, his], color=C["cadence"], width=0.6, capsize=2.5,
           error_kw=dict(lw=0.8))
    for i, c in enumerate(ceils):
        if np.isfinite(c):
            ax.hlines(c, i - 0.34, i + 0.34, color=C["ceiling"], ls=(0, (2, 1.5)), lw=1.0)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("$\\Delta R^2$ (held-out animal)")
    ax.set_title("Transfer depends on the regime", loc="left")
    ax.text(0.02, 0.94, "dashed = noise ceiling", transform=ax.transAxes, fontsize=6.5,
            color=C["ceiling"])

    # panel b: methods within the shared regime
    ax = axes[1]
    panel_label(ax, "b")
    ref = loaded.get("shared") or loaded[order[0]]
    ser = _method_series(ref["summary"])
    pretty = {"cadence": "CADENCE", "oracle": "oracle", "ma_latent": "aligned\ngroup mean",
              "ma_cca": "manifold\nalignment", "unit_ridge": "unit\nencoding",
              "no_effect": "no effect", "ctrl_permuted_obs": "permuted\nunits",
              "ctrl_scrambled_interv": "scrambled\nstimulus"}
    keys = [k for k in ["cadence", "oracle", "ma_latent", "ma_cca", "unit_ridge",
                        "no_effect", "ctrl_permuted_obs", "ctrl_scrambled_interv"] if k in ser]
    cols = {"cadence": C["cadence"], "oracle": C["oracle"], "no_effect": C["null"],
            "ma_latent": C["ma"], "ma_cca": C["ma"], "unit_ridge": C["ridge"],
            "ctrl_permuted_obs": C["ctrl"], "ctrl_scrambled_interv": C["ctrl"]}
    x = np.arange(len(keys))
    m = [ser[k][0] for k in keys]
    lo = [ser[k][0] - ser[k][1] for k in keys]
    hi = [ser[k][2] - ser[k][0] for k in keys]
    ax.bar(x, m, yerr=[lo, hi], color=[cols[k] for k in keys], width=0.65, capsize=2,
           error_kw=dict(lw=0.7))
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(k, k) for k in keys], rotation=42, ha="right", fontsize=6)
    ax.set_ylabel("$\\Delta R^2$")
    ax.set_title("Baselines and controls", loc="left")

    # panel c: conserved vs idiosyncratic directions
    ax = axes[2]
    panel_label(ax, "c")
    width = 0.36
    for j, r in enumerate(order):
        g = loaded[r]["summary"]["methods"].get("cadence", {})
        cons = g.get("group.group:conserved", {}).get("mean", np.nan)
        idio = g.get("group.group:idiosyncratic", {}).get("mean", np.nan)
        ax.bar(j - width / 2, cons, width, color=C["oracle"],
               label="conserved directions" if j == 0 else None)
        ax.bar(j + width / 2, idio, width, color=C["ctrl"],
               label="idiosyncratic" if j == 0 else None)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([r.replace("heterogeneous", "hetero.") for r in order])
    ax.set_ylabel("$\\Delta R^2$")
    ax.set_title("Only conserved directions transfer", loc="left")
    ax.legend(loc="best")

    fig.tight_layout()
    save(fig, out, "fig2_teacher")


# ---------------------------------------------------------------------------
# Figure 3 -- ICMS dataset and the measured causal effect
# ---------------------------------------------------------------------------
def fig_dataset(ds, audit: dict, out: Path):
    use_style()
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)
    bin_s = ds.bin_s

    ref = max(ds.sets, key=lambda s: s.n_obs)
    ypost = ref.y[:, ref.t0 :]
    base = ypost[~ref.perturbed].mean(0)
    conds = sorted({int(c) for c in ref.cond[ref.perturbed]})
    t = (np.arange(ypost.shape[1])) * bin_s

    # a: population response by amplitude at one contact
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a")
    chans = {}
    for c in conds:
        a = ref.meta["cond_amp"][c] if c in ref.meta["cond_amp"] else ref.meta["cond_amp"][str(c)]
        ch = ref.meta["cond_channel"][c] if c in ref.meta["cond_channel"] else ref.meta["cond_channel"][str(c)]
        chans.setdefault(int(ch), []).append((float(a), c))
    ch0 = max(chans, key=lambda k: len(chans[k]))
    series = sorted(chans[ch0])
    cmap = plt.get_cmap("viridis")
    for i, (a, c) in enumerate(series):
        d = (ypost[ref.cond == c].mean(0) - base).mean(1)
        ax.plot(t, d, color=cmap(i / max(len(series) - 1, 1)), label=f"{a:g} µA")
    ax.axvspan(0, 0.7, color="#ffe08a", alpha=0.45, lw=0, zorder=0)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xlabel("time from stimulation onset (s)")
    ax.set_ylabel("$\\Delta$ rate (spikes/bin/unit)")
    ax.set_title(f"Graded response\n{ref.animal.replace('sub-','')}, contact {ch0}", loc="left")
    ax.legend(fontsize=6, ncol=2)

    # b: is the effect organised around the stimulating contact? (pooled)
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b")
    stim_bins = int(round(0.7 / bin_s))
    dz_all, d_all = [], []
    for s in ds.sets:
        yp = s.y[:, s.t0 :]
        b = yp[~s.perturbed].mean(0)
        uy = np.asarray(s.meta["unit_y_um"], float)
        for c in np.unique(s.cond[s.perturbed]):
            c = int(c)
            dep = s.meta["cond_depth_um"].get(c, s.meta["cond_depth_um"].get(str(c)))
            d = (yp[s.cond == c].mean(0) - b)[:stim_bins].mean(0)
            dz_all.append(uy - float(dep)); d_all.append(d)
    dz = np.concatenate(dz_all); dd = np.concatenate(d_all)
    edges = np.arange(-1700, 1701, 200)
    cx, cy, ce_ = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dz >= lo) & (dz < hi)
        if m.sum() < 20:
            continue
        cx.append((lo + hi) / 2); cy.append(dd[m].mean())
        ce_.append(dd[m].std() / np.sqrt(m.sum()))
    ax.errorbar(cx, cy, yerr=ce_, marker="o", ms=2.5, color=C["cadence"], lw=1.0, capsize=1.5)
    ax.axvline(0, color=C["warm"], ls=(0, (3, 2)), lw=1.0)
    ax.text(0, ax.get_ylim()[1], " contact", color=C["warm"], fontsize=6, va="top")
    ax.axhline(0, color="#999999", lw=0.7)
    r = np.corrcoef(np.abs(dz), dd)[0, 1]
    ax.set_xlabel("unit depth − contact depth (µm)")
    ax.set_ylabel("$\\Delta$ rate during stim")
    ax.set_title(f"Effect is spatially diffuse\n$r(|\\Delta z|,\\Delta$rate$)={r:+.3f}$", loc="left")

    # c: within-animal dose-response
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c")
    for i, an in enumerate(ds.animals):
        rows: dict[float, list[float]] = {}
        for s in ds.sets:
            if s.animal != an:
                continue
            yp = s.y[:, s.t0 :]
            b = yp[~s.perturbed].mean(0)
            for c in np.unique(s.cond[s.perturbed]):
                c = int(c)
                a = s.meta["cond_amp"].get(c, s.meta["cond_amp"].get(str(c)))
                rows.setdefault(float(a), []).append(
                    float((yp[s.cond == c].mean(0) - b)[:stim_bins].mean())
                )
        xs = sorted(rows)
        ax.plot(xs, [np.mean(rows[a]) for a in xs], "o-", ms=2.5, lw=1.0,
                color=ANIMAL_COLORS[i % len(ANIMAL_COLORS)],
                label=an.replace("sub-ICMS", "m"))
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xlabel("amplitude (µA)")
    ax.set_ylabel("mean $\\Delta$ rate")
    ax.set_title("Neural dose–response saturates", loc="left")
    ax.legend(fontsize=5.5, ncol=2)

    # d: behavioural detection curves by amplitude
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "d")
    for i, (a, c) in enumerate(series):
        if ref.behavior is None:
            break
        d = ref.behavior[:, ref.t0 :, 2][ref.cond == c].mean(0)
        ax.plot(t, d, color=cmap(i / max(len(series) - 1, 1)))
    if ref.behavior is not None:
        ax.plot(t, ref.behavior[:, ref.t0 :, 2][~ref.perturbed].mean(0), color="#888888",
                ls=(0, (3, 2)), label="no stim")
    ax.axvspan(0, 0.7, color="#ffe08a", alpha=0.45, lw=0, zorder=0)
    ax.set_xlabel("time from stimulation onset (s)")
    ax.set_ylabel("P(reported by $t$)")
    ax.set_title("Behavioural response", loc="left")
    ax.legend(fontsize=6)

    # e: per-session noise ceilings
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "e")
    ce = [v["ceiling"] for v in audit["sets"].values() if v.get("ceiling")]
    cb = [v["ceiling_behavior"] for v in audit["sets"].values() if v.get("ceiling_behavior")]
    ax.hist(ce, bins=np.linspace(0.5, 1.0, 18), color=C["cadence"], alpha=0.8, label="units")
    ax.hist(cb, bins=np.linspace(0.5, 1.0, 18), color=C["oracle"], alpha=0.6, label="behaviour")
    ax.set_xlabel("split-half $\\Delta R^2$ ceiling")
    ax.set_ylabel("sessions")
    ax.set_title("Effects are well estimated", loc="left")
    ax.legend(fontsize=6)

    # f: dataset composition
    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, "f")
    animals = ds.animals
    nses = [sum(1 for s in ds.sets if s.animal == a) for a in animals]
    nun = [sum(s.n_obs for s in ds.sets if s.animal == a) for a in animals]
    x = np.arange(len(animals))
    ax.bar(x - 0.2, nses, 0.38, color=C["cadence"], label="sessions")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, nun, 0.38, color=C["ridge"], label="units")
    ax2.spines["right"].set_visible(True)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("sub-ICMS", "m") for a in animals], rotation=35, ha="right")
    ax.set_ylabel("sessions", color=C["cadence"])
    ax2.set_ylabel("units", color=C["ridge"])
    ax.set_title("Cohort", loc="left")

    save(fig, out, "fig3_dataset")


# ---------------------------------------------------------------------------
# Figure 4 -- the generalisation ladder
# ---------------------------------------------------------------------------
LEVEL_LABEL = {
    "unit": "single units",
    "depth_band": "depth bands",
    "population": "population rate",
    "wheel_speed": "wheel speed",
    "detection_prob": "detection prob.",
}
GEN_LABEL = {
    "in_sample": "in-sample",
    "cross_session": "new session\n(same animal)",
    "cross_animal": "new animal",
    "cross_animal_unseen_amp": "new animal +\nunseen amplitude",
}


def fig_ladder(ladder: dict, out: Path):
    use_style()
    res = ladder["results"]
    levels = [k for k in LEVEL_LABEL if any(r["level"] == k for r in res.values())]
    gens = list(GEN_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw=dict(width_ratios=[1.55, 1.0]))

    ax = axes[0]; panel_label(ax, "a")
    w = 0.2
    for gi, g in enumerate(gens):
        m, lo, hi = [], [], []
        for lv in levels:
            r = res.get(f"{lv}|{g}|shared_operator")
            if r is None:
                m.append(np.nan); lo.append(0); hi.append(0); continue
            m.append(r["delta_r2"]); lo.append(r["delta_r2"] - r["ci"][0])
            hi.append(r["ci"][1] - r["delta_r2"])
        x = np.arange(len(levels)) + (gi - 1.5) * w
        shade = 0.25 + 0.25 * gi
        ax.bar(x, m, w, yerr=[lo, hi], capsize=1.5,
               color=plt.get_cmap("viridis")(1 - shade), label=GEN_LABEL[g].replace("\n", " "),
               error_kw=dict(lw=0.6))
    for li, lv in enumerate(levels):
        r = res.get(f"{lv}|cross_animal|shared_operator")
        if r:
            ax.hlines(r["ceiling"], li - 0.45, li + 0.45, color=C["ceiling"],
                      ls=(0, (2, 1.5)), lw=1.0)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([LEVEL_LABEL[lv] for lv in levels], rotation=20, ha="right")
    ax.set_ylabel("$\\Delta R^2$")
    ax.set_title("Causal-response transfer by readout and generalisation level", loc="left")
    ax.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.55))

    ax = axes[1]; panel_label(ax, "b")
    lv = "detection_prob" if any(r["level"] == "detection_prob" for r in res.values()) else levels[-1]
    r = res.get(f"{lv}|cross_animal|shared_operator")
    if r:
        per = r["per_animal"]
        names = list(per)
        vals = [per[k] for k in names]
        ax.bar(np.arange(len(names)), vals,
               color=[ANIMAL_COLORS[i % len(ANIMAL_COLORS)] for i in range(len(names))],
               width=0.62)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.hlines(r["ceiling"], -0.5, len(names) - 0.5, color=C["ceiling"],
                  ls=(0, (2, 1.5)), lw=1.0)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([n.replace("sub-ICMS", "m") for n in names])
        ax.set_ylabel("$\\Delta R^2$")
        ax.set_title(f"Every held-out animal\n({LEVEL_LABEL[lv]})", loc="left")
    fig.tight_layout()
    save(fig, out, "fig4_ladder")


# ---------------------------------------------------------------------------
# Figure 5 -- behavioural prediction traces
# ---------------------------------------------------------------------------
def fig_behavior(ds, traces: dict, out: Path):
    use_style()
    animals = ds.animals
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(animals):
        ax = axes.flat[i]
        tr = traces.get(a)
        if not tr:
            ax.axis("off"); continue
        t = np.asarray(tr["t"])
        amps = tr["amps"]
        for j, amp in enumerate(amps):
            col = cmap(j / max(len(amps) - 1, 1))
            ax.plot(t, tr["measured"][j], color=col, lw=1.3)
            ax.plot(t, tr["predicted"][j], color=col, lw=1.0, ls=(0, (2.4, 1.6)))
        ax.axvspan(0, 0.7, color="#ffe08a", alpha=0.4, lw=0, zorder=0)
        ax.set_title(f"{a.replace('sub-ICMS','mouse ')}  $\\Delta R^2$={tr['delta_r2']:+.2f}",
                     loc="left", fontsize=7.5)
        if i == 0:
            ax.plot([], [], color="#333333", lw=1.3, label="measured")
            ax.plot([], [], color="#333333", lw=1.0, ls=(0, (2.4, 1.6)), label="predicted")
            ax.legend(fontsize=6, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("time from stim onset (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("$\\Delta$ P(reported)")
    fig.suptitle("Zero-shot prediction of the time-resolved behavioural response "
                 "in a held-out animal", fontsize=8.5, fontweight="bold", y=1.005)
    fig.tight_layout()
    save(fig, out, "fig5_behavior")


# ---------------------------------------------------------------------------
# Figure 6 -- the recovered shared operator
# ---------------------------------------------------------------------------
def fig_operator(kernel: dict, out: Path):
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    ax = axes[0]; panel_label(ax, "a")
    K = np.asarray(kernel["depth_time"])
    ext = kernel["extent"]
    im = ax.imshow(K, aspect="auto", origin="lower", cmap="RdBu_r",
                   extent=ext, vmin=-np.abs(K).max(), vmax=np.abs(K).max())
    ax.axhline(0, color="k", lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("time from stim onset (s)")
    ax.set_ylabel("unit depth − contact depth (µm)")
    ax.set_title("Shared drive kernel", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    ax = axes[1]; panel_label(ax, "b")
    ax.plot(kernel["amp_grid"], kernel["amp_gain"], color=C["cadence"], marker="o", ms=2.5)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xlabel("amplitude (µA)")
    ax.set_ylabel("operator gain (a.u.)")
    ax.set_title("Recovered dose function", loc="left")

    ax = axes[2]; panel_label(ax, "c")
    lab = kernel["consistency_labels"]
    val = kernel["consistency"]
    ax.bar(np.arange(len(lab)), val, color=C["shared"], width=0.6)
    ax.set_xticks(np.arange(len(lab)))
    ax.set_xticklabels(lab, rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel("cosine between animals")
    ax.set_ylim(0, 1)
    ax.set_title("Operator consistency", loc="left")
    fig.tight_layout()
    save(fig, out, "fig6_operator")
