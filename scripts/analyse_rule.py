"""What is the shared rule, stated in one sentence?

The operator transfers, so something about a neuron's ordinary activity must predict
how a perturbation will move it, and the same something must hold in every animal. This
asks what it is, without fitting anything, by correlating the individual part of a
neuron's measured response with three properties measured from its control trials:

  * how fast it fires,
  * how much its firing grows over the delay, which is its preparatory ramp,
  * how strongly it distinguishes the two trial types, which is its selectivity.

Every correlation is computed inside one recording, so nothing is pooled across animals
except the correlations themselves, and each animal contributes one number to an exact
sign test. The control trials that supply the properties are disjoint from the ones that
define the response, so a correlation cannot arise from shared noise. A null column
repeats the whole thing with the perturbed trials replaced by control trials.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from cadence import individuality as I
from cadence import metrics as M

warnings.filterwarnings("ignore")


def properties(s, feat_idx) -> dict[str, np.ndarray]:
    """Per-neuron descriptions, all from the half of the control trials the response
    is not measured against."""
    yc = s.y[feat_idx][:, s.t0 :]
    rate = np.nan_to_num(np.nanmean(yc, (0, 1)))
    pre = np.nan_to_num(np.nanmean(s.y[feat_idx][:, : max(s.t0, 1)], (0, 1)))
    ramp = np.nan_to_num(np.nanmean(yc, 0)).mean(0) - pre
    sel = np.zeros_like(rate)
    if s.behavior is not None:
        ch = s.behavior[feat_idx][:, 0, 0]
        fin = np.isfinite(ch)
        if fin.any() and np.all(np.isin(ch[fin], (0.0, 1.0))):
            L, R = ch == 0, ch == 1
            if L.sum() >= 8 and R.sum() >= 8:
                sel = np.abs(np.nan_to_num(
                    np.nanmean(yc[R], (0,)) - np.nanmean(yc[L], (0,))).mean(0))
    return {"firing rate": rate, "preparatory ramp": ramp, "selectivity": sel}


def response(s, feat_idx, base_idx, null=False):
    """The individual part of each neuron's response, averaged over the window, one
    value per (neuron, condition)."""
    Y = s.y[:, s.t0 :]
    if null:
        h = len(base_idx) // 2
        rng = np.random.default_rng(abs(hash(s.key)) % (2 ** 31))
        sh = rng.permutation(base_idx)
        src_all, base_idx = np.sort(sh[:h]), np.sort(sh[h:])
    base = np.nanmean(Y[base_idx], 0)
    out = []
    for c in np.unique(s.cond[s.perturbed]):
        m = np.flatnonzero(s.cond == c)
        if len(m) < 6:
            continue
        src = src_all if null else m
        d = I.centre(np.nanmean(Y[src], 0) - base)
        out.append(np.nan_to_num(d).mean(0))                  # (n_obs,)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, nargs="+",
                    default=[Path("data/proc/alm.pkl"), Path("data/proc/alm_wide.pkl")])
    ap.add_argument("--tag", default="almall")
    ap.add_argument("--min-units", type=int, default=8)
    args = ap.parse_args()

    ds = pickle.load(args.cache[0].open("rb"))["dataset"]
    for c in args.cache[1:]:
        ds.sets = list(ds.sets) + list(pickle.load(c.open("rb"))["dataset"].sets)
    print(f"{len(ds.sets)} sessions, {len(ds.animals)} animals")

    names = ["firing rate", "preparatory ramp", "selectivity",
             "selectivity given rate"]
    per: dict[str, dict[str, list]] = {n: {} for n in names}
    per_null: dict[str, dict[str, list]] = {n: {} for n in names}
    rate_sel = []
    for s in ds.sets:
        if s.n_obs < args.min_units:
            continue
        feat_idx, base_idx = I.control_split(s)
        props = properties(s, feat_idx)
        rate, sel = props["firing rate"], props["selectivity"]
        if np.std(rate) > 1e-9 and np.std(sel) > 1e-9:
            rate_sel.append(float(np.corrcoef(rate, sel)[0, 1]))
        for null, store in ((False, per), (True, per_null)):
            for d in response(s, feat_idx, base_idx, null=null):
                for n in names:
                    if n == "selectivity given rate":
                        continue
                    x = props[n]
                    if np.std(x) < 1e-9 or np.std(d) < 1e-9:
                        continue
                    store[n].setdefault(s.animal, []).append(
                        float(np.corrcoef(x, d)[0, 1]))
                # Selectivity is measured in spikes, so a fast neuron has a large
                # one whatever its tuning, and the two are correlated across cells.
                # Regressing firing rate out of both asks whether selectivity says
                # anything the rate does not.
                if min(np.std(rate), np.std(sel), np.std(d)) > 1e-9:
                    def resid(a, b):
                        b = b - b.mean()
                        a = a - a.mean()
                        return a - b * float(np.dot(a, b)) / float(np.dot(b, b))
                    sr, dr = resid(sel, rate), resid(d, rate)
                    if min(np.std(sr), np.std(dr)) > 1e-9:
                        store["selectivity given rate"].setdefault(
                            s.animal, []).append(float(np.corrcoef(sr, dr)[0, 1]))

    rep = {}
    if rate_sel:
        rep["rate_vs_selectivity"] = dict(
            mean_r=float(np.mean(rate_sel)), n=len(rate_sel),
            n_positive=int(np.sum(np.array(rate_sel) > 0)))
        print(f"\nselectivity against firing rate across recordings: "
              f"r = {np.mean(rate_sel):+.3f}, positive in "
              f"{int(np.sum(np.array(rate_sel) > 0))}/{len(rate_sel)}")
    print(f"\n{'property':26s} {'r':>7s} {'animals<0':>10s} {'sign p':>8s} "
          f"{'null r':>8s} {'null<0':>8s}")
    print("-" * 68)
    for n in names:
        an = sorted(per[n])
        v = np.array([float(np.nanmean(per[n][a])) for a in an])
        vn = np.array([float(np.nanmean(per_null[n].get(a, [np.nan]))) for a in an])
        st = M.animal_sign_test(list(-v))          # is the correlation negative?
        rep[n] = dict(mean_r=float(np.nanmean(v)), n=len(v),
                      n_negative=int((v < 0).sum()), p=st["p"],
                      null_mean_r=float(np.nanmean(vn)),
                      null_negative=int(np.nansum(vn < 0)),
                      per_animal={a: float(np.nanmean(per[n][a])) for a in an})
        print(f"{n:26s} {np.nanmean(v):+7.3f} {int((v < 0).sum()):>4d}/{len(v):<4d} "
              f"{st['p']:8.3f} {np.nanmean(vn):+8.3f} "
              f"{int(np.nansum(vn < 0)):>4d}/{len(vn):<4d}")

    # --- can the rule alone predict, with nothing else in it? -----------------
    # A minimal shared operator: the individual part of a neuron's response is a
    # time course times its selectivity, plus a time course times its firing rate.
    # Twelve shared numbers in total, fitted leave-one-animal-out, no position, no
    # dose dependence, no network. If this recovers a useful share of what the full
    # operator recovers, then the rule really is the whole of it.
    ex = []
    for s in ds.sets:
        if s.n_obs < args.min_units:
            continue
        feat_idx, base_idx = I.control_split(s)
        props = properties(s, feat_idx)
        Y = s.y[:, s.t0 :]
        base = np.nanmean(Y[base_idx], 0)
        T = Y.shape[1]
        B = I.raised_cosine(T, 6)
        z = []
        for n in ("selectivity", "firing rate"):
            x = props[n].astype(float)
            sd = np.std(x)
            z.append((x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x))
        for c in np.unique(s.cond[s.perturbed]):
            m = np.flatnonzero(s.cond == c)
            if len(m) < 6:
                continue
            y = I.centre(np.nanmean(Y[m], 0) - base)
            sc = max(float(np.nanstd(Y[feat_idx].reshape(-1, s.n_obs), 0).mean()), 1e-3)
            X = np.einsum("bt,pn->tnbp", B, np.stack(z)).reshape(T, s.n_obs, -1)
            ex.append(dict(animal=s.animal, X=np.nan_to_num(X / 1.0).astype(np.float32),
                           y=(np.nan_to_num(y) / sc).astype(np.float32)))
    if ex:
        K = ex[0]["X"].shape[-1]
        blk = []
        for e in ex:
            Xf = e["X"].reshape(-1, K).astype(np.float64)
            yf = e["y"].ravel().astype(np.float64)
            w = 1.0 / max(len(yf), 1)
            blk.append((w * Xf.T @ Xf, w * Xf.T @ yf))
        an = sorted({e["animal"] for e in ex})
        vals = []
        for a in an:
            XX = sum(b[0] for e, b in zip(ex, blk) if e["animal"] != a)
            Xy = sum(b[1] for e, b in zip(ex, blk) if e["animal"] != a)
            th = np.linalg.solve(XX + 1.0 * (np.trace(XX) / K) * np.eye(K), Xy)
            n = de = 0.0
            for e in ex:
                if e["animal"] != a:
                    continue
                n += float(np.nansum((e["y"] - e["X"] @ th.astype(np.float32)) ** 2))
                de += float(np.nansum(e["y"] ** 2))
            vals.append(1.0 - n / de if de > 0 else np.nan)
        v = np.array(vals)
        st = M.animal_sign_test(list(v))
        rep["rule_only_operator"] = dict(
            n_params=K, mean=float(np.nanmean(v)), median=float(np.nanmedian(v)),
            n_positive=int((v > 0).sum()), n=len(v), p=st["p"],
            pooled=None, per_animal={a: float(x) for a, x in zip(an, v)})
        print(f"\nthe rule on its own, {K} shared numbers, leave one animal out: "
              f"mean {np.nanmean(v):+.4f}, median {np.nanmedian(v):+.4f}, "
              f"{int((v > 0).sum())}/{len(v)} animals above zero, p = {st['p']:.4f}")

    out = Path(f"results/rule_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
