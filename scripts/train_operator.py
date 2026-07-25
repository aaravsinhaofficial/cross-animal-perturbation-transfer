"""Train the learned cross-animal perturbation operator, leave-one-animal-out.

For each held-out animal the model is trained on the others, with one further
animal held out from training as a validation set for early stopping, so nothing
about the test animal influences the fit or the stopping point. The test animal
contributes only its unperturbed activity and the stimulus settings.

Scored against the same baselines as everything else, with animal-level statistics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import torch

from cadence import metrics as M
from cadence import neural_operator as NO

warnings.filterwarnings("ignore")


def make_measured():
    cache: dict = {}

    def measured(s, conds):
        if s.key not in cache:
            Y = s.y[:, s.t0 :]
            base = np.nanmean(Y[~s.perturbed], 0)
            cache[s.key] = {int(c): np.nanmean(Y[s.cond == c], 0) - base
                            for c in np.unique(s.cond[s.perturbed])}
        return {c: cache[s.key][c] for c in conds if c in cache[s.key]}

    return measured


def group_average(train, s, conds, measured):
    by: dict[float, list] = {}
    for t in train:
        cs = [int(c) for c in np.unique(t.cond[t.perturbed])]
        for c, D in measured(t, cs).items():
            by.setdefault(round(float(t.meta["cond_amp"][c]), 3), []).append(
                np.nanmean(D, 1))
    if not by:
        return {}
    amps = np.array(sorted(by))
    stack = np.stack([np.nanmean(by[a], 0) for a in amps])
    out = {}
    for c in conds:
        a = float(s.meta["cond_amp"][c])
        cv = (np.stack([np.interp(a, amps, stack[:, t]) for t in range(stack.shape[1])])
              if len(amps) > 1 else stack[0])
        out[c] = np.tile(cv[:, None], (1, s.n_obs))
    return out


def evaluate(model, ex, cfg, device):
    """Per (session, condition) predictions, grouped back into per-session dicts."""
    model.eval()
    out: dict[str, dict[int, np.ndarray]] = {}
    with torch.no_grad():
        for i in range(0, len(ex), 16):
            b = ex[i : i + 16]
            neu, stim, rel, ctrl, tgt, mask = NO.collate(b, device)
            pred = model(neu, stim, rel, ctrl, mask).cpu().numpy()
            for j, e in enumerate(b):
                # undo the per-recording normalisation
                out.setdefault(e["key"], {})[e["cond"]] = pred[j, : e["n"]].T * e["scale"]
    return out


def run_fold(ds, held, cfg, measured, device, seed=0):
    animals = [a for a in ds.animals if a != held]
    # The validation animal depends only on which animal is held out, so every seed
    # sees the same one. That is what makes cross-seed disagreement measured on the
    # validation animal comparable with disagreement measured on the test animal.
    val_animal = animals[zlib.crc32(held.encode()) % len(animals)]
    rng = np.random.default_rng(seed)
    tr_sets = [s for s in ds.sets if s.animal not in (held, val_animal)]
    va_sets = [s for s in ds.sets if s.animal == val_animal]
    te_sets = [s for s in ds.sets if s.animal == held]
    if not tr_sets or not te_sets:
        return None

    tr = NO.pack_fold(tr_sets, cfg, measured, device)
    va = NO.pack_fold(va_sets, cfg, measured, device)
    te = NO.pack_fold(te_sets, cfg, measured, device)
    if not tr or not te:
        return None

    torch.manual_seed(seed)
    T = tr[0]["tgt"].shape[1]
    model = NO.NeuralOperator(cfg, tr[0]["neu"].shape[1], tr[0]["stim"].shape[0],
                              tr[0]["rel"].shape[1], T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best, best_state, bad = np.inf, None, 0
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), cfg.batch_sessions):
            b = [tr[j] for j in order[i : i + cfg.batch_sessions]]
            neu, stim, rel, ctrl, tgt, mask = NO.collate(b, device)
            loss = NO.masked_loss(model(neu, stim, rel, ctrl, mask), tgt, mask,
                                  cfg.huber_delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        if va:
            model.eval()
            with torch.no_grad():
                vl, vn = 0.0, 0
                for i in range(0, len(va), 16):
                    b = va[i : i + 16]
                    neu, stim, rel, ctrl, tgt, mask = NO.collate(b, device)
                    vl += float(NO.masked_loss(model(neu, stim, rel, ctrl, mask),
                                               tgt, mask, cfg.huber_delta)) * len(b)
                    vn += len(b)
                vl /= max(vn, 1)
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, te, te_sets, va, va_sets, val_animal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, nargs="+", default=[Path("data/proc/alm.pkl")])
    ap.add_argument("--tag", default="alm")
    ap.add_argument("--min-ceiling", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=260)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"results/operator_{args.tag}.json")
    cfg = NO.OperatorConfig(epochs=args.epochs, d_model=args.d_model,
                            n_layers=args.n_layers, lr=args.lr, device=args.device)
    measured = make_measured()

    sets = []
    for c in args.cache:
        d = pickle.load(c.open("rb"))["dataset"]
        sets += d.sets
    class DS:  # noqa: N801
        pass
    ds = DS()
    ds.sets = sets
    if args.min_ceiling > 0:
        keep = []
        for s in ds.sets:
            cs = [int(x) for x in np.unique(s.cond[s.perturbed])]
            if not cs:
                continue
            y = np.nan_to_num(s.y[:, s.t0 :])
            k = np.isin(s.cond, cs) | (~s.perturbed)
            if M.noise_ceiling(y[k], s.cond[k], s.perturbed[k],
                               n_splits=80)["delta_r2_ceiling"] >= args.min_ceiling:
                keep.append(s)
        ds.sets = keep
    seen, ds.animals = set(), []
    for s in ds.sets:
        if s.animal not in seen:
            seen.add(s.animal); ds.animals.append(s.animal)
    print(f"{len(ds.animals)} animals, {len(ds.sets)} sessions, "
          f"{sum(s.n_obs for s in ds.sets)} neurons", flush=True)

    rows = {"operator": [], "group": [], "blend": []}
    groups, cors, weights = [], [], []
    t0 = time.time()
    for a in ds.animals:
        te_preds, va_preds, va_sets_all, val_animals = [], [], None, []
        te_sets = None
        for sd in args.seeds:
            got = run_fold(ds, a, cfg, measured, args.device, seed=sd)
            if got is None:
                continue
            model, te, te_sets, va, va_sets, val_animal = got
            te_preds.append(evaluate(model, te, cfg, args.device))
            va_preds.append(evaluate(model, va, cfg, args.device) if va else {})
            va_sets_all = va_sets
            val_animals.append(val_animal)
        if not te_preds or te_sets is None:
            continue

        def mean_pred(preds, key, c):
            vs = [p[key][c] for p in preds if c in p.get(key, {})]
            return np.mean(vs, axis=0) if vs else None

        def disagreement(preds, key, c):
            """How much the independently seeded models disagree, relative to the
            size of what they predict. Computable without any test label, and a
            usable proxy for how far the held-out animal sits outside the
            training distribution."""
            vs = [p[key][c] for p in preds if c in p.get(key, {})]
            if len(vs) < 2:
                return 0.0
            a = np.stack(vs)
            mu = a.mean(0)
            num = float(((a - mu) ** 2).mean())
            den = float((mu ** 2).mean()) + 1e-9
            return num / den

        train = [t for t in ds.sets if t.animal != a]
        # choose how far to trust the network versus the group average, using only
        # the validation animal, which was excluded from training
        num = den = 0.0
        dis_va = []
        if va_sets_all:
            vtrain = [t for t in ds.sets if t.animal not in (a, val_animals[0])]
            for s in va_sets_all:
                cs0 = [int(c) for c in np.unique(s.cond[s.perturbed])]
                dl = measured(s, cs0)
                ga = group_average(vtrain, s, cs0, measured)
                for c in cs0:
                    B = mean_pred(va_preds, s.key, c)
                    if c not in dl or B is None:
                        continue
                    A = dl[c]
                    G = ga.get(c, np.zeros_like(A))
                    e = float((A * A).sum()) + 1e-9
                    d = B - G
                    num += float(((A - G) * d).sum()) / e
                    den += float((d * d).sum()) / e
                    dis_va.append(disagreement(va_preds, s.key, c))
        w0 = float(np.clip(num / den, 0.0, 1.0)) if den > 1e-9 else 0.0
        # reference level of disagreement, measured on the validation animal
        dis_ref = float(np.median(dis_va)) if dis_va else 0.0
        weights.append(w0)

        for s in te_sets:
            cs0 = [int(c) for c in np.unique(s.cond[s.perturbed])]
            dl = measured(s, cs0)
            cs = [c for c in cs0 if c in dl and mean_pred(te_preds, s.key, c) is not None]
            if not cs:
                continue
            A = np.stack([dl[c] for c in cs])
            B = np.stack([mean_pred(te_preds, s.key, c) for c in cs])
            ga = group_average(train, s, cs, measured)
            G = np.stack([ga.get(c, np.zeros_like(dl[c])) for c in cs])
            v = M.delta_r2(A, B)
            if not np.isfinite(v):
                continue
            # trust the network less when its seeds disagree more than they did on
            # the validation animal
            dis = float(np.mean([disagreement(te_preds, s.key, c) for c in cs]))
            w = w0 / (1.0 + max(dis - dis_ref, 0.0) / max(dis_ref, 1e-3))
            rows["operator"].append(v)
            rows["group"].append(M.delta_r2(A, G))
            rows["blend"].append(M.delta_r2(A, G + w * (B - G)))
            cors.append(M.corr(A, G + w * (B - G)))
            groups.append(a)
        print(f"  [{a}] w={w:.2f} dis={dis:.2f}/{dis_ref:.2f} "
              f"operator={rows['operator'][-1]:+.3f} "
              f"group={rows['group'][-1]:+.3f} blend={rows['blend'][-1]:+.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = {}
    print(f"\n{'method':12s} {'dR2':>7s} {'95% CI':>18s} {'r':>7s} "
          f"{'animals>0':>10s} {'p':>10s}")
    print("-" * 70)
    for k in ("group", "operator", "blend"):
        rep = M.animal_level_report(rows[k], groups)
        rep["delta_corr"] = float(np.nanmean(cors)) if k == "blend" else None
        res[k] = rep
        print(f"{k:12s} {rep['animal_mean']:+7.3f} "
              f"[{rep['ci_lo']:+.2f},{rep['ci_hi']:+.2f}]".rjust(19) +
              f" {(rep['delta_corr'] or float('nan')):+7.3f} "
              f"{rep['sign_test']['n_positive']:>4d}/{rep['sign_test']['n']:<4d} "
              f"{rep['permutation']['p']:10.2e}")
    res["blend_weights"] = weights
    for name in ("operator", "blend"):
        ks = [k for k in res[name]["per_animal"] if k in res["group"]["per_animal"]]
        t = M.animal_permutation_test([res[name]["per_animal"][k] for k in ks],
                                      [res["group"]["per_animal"][k] for k in ks])
        res[f"test_{name}_vs_group"] = t
        print(f"{name:9s} vs group average: diff={t['mean_diff']:+.3f} p={t['p']:.2e} "
              f"(n={t['n']}, floor {t['p_floor']:.2e})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
