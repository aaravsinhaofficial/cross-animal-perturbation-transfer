"""Leave-one-animal-out training of the residual cross-animal operator.

For each held-out animal the network is trained on the others, with one further
animal set aside for early stopping. The held-out animal contributes only its
unperturbed activity and the stimulus settings the experimenter chose.

Every session, including the training ones, is handed a stereotype response built
with its own animal excluded, so the network is always predicting a correction to
something it could have known without that animal.

Per-neuron predictions are written out so that later analyses (which neurons the
correction helps on, how the gap scales with measurement quality) do not need a
retrain.
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
from cadence import individuality as IND
from cadence import operator2 as O2

warnings.filterwarnings("ignore")


def make_measured():
    cache: dict = {}

    def measured(s, conds):
        """The effect, measured against the half of the control trials the model
        never sees as a feature."""
        if s.key not in cache:
            Y = s.y[:, s.t0 :]
            _, base_idx = IND.control_split(s)
            base = np.nanmean(Y[base_idx], 0)
            cache[s.key] = {int(c): np.nanmean(Y[s.cond == c], 0) - base
                            for c in np.unique(s.cond[s.perturbed])}
        return {c: cache[s.key][c] for c in conds if c in cache[s.key]}

    return measured


def predict_all(model, ex, cfg, device):
    model.eval()
    out: dict[str, dict[int, np.ndarray]] = {}
    with torch.no_grad():
        for i in range(0, len(ex), 16):
            b = ex[i : i + 16]
            neu, stim, rel, prof, base, tgt, mask = O2.collate(b, device)
            p = model(neu, stim, rel, prof, base, mask).cpu().numpy()
            for j, e in enumerate(b):
                out.setdefault(e["key"], {})[e["cond"]] = p[j, : e["n"]].T * e["scale"]
    return out


def reliability(s, conds, pred: dict) -> float:
    """How well this animal's individual response can be measured at all.

    The noise on an effect estimated from n stimulation trials against n0 control
    trials has variance v(1/n + 1/n0), where v is the trial-to-trial variance, and
    all three quantities come from control trials and from the experimenter's own
    design. Comparing that noise with the size of the correction the model wants to
    make gives a number in (0, 1) that says whether the correction is even resolvable
    in this animal. No perturbation response from this animal is read.
    """
    Y = s.y[:, s.t0 :]
    _, ci = IND.control_split(s)
    n0 = max(len(ci), 1)
    v0 = np.nanvar(Y[ci], 0)
    noise = sig = 0.0
    for c in conds:
        nc = max(int((s.cond == c).sum()), 1)
        noise += float(np.nansum(v0 * (1.0 / nc + 1.0 / n0)))
        if c in pred:
            sig += float(np.nansum(pred[c] ** 2))
    return sig / (sig + noise) if sig + noise > 0 else 0.0


def disagreement(preds, key, c) -> float:
    """Spread across independently seeded models, relative to what they predict."""
    vs = [p[key][c] for p in preds if c in p.get(key, {})]
    if len(vs) < 2:
        return 0.0
    a = np.stack(vs)
    mu = a.mean(0)
    return float(((a - mu) ** 2).mean()) / (float((mu ** 2).mean()) + 1e-9)


def run_fold(ds, held, cfg, measured, stereo, device, seed=0, cache=None):
    animals = [a for a in ds.animals if a != held]
    val_animal = animals[zlib.crc32(held.encode()) % len(animals)]
    rng = np.random.default_rng(seed + 991)
    tr_sets = [s for s in ds.sets if s.animal not in (held, val_animal)]
    va_sets = [s for s in ds.sets if s.animal == val_animal]
    te_sets = [s for s in ds.sets if s.animal == held]
    if not tr_sets or not te_sets:
        return None

    # every session sees a stereotype that excludes its own animal, the test animal
    # and the validation animal, so training and validation are on equal footing
    def base_tr(s):
        return stereo.predict(s, [int(c) for c in np.unique(s.cond[s.perturbed])],
                              {held, val_animal, s.animal})

    def base_te(s):
        return stereo.predict(s, [int(c) for c in np.unique(s.cond[s.perturbed])],
                              {held})

    tr = O2.pack(tr_sets, cfg, measured, base_tr, cache)
    va = O2.pack(va_sets, cfg, measured, base_tr, cache)
    te = O2.pack(te_sets, cfg, measured, base_te, cache)
    if not tr or not te:
        return None

    # each animal contributes equally to the loss regardless of session count
    cnt: dict[str, int] = {}
    for e in tr:
        cnt[e["animal"]] = cnt.get(e["animal"], 0) + 1
    for e in tr:
        e["w"] = 1.0 / cnt[e["animal"]]

    torch.manual_seed(seed)
    T = tr[0]["tgt"].shape[1]
    model = O2.Operator2(cfg, tr[0]["neu"].shape[1], tr[0]["stim"].shape[0],
                         tr[0]["rel"].shape[1], T,
                         n_prof=tr[0]["prof"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best, best_state, bad = np.inf, None, 0
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), cfg.batch_sessions):
            b = [tr[j] for j in order[i : i + cfg.batch_sessions]]
            neu, stim, rel, prof, base, tgt, mask = O2.collate(b, device, rng, cfg)
            w = torch.as_tensor([e["w"] for e in b], dtype=torch.float32,
                                device=device)
            loss = O2.masked_loss(model(neu, stim, rel, prof, base, mask), tgt, mask,
                                  cfg.huber_delta, weight=w)
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
                    neu, stim, rel, prof, base, tgt, mask = O2.collate(b, device)
                    vl += float(O2.masked_loss(model(neu, stim, rel, prof, base, mask),
                                               tgt, mask, cfg.huber_delta)) * len(b)
                    vn += len(b)
                vl /= max(vn, 1)
            if vl < best - 1e-6:
                best, bad, best_state = vl, 0, {
                    k: v.detach().clone() for k, v in model.state_dict().items()}
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
    ap.add_argument("--tag", default="alm2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-ceiling", type=float, default=0.0)
    ap.add_argument("--save-preds", action="store_true", default=True)
    args = ap.parse_args()

    ds = pickle.load(args.cache[0].open("rb"))["dataset"]
    for c in args.cache[1:]:
        ds.sets = list(ds.sets) + list(pickle.load(c.open("rb"))["dataset"].sets)
    measured = make_measured()
    stereo = O2.Stereotype(ds.sets, measured)
    cfg = O2.Operator2Config(epochs=args.epochs, device=args.device)
    print(f"{len(ds.sets)} sessions, {len(ds.animals)} animals, device {args.device}",
          flush=True)

    tcache: dict = {}
    rows = {k: [] for k in ("group", "operator", "blend")}
    groups, cors, weights = [], [], []
    store: dict[str, np.ndarray] = {}
    t0 = time.time()
    for a in ds.animals:
        te_preds, va_preds, va_sets_all, val_animals, te_sets = [], [], None, [], None
        for sd in args.seeds:
            got = run_fold(ds, a, cfg, measured, stereo, args.device, seed=sd,
                           cache=tcache)
            if got is None:
                continue
            model, te, te_sets, va, va_sets, val_animal = got
            te_preds.append(predict_all(model, te, cfg, args.device))
            va_preds.append(predict_all(model, va, cfg, args.device) if va else {})
            va_sets_all = va_sets
            val_animals.append(val_animal)
        if not te_preds or te_sets is None:
            continue

        def mean_pred(preds, key, c):
            vs = [p[key][c] for p in preds if c in p.get(key, {})]
            return np.mean(vs, axis=0) if vs else None

        # how much to trust the correction, fitted on the validation animal only,
        # together with the two label-free quantities used to carry that weight over
        # to an animal that may be measured very differently
        num = den = 0.0
        rel_va, dis_va = [], []
        if va_sets_all:
            for s in va_sets_all:
                cs0 = [int(c) for c in np.unique(s.cond[s.perturbed])]
                dl = measured(s, cs0)
                ga = stereo.predict(s, cs0, {a, val_animals[0]})
                corr = {}
                for c in cs0:
                    B = mean_pred(va_preds, s.key, c)
                    if c not in dl or B is None or c not in ga:
                        continue
                    A, G = dl[c], ga[c]
                    e = float((A * A).sum()) + 1e-9
                    d = B - G
                    corr[c] = d
                    num += float(((A - G) * d).sum()) / e
                    den += float((d * d).sum()) / e
                    dis_va.append(disagreement(va_preds, s.key, c))
                if corr:
                    rel_va.append(reliability(s, list(corr), corr))
        w0 = float(np.clip(num / den, 0.0, 1.0)) if den > 1e-9 else 0.0
        rel_ref = float(np.median(rel_va)) if rel_va else 0.0
        dis_ref = float(np.median(dis_va)) if dis_va else 0.0

        for s in te_sets:
            cs0 = [int(c) for c in np.unique(s.cond[s.perturbed])]
            dl = measured(s, cs0)
            ga = stereo.predict(s, cs0, {a})
            cs = [c for c in cs0
                  if c in dl and c in ga and mean_pred(te_preds, s.key, c) is not None]
            if not cs:
                continue
            A = np.stack([dl[c] for c in cs])
            B = np.stack([mean_pred(te_preds, s.key, c) for c in cs])
            G = np.stack([ga[c] for c in cs])
            if not np.isfinite(M.delta_r2(A, B)):
                continue
            # scale the trust by how resolvable this animal's individual response is
            # relative to the validation animal, and by how much the seeds disagree
            corr = {c: B[i] - G[i] for i, c in enumerate(cs)}
            rel = reliability(s, cs, corr)
            dis = float(np.mean([disagreement(te_preds, s.key, c) for c in cs]))
            w = w0 * float(np.clip(rel / max(rel_ref, 1e-6), 0.0, 1.0))
            w /= 1.0 + max(dis - dis_ref, 0.0) / max(dis_ref, 1e-3)
            weights.append(w)
            rows["operator"].append(M.delta_r2(A, B))
            rows["group"].append(M.delta_r2(A, G))
            rows["blend"].append(M.delta_r2(A, G + w * (B - G)))
            cors.append(M.corr(A, B))
            groups.append(a)
            if args.save_preds:
                fi, _ = IND.control_split(s)
                yc = s.y[fi][:, s.t0 :]
                store[f"{s.key}|A"] = A.astype(np.float32)
                store[f"{s.key}|B"] = B.astype(np.float32)
                store[f"{s.key}|G"] = G.astype(np.float32)
                store[f"{s.key}|ctrl"] = np.nanmean(yc, 0).astype(np.float32)
                store[f"{s.key}|cond"] = np.array(cs)
                store[f"{s.key}|amp"] = np.array(
                    [float(s.meta["cond_amp"][c]) for c in cs], np.float32)
                store[f"{s.key}|animal"] = np.array([a])
        if groups and groups[-1] == a:
            print(f"  [{a}] w={weights[-1]:.2f} operator={rows['operator'][-1]:+.3f} "
                  f"group={rows['group'][-1]:+.3f} blend={rows['blend'][-1]:+.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    res = {}
    print(f"\n{'method':12s} {'dR2':>7s} {'95% CI':>18s} {'r':>7s} "
          f"{'animals>0':>10s} {'p':>10s}")
    print("-" * 70)
    for k in ("group", "operator", "blend"):
        rep = M.animal_level_report(rows[k], groups)
        rep["delta_corr"] = float(np.nanmean(cors)) if k == "operator" else None
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

    out = Path(f"results/operator_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    if store:
        np.savez_compressed(f"results/preds_{args.tag}.npz", **store)
        print(f"wrote results/preds_{args.tag}.npz")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
