"""Leave-one-animal-out experiment engine.

For every animal in turn:

1. **Train** shared parameters (F, G, intervention encoder, behavioural readout,
   encoder core) plus the private parameters of the *other* animals. The held-out
   animal contributes nothing. If an intervention holdout is active, the held-out
   intervention settings are additionally deleted from every training animal.
2. **Calibrate** the held-out animal's private parameters with the shared
   parameters frozen, on its **unperturbed trials only**.
3. **Predict** the time-resolved neural and behavioural response to each
   intervention condition, zero-shot, and score it against the measurement.

Baselines, an oracle upper bound and negative controls run inside the same fold,
so every method sees exactly the same data.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import baselines as B
from . import metrics as M
from .data.containers import AnimalTrials, Dataset
from .data.icms import stable_seed
from .eval.breakdown import (
    dose_response,
    group_icms,
    group_teacher,
    per_condition_ceiling,
    per_condition_scores,
)
from .holdout import InterventionHoldout, eval_conditions, strip_training_conditions
from .model import Cadence
from .training import (
    TensorSet,
    TrainConfig,
    build_config,
    calibrate_animal,
    feature_tensors,
    fit,
    pack,
    predicted_condition_mean,
    predicted_delta,
    reset_animal,
)


@dataclass
class ExperimentConfig:
    latent_dim: int = 32
    f_hidden: int = 128
    f_layers: int = 2
    g_rank: int = 24
    residual_rank: int = 3
    interv_code_dim: int = 8
    interv_kernel: int = 0
    encoder_hidden: int = 128
    encoder_readin: int = 32
    dt: float = 1.0
    obs_likelihood: str = "poisson"
    residual_l2: float = 1e-3
    obs_l2: float = 1e-4
    g_l2: float = 1e-5
    unit_embed_hidden: int = 96
    free_obs_scale: float = 0.1
    free_obs_l2: float = 3e-2
    tie_readin_to_obs: bool = True

    train: TrainConfig = field(default_factory=TrainConfig)
    calib: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=250, patience=50))
    methods: tuple[str, ...] = (
        "cadence",
        "no_effect",
        "ma_cca",
        "ma_latent",
        "unit_ridge",
        "oracle",
        "ctrl_permuted_obs",
        "ctrl_scrambled_interv",
    )
    seeds: tuple[int, ...] = (0,)
    ma_dim: int = 12
    device: str = "cuda"
    out_dir: Path = Path("results")
    tag: str = "exp"
    holdout: InterventionHoldout = field(default_factory=InterventionHoldout)
    breakdown: str = "auto"          # 'auto' | 'teacher' | 'icms' | 'none'
    ceiling_splits: int = 200
    verbose_fit: bool = True

    def model_kwargs(self) -> dict:
        return dict(
            latent_dim=self.latent_dim,
            f_hidden=self.f_hidden,
            f_layers=self.f_layers,
            g_rank=self.g_rank,
            residual_rank=self.residual_rank,
            interv_code_dim=self.interv_code_dim,
            interv_kernel=self.interv_kernel,
            encoder_hidden=self.encoder_hidden,
            encoder_readin=self.encoder_readin,
            dt=self.dt,
            obs_likelihood=self.obs_likelihood,
            residual_l2=self.residual_l2,
            obs_l2=self.obs_l2,
            g_l2=self.g_l2,
            unit_embed_hidden=self.unit_embed_hidden,
            free_obs_scale=self.free_obs_scale,
            free_obs_l2=self.free_obs_l2,
            tie_readin_to_obs=self.tie_readin_to_obs,
        )


# ---------------------------------------------------------------------------
def _obs_maps(model: Cadence, keys) -> dict[str, np.ndarray]:
    return {k: model.animals[k].C.detach().cpu().numpy() for k in keys}


def _restrict(delta: dict[int, np.ndarray], conds: list[int]) -> dict[int, np.ndarray]:
    keep = set(int(c) for c in conds)
    return {c: v for c, v in delta.items() if int(c) in keep}


def _score(s: AnimalTrials, delta_pred: dict, conds: list[int], which: str = "neural") -> dict:
    arr = s.y if which == "neural" else s.behavior
    if arr is None:
        return {}
    y = arr[:, s.t0 :]
    keep = np.isin(s.cond, conds) | (~s.perturbed)
    return M.evaluate_delta(
        y[keep], s.cond[keep], s.perturbed[keep], _restrict(delta_pred, conds)
    )


def _breakdown_fn(cfg: ExperimentConfig, ds_name: str):
    mode = cfg.breakdown
    if mode == "auto":
        mode = "teacher" if ds_name.startswith("teacher") else "icms"
    if mode == "teacher":
        return group_teacher
    if mode == "icms":
        return group_icms
    return None


def evaluate_fold(
    ds: Dataset,
    held_out: str,
    cfg: ExperimentConfig,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    t_start = time.time()
    raw_train = [s for s in ds.sets if s.animal != held_out]
    test_sets = [s for s in ds.sets if s.animal == held_out]
    assert raw_train and test_sets, held_out

    # ---- apply the intervention holdout to the training animals -----------
    train_sets = []
    for s in raw_train:
        t = strip_training_conditions(s, cfg.holdout)
        if t is not None:
            train_sets.append(t)
    if not train_sets:
        raise RuntimeError(f"holdout {cfg.holdout.describe()} removed all training data")

    # ---- conditions to score in the held-out animal -----------------------
    eval_conds = {s.key: eval_conditions(s, cfg.holdout) for s in test_sets}
    test_sets = [s for s in test_sets if eval_conds[s.key]]
    if not test_sets:
        raise RuntimeError(f"no scorable conditions for {held_out}")

    all_sets = train_sets + test_sets
    mcfg = build_config(all_sets, ds.n_u, ds.n_raw, ds.n_beh, **cfg.model_kwargs())
    torch.manual_seed(seed)
    # per-unit features come from unperturbed activity and static metadata only,
    # so computing them for the held-out animal is inside the protocol
    feats = feature_tensors(all_sets, cfg.device)
    model = Cadence(mcfg, unit_features=feats).to(cfg.device)
    tsets = pack(all_sets, cfg.device)
    train_keys = [s.key for s in train_sets]
    test_keys = [s.key for s in test_sets]

    # ---- phase 1: shared training on the other animals -------------------
    tcfg = copy.deepcopy(cfg.train)
    tcfg.device = cfg.device
    tcfg.seed = seed
    tcfg.verbose = cfg.verbose_fit
    fit_info = fit(
        model, {k: tsets[k] for k in train_keys}, tcfg,
        train_shared=True, animal_keys=train_keys, use_perturbed=True,
        tag=f"shared/{held_out}",
    )

    # ---- phase 2: calibrate on the held-out animal's unperturbed data ----
    ccfg = copy.deepcopy(cfg.calib)
    ccfg.device = cfg.device
    ccfg.seed = seed + 1
    ccfg.verbose = cfg.verbose_fit
    for k in test_keys:
        reset_animal(model, k, seed=seed * 7919 + stable_seed(k) % 10000)
    calib_info = calibrate_animal(model, tsets, test_keys, ccfg, tag=f"calib/{held_out}")
    calibrated_state = copy.deepcopy(model.state_dict())

    per_set: dict[str, dict] = {}
    grouper = _breakdown_fn(cfg, ds.name)

    def add(method: str, key: str, neural: dict, behav: dict | None = None, extra: dict | None = None):
        per_set.setdefault(method, {})[key] = {
            "neural": neural, "behavior": behav, **(extra or {})
        }

    # ---- CADENCE zero-shot ------------------------------------------------
    for s in test_sets:
        ts = tsets[s.key]
        ec = eval_conds[s.key]
        dpred, _ = predicted_delta(model, ts, init_from="unperturbed")
        neural = _score(s, dpred, ec)
        dpred_own, _ = predicted_delta(model, ts, init_from="own")
        neural["delta_r2_own_init"] = _score(s, dpred_own, ec).get("delta_r2", float("nan"))

        y_post = s.y[:, s.t0 :]
        cmean = predicted_condition_mean(model, ts)
        raw_scores = {
            int(c): M.raw_r2(y_post[s.cond == c].mean(0), p)
            for c, p in cmean.items() if int(c) in set(ec)
        }
        pcs = per_condition_scores(s, _restrict(dpred, ec))
        extra = {
            "raw_r2_mean": float(np.nanmean(list(raw_scores.values()))) if raw_scores else float("nan"),
            "per_cond": pcs,
            "groups": grouper(s, pcs) if grouper else {},
            "dose": dose_response(s, _restrict(dpred, ec)),
            "ceiling": M.noise_ceiling(
                y_post[np.isin(s.cond, ec) | (~s.perturbed)],
                s.cond[np.isin(s.cond, ec) | (~s.perturbed)],
                s.perturbed[np.isin(s.cond, ec) | (~s.perturbed)],
                n_splits=cfg.ceiling_splits, seed=seed,
            ),
            "per_cond_ceiling": per_condition_ceiling(s, n_splits=120, seed=seed),
        }
        behav = None
        if s.behavior is not None and model.behavior is not None:
            bd, _ = predicted_delta(model, ts, field="behavior", init_from="unperturbed")
            behav = _score(s, bd, ec, "behavior")
            keep = np.isin(s.cond, ec) | (~s.perturbed)
            extra["ceiling_behavior"] = M.noise_ceiling(
                s.behavior[:, s.t0 :][keep], s.cond[keep], s.perturbed[keep],
                n_splits=cfg.ceiling_splits, seed=seed,
            )
            extra["behavior_per_channel"] = {
                int(ch): _score(
                    _channel_view(s, ch), _channel_delta(bd, ch), ec, "behavior"
                ).get("delta_r2", float("nan"))
                for ch in range(s.behavior.shape[-1])
            }
        add("cadence", s.key, neural, behav, extra)

    obs_maps = _obs_maps(model, [s.key for s in all_sets])

    # ---- baselines --------------------------------------------------------
    for s in test_sets:
        ec = eval_conds[s.key]
        specs = [
            ("no_effect", lambda s=s: B.no_effect(s)),
            ("ma_cca", lambda s=s: B.ma_cca(train_sets, s, d=cfg.ma_dim)),
            ("ma_latent", lambda s=s: B.ma_latent(train_sets, s, obs_maps)),
            ("unit_ridge", lambda s=s: B.unit_feature_ridge(train_sets, s)),
        ]
        for name, fn in specs:
            if name not in cfg.methods:
                continue
            try:
                d = fn()
                pcs = per_condition_scores(s, _restrict(d, ec))
                add(name, s.key, _score(s, d, ec), None,
                    {"per_cond": pcs, "groups": grouper(s, pcs) if grouper else {},
                     "dose": dose_response(s, _restrict(d, ec))})
            except Exception as exc:
                add(name, s.key, {"error": repr(exc)})

    # ---- negative controls ------------------------------------------------
    if "ctrl_permuted_obs" in cfg.methods:
        state = copy.deepcopy(model.state_dict())
        rng = np.random.default_rng(seed + 99)
        for k in test_keys:
            n = model.cfg.obs_dims[k]
            dev = model.animals[k].C_free.device
            model.animals[k].permute_units(
                torch.as_tensor(rng.permutation(n), device=dev)
            )
        for s in test_sets:
            d, _ = predicted_delta(model, tsets[s.key], init_from="unperturbed")
            add("ctrl_permuted_obs", s.key, _score(s, d, eval_conds[s.key]))
        model.load_state_dict(state)

    if "ctrl_scrambled_interv" in cfg.methods:
        for s in test_sets:
            ec = eval_conds[s.key]
            d, _ = predicted_delta(model, tsets[s.key], init_from="unperturbed")
            cs = sorted(d)
            if len(cs) > 1:
                rng = np.random.default_rng(seed + 7)
                perm = rng.permutation(len(cs))
                while np.all(perm == np.arange(len(cs))):
                    perm = rng.permutation(len(cs))
                d = {cs[i]: d[cs[perm[i]]] for i in range(len(cs))}
            add("ctrl_scrambled_interv", s.key, _score(s, d, ec))

    # ---- oracle upper bound ----------------------------------------------
    if "oracle" in cfg.methods:
        ocfg = copy.deepcopy(cfg.calib)
        ocfg.device = cfg.device
        ocfg.seed = seed + 2
        ocfg.verbose = cfg.verbose_fit
        ocfg.animal_params_unperturbed_only = False
        model.set_shared_grad(False)
        fit(model, {k: tsets[k] for k in test_keys}, ocfg,
            train_shared=False, animal_keys=test_keys, use_perturbed=True,
            tag=f"oracle/{held_out}")
        model.set_shared_grad(True)
        for s in test_sets:
            ts = tsets[s.key]
            ec = eval_conds[s.key]
            d, _ = predicted_delta(model, ts, init_from="unperturbed")
            behav = None
            if s.behavior is not None and model.behavior is not None:
                bd, _ = predicted_delta(model, ts, field="behavior", init_from="unperturbed")
                behav = _score(s, bd, ec, "behavior")
            add("oracle", s.key, _score(s, d, ec), behav)
        model.load_state_dict(calibrated_state)

    results = _aggregate(per_set)
    out = {
        "held_out": held_out,
        "seed": seed,
        "holdout": cfg.holdout.describe(),
        "n_train_animals": len({s.animal for s in train_sets}),
        "n_test_sets": len(test_sets),
        "eval_conds": {k: v for k, v in eval_conds.items()},
        "results": results,
        "per_set": per_set,
        "fit": {"best_val": fit_info["best_val"], "calib_val": calib_info["best_val"]},
        "wall_s": time.time() - t_start,
    }
    if verbose:
        c = results.get("cadence", {}).get("neural", {})
        ce = results.get("cadence", {}).get("ceiling", {})
        gp = results.get("cadence", {}).get("groups", {})
        extra = " ".join(
            f"{k}={v:+.2f}" for k, v in gp.items() if k.startswith("group:")
        )
        print(
            f"  [{held_out}] CADENCE dR2={c.get('delta_r2', float('nan')):+.3f} "
            f"r={c.get('delta_corr', float('nan')):+.3f} "
            f"ceil={ce.get('delta_r2_ceiling', float('nan')):.3f} | "
            f"ma_cca={results.get('ma_cca',{}).get('neural',{}).get('delta_r2',float('nan')):+.3f} "
            f"ma_lat={results.get('ma_latent',{}).get('neural',{}).get('delta_r2',float('nan')):+.3f} "
            f"ridge={results.get('unit_ridge',{}).get('neural',{}).get('delta_r2',float('nan')):+.3f} "
            f"oracle={results.get('oracle',{}).get('neural',{}).get('delta_r2',float('nan')):+.3f} "
            f"{extra} ({out['wall_s']:.0f}s)",
            flush=True,
        )
    return out


def _channel_view(s: AnimalTrials, ch: int) -> AnimalTrials:
    v = s.subset(np.arange(s.n_trials))
    v.behavior = s.behavior[:, :, ch : ch + 1]
    return v


def _channel_delta(d: dict[int, np.ndarray], ch: int) -> dict[int, np.ndarray]:
    return {c: v[:, ch : ch + 1] for c, v in d.items()}


def _aggregate(per_set: dict[str, dict]) -> dict:
    """Average each method's scores over the held-out animal's observation sets."""
    results: dict[str, dict] = {}
    for method, sets_d in per_set.items():
        agg: dict = {}
        for which in ("neural", "behavior"):
            vals: dict[str, list[float]] = {}
            for d in sets_d.values():
                sub = d.get(which)
                if not sub or "error" in sub:
                    continue
                for mk, mv in sub.items():
                    if isinstance(mv, (int, float)) and np.isfinite(mv):
                        vals.setdefault(mk, []).append(mv)
            agg[which] = {k: float(np.mean(v)) for k, v in vals.items()}
        for key in ("raw_r2_mean",):
            v = [d[key] for d in sets_d.values() if np.isfinite(d.get(key, np.nan))]
            if v:
                agg[key] = float(np.mean(v))
        for key in ("ceiling", "ceiling_behavior"):
            cs = [d[key] for d in sets_d.values() if d.get(key)]
            if cs:
                agg[key] = {
                    k: float(np.nanmean([c.get(k, np.nan) for c in cs])) for k in cs[0]
                }
        gs = [d["groups"] for d in sets_d.values() if d.get("groups")]
        if gs:
            keys = {k for g in gs for k in g}
            agg["groups"] = {
                k: float(np.nanmean([g[k]["delta_r2"] for g in gs if k in g])) for k in keys
            }
        ds = [d["dose"] for d in sets_d.values() if d.get("dose")]
        if ds:
            agg["dose_corr"] = float(
                np.nanmean([d["corr_measured_vs_predicted"] for d in ds])
            )
            agg["dose_slope_ratio"] = float(np.nanmedian([d["slope_ratio"] for d in ds]))
        bp = [d["behavior_per_channel"] for d in sets_d.values() if d.get("behavior_per_channel")]
        if bp:
            agg["behavior_per_channel"] = {
                k: float(np.nanmean([b[k] for b in bp if np.isfinite(b.get(k, np.nan))]))
                for k in bp[0]
            }
        results[method] = agg
    return results


def run_loao(ds: Dataset, cfg: ExperimentConfig, animals=None) -> dict:
    animals = animals or ds.animals
    folds: list[dict] = []
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"{cfg.tag}.json"

    def flush():
        out = {
            "dataset": ds.name,
            "config": _jsonable(asdict(cfg)),
            "animals": animals,
            "folds": folds,
            "summary": summarise(folds) if folds else {},
        }
        path.write_text(json.dumps(out, indent=1, default=_jsonable))
        return out

    for seed in cfg.seeds:
        for a in animals:
            try:
                folds.append(evaluate_fold(ds, a, cfg, seed=seed))
            except Exception as exc:
                print(f"  ! fold {a} seed {seed} failed: {exc!r}", flush=True)
            flush()          # keep partial results on disk after every fold
    out = flush()
    print(f"\nwrote {path}")
    if out["summary"]:
        print_summary(out["summary"])
    return out


def _jsonable(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def summarise(folds: list[dict]) -> dict:
    methods = sorted({m for f in folds for m in f["results"]})
    out: dict = {"methods": {}, "n_folds": len(folds)}
    per_method: dict[str, dict[str, float]] = {}
    for m in methods:
        vals: dict[str, list[float]] = {}
        for f in folds:
            r = f["results"].get(m, {})
            for which in ("neural", "behavior"):
                for k, v in (r.get(which) or {}).items():
                    vals.setdefault(f"{which}.{k}", []).append(v)
            for k in ("raw_r2_mean", "dose_corr", "dose_slope_ratio"):
                if k in r and np.isfinite(r[k]):
                    vals.setdefault(k, []).append(r[k])
            for ck, cv in (r.get("ceiling") or {}).items():
                vals.setdefault(f"ceiling.{ck}", []).append(cv)
            for ck, cv in (r.get("ceiling_behavior") or {}).items():
                vals.setdefault(f"ceiling_beh.{ck}", []).append(cv)
            for gk, gv in (r.get("groups") or {}).items():
                vals.setdefault(f"group.{gk}", []).append(gv)
            for bk, bv in (r.get("behavior_per_channel") or {}).items():
                vals.setdefault(f"beh_ch.{bk}", []).append(bv)
        summ = {}
        for k, v in vals.items():
            mean, lo, hi = M.bootstrap_ci(v)
            summ[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(v)}
        out["methods"][m] = summ
        per_method[m] = {
            f"{f['held_out']}|s{f['seed']}":
                f["results"].get(m, {}).get("neural", {}).get("delta_r2", np.nan)
            for f in folds
        }
    out["per_animal_delta_r2"] = per_method
    if "cadence" in per_method:
        keys = list(per_method["cadence"])
        a = [per_method["cadence"][k] for k in keys]
        tests = {}
        for m in methods:
            if m == "cadence":
                continue
            b = [per_method[m].get(k, np.nan) for k in keys]
            diff, p = M.paired_permutation_test(a, b)
            _, pw = M.wilcoxon_signed_rank(a, b)
            tests[m] = {"mean_diff": diff, "p_perm": p, "wilcoxon_p": pw, "n": len(keys)}
        out["paired_tests_vs_cadence"] = tests
        pos = [x for x in a if np.isfinite(x)]
        out["cadence_folds_above_zero"] = f"{sum(x > 0 for x in pos)}/{len(pos)}"
    return out


def print_summary(summary: dict) -> None:
    print(f"\n{'method':24s} {'neural dR2 [95% CI]':>30s} {'r':>8s} {'behav dR2':>11s}")
    print("-" * 78)
    for m, s in summary["methods"].items():
        d = s.get("neural.delta_r2", {})
        r = s.get("neural.delta_corr", {})
        b = s.get("behavior.delta_r2", {})
        print(
            f"{m:24s} {d.get('mean', float('nan')):+.3f} "
            f"[{d.get('ci_lo', float('nan')):+.3f},{d.get('ci_hi', float('nan')):+.3f}]"
            f"{'':6s} {r.get('mean', float('nan')):+.3f} "
            f"{b.get('mean', float('nan')):+11.3f}"
        )
    ce = summary["methods"].get("cadence", {}).get("ceiling.delta_r2_ceiling", {})
    print(f"\nnoise ceiling: {ce.get('mean', float('nan')):.3f}")
    print(f"folds above zero: {summary.get('cadence_folds_above_zero')}")
    for m, t in (summary.get("paired_tests_vs_cadence") or {}).items():
        print(f"  cadence vs {m:22s} diff={t['mean_diff']:+.3f} p_perm={t['p_perm']:.2e}")
