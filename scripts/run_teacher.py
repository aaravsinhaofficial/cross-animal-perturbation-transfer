"""Teacher-RNN benchmark: identifiability, ground-truth recovery and the
conserved-vs-idiosyncratic dissociation.

Three regimes (see ``cadence.teacher``):

  shared         one teacher defines the species operator -> parameter recovery
  heterogeneous  independently trained teachers -> realistic partial conservation
  degenerate     symmetric flow -> transfer must fail (falsification control)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from cadence.eval.identifiability import direction_recovery, latent_recovery
from cadence.experiment import ExperimentConfig, _jsonable, run_loao
from cadence.holdout import InterventionHoldout
from cadence.model import Cadence
from cadence.teacher import INTERV_TYPES, SyntheticConfig, build_synthetic_dataset
from cadence.training import TrainConfig, build_config, feature_tensors, pack


def identifiability_report(ds, truth, cfg: ExperimentConfig, held_out: str, seed: int = 0) -> dict:
    """Re-run one fold and, in addition, measure ground-truth recovery."""
    import copy

    from cadence.data.icms import stable_seed
    from cadence.experiment import evaluate_fold  # noqa: F401  (kept for parity)
    from cadence.training import calibrate_animal, fit, reset_animal

    train_sets = [s for s in ds.sets if s.animal != held_out]
    test_sets = [s for s in ds.sets if s.animal == held_out]
    all_sets = train_sets + test_sets
    mcfg = build_config(all_sets, ds.n_u, ds.n_raw, ds.n_beh, **cfg.model_kwargs())
    torch.manual_seed(seed)
    feats = feature_tensors(all_sets, cfg.device)
    model = Cadence(mcfg, unit_features=feats).to(cfg.device)
    tsets = pack(all_sets, cfg.device)
    tk = [s.key for s in train_sets]
    sk = [s.key for s in test_sets]

    tcfg = copy.deepcopy(cfg.train); tcfg.device = cfg.device; tcfg.seed = seed
    fit(model, {k: tsets[k] for k in tk}, tcfg, animal_keys=tk, use_perturbed=True, tag="id/shared")
    ccfg = copy.deepcopy(cfg.calib); ccfg.device = cfg.device; ccfg.seed = seed + 1
    for k in sk:
        reset_animal(model, k, seed=seed * 7919 + stable_seed(k) % 10000)
    calibrate_animal(model, tsets, sk, ccfg, tag="id/calib")

    out: dict = {}
    for s in test_sets:
        ts = tsets[s.key]
        gt = truth["animals"][s.animal]
        R = gt.get("R_unperturbed")
        if R is None:
            continue
        # model latents on the same unperturbed trials
        unp = np.where(~s.perturbed)[0][: len(R)]
        with torch.no_grad():
            idx = torch.as_tensor(unp, device=cfg.device)
            y_pre, _, u_pre, u_post, _, _, _ = ts.slice_batch(idx)
            z0 = model.encode(ts.key, y_pre, u_pre)
            roll = model.rollout(ts.key, z0, u_post, None, ts.T - ts.t0, use_intervention=False)
            z = roll["z"].cpu().numpy()
        r_true = R[: len(unp), s.t0 :, :]
        rec = latent_recovery(z, r_true)
        rec.pop("map", None)

        # recovered intervention directions: the latent push B a(type) at unit amplitude
        model_dirs = {}
        with torch.no_grad():
            for ti, tname in enumerate(INTERV_TYPES):
                raw = torch.zeros(1, 1, ds.n_raw, device=cfg.device)
                raw[0, 0, 0] = 1.0
                raw[0, 0, 1 + ti] = 1.0
                a = model.interv(raw, torch.ones(1, 1, device=cfg.device))
                d = model.G.state_independent(a[:, 0])[0].cpu().numpy()
                model_dirs[tname] = d
        dirs = direction_recovery(z, r_true, model_dirs, gt["dirs"])
        out[s.key] = {"latent": rec, "directions": dirs}
        print(f"  [{s.key}] latent CCA(top8)={rec['cca_mean_top']:.3f} "
              f"linR2={rec['linear_readout_r2']:.3f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in dirs.items()), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="heterogeneous",
                    choices=["shared", "heterogeneous", "degenerate"])
    ap.add_argument("--n-animals", type=int, default=8)
    ap.add_argument("--n-units", type=int, default=64)
    ap.add_argument("--trials-per-cond", type=int, default=250)
    ap.add_argument("--unperturbed", type=int, default=1000)
    ap.add_argument("--amplitudes", type=float, nargs="*", default=[1.5, 3.0, 4.5, 6.0])
    ap.add_argument("--rate-scale", type=float, default=6.0)
    ap.add_argument("--teacher-steps", type=int, default=900)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--g-rank", type=int, default=24)
    ap.add_argument("--residual-rank", type=int, default=3)
    ap.add_argument("--interv-kernel", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--calib-epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--weight-delta", type=float, default=4.0)
    ap.add_argument("--holdout", default="none", choices=["none", "amplitude", "type"])
    ap.add_argument("--amplitudes-holdout", type=float, nargs="*", default=[4.5])
    ap.add_argument("--types-holdout", nargs="*", default=["pc1"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--identifiability", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--methods", nargs="*", default=["cadence","no_effect","ma_latent","oracle","ctrl_permuted_obs","ctrl_scrambled_interv"])
    args = ap.parse_args()
    tag = args.tag or f"teacher_{args.regime}"

    scfg = SyntheticConfig(
        regime=args.regime, n_animals=args.n_animals, n_units=args.n_units,
        trials_per_condition=args.trials_per_cond, unperturbed_trials=args.unperturbed,
        amplitudes=tuple(args.amplitudes), rate_scale=args.rate_scale,
        teacher_steps=args.teacher_steps, device=args.device, seed=0,
    )
    if args.cache and args.cache.exists():
        with args.cache.open("rb") as fh:
            ds, truth = pickle.load(fh)
        print(f"loaded synthetic dataset from {args.cache}")
    else:
        ds, truth = build_synthetic_dataset(scfg)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with args.cache.open("wb") as fh:
                pickle.dump((ds, truth), fh, protocol=4)
    print(ds.summary())
    print(f"teacher task accuracy: {[round(a,3) for a in truth['task_accuracy']]}")

    ho = InterventionHoldout(kind="none")
    if args.holdout == "amplitude":
        ho = InterventionHoldout(kind="amplitude", amplitudes=tuple(args.amplitudes_holdout))
    elif args.holdout == "type":
        ho = InterventionHoldout(kind="type", types=tuple(args.types_holdout))

    cfg = ExperimentConfig(
        latent_dim=args.latent_dim, g_rank=args.g_rank, residual_rank=args.residual_rank,
        interv_kernel=args.interv_kernel,
        train=TrainConfig(epochs=args.epochs, steps_per_epoch=args.steps, batch_size=args.batch,
                          weight_delta=args.weight_delta, log_every=50),
        calib=TrainConfig(epochs=args.calib_epochs, steps_per_epoch=args.steps,
                          batch_size=args.batch, log_every=50, patience=40),
        seeds=tuple(args.seeds), device=args.device, out_dir=Path(args.out_dir),
        tag=tag, holdout=ho, breakdown="teacher", verbose_fit=not args.quiet,
        methods=tuple(args.methods),
    )
    print(f"holdout: {cfg.holdout.describe()}")
    run_loao(ds, cfg, animals=args.animals)

    if args.identifiability:
        print("\n=== ground-truth recovery (held-out animal) ===")
        rep = {}
        for a in (args.animals or ds.animals)[:3]:
            rep[a] = identifiability_report(ds, truth, cfg, a, seed=args.seeds[0])
        p = Path(args.out_dir) / f"{tag}_identifiability.json"
        p.write_text(json.dumps(_jsonable(rep), indent=1))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
