"""Leave-one-animal-out zero-shot intervention transfer on the ICMS dataset."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from cadence.experiment import ExperimentConfig, evaluate_fold, print_summary, run_loao, summarise
from cadence.holdout import InterventionHoldout
from cadence.training import TrainConfig


def build_cfg(args) -> ExperimentConfig:
    ho = InterventionHoldout(kind="none")
    if args.holdout == "amplitude":
        ho = InterventionHoldout(kind="amplitude", amplitudes=tuple(args.amplitudes))
    elif args.holdout == "amplitude_extrap":
        ho = InterventionHoldout(kind="amplitude_extrap", amp_threshold=args.amp_threshold)
    elif args.holdout == "depth":
        ho = InterventionHoldout(kind="depth", depth_band_um=tuple(args.depth_band))
    return ExperimentConfig(
        latent_dim=args.latent_dim,
        f_hidden=args.f_hidden,
        g_rank=args.g_rank,
        residual_rank=args.residual_rank,
        interv_code_dim=args.interv_code_dim,
        interv_kernel=args.interv_kernel,
        residual_l2=args.residual_l2,
        train=TrainConfig(
            epochs=args.epochs, steps_per_epoch=args.steps, batch_size=args.batch,
            weight_delta=args.weight_delta, weight_behavior=args.weight_behavior,
            lr_shared=args.lr, lr_animal=args.lr_animal, log_every=args.log_every,
            patience=args.patience,
        ),
        calib=TrainConfig(
            epochs=args.calib_epochs, steps_per_epoch=args.steps, batch_size=args.batch,
            lr_animal=args.lr_animal, log_every=args.log_every,
            patience=args.calib_patience, weight_behavior=args.weight_behavior,
        ),
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        device=args.device,
        out_dir=Path(args.out_dir),
        tag=args.tag,
        holdout=ho,
        breakdown="icms",
        verbose_fit=not args.quiet,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/proc/icms.pkl"))
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--f-hidden", type=int, default=128)
    ap.add_argument("--g-rank", type=int, default=24)
    ap.add_argument("--residual-rank", type=int, default=3)
    ap.add_argument("--residual-l2", type=float, default=1e-3)
    ap.add_argument("--interv-code-dim", type=int, default=8)
    ap.add_argument("--interv-kernel", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--calib-epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-animal", type=float, default=6e-3)
    ap.add_argument("--weight-delta", type=float, default=4.0)
    ap.add_argument("--weight-behavior", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--calib-patience", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--holdout", default="none",
                    choices=["none", "amplitude", "amplitude_extrap", "depth"])
    ap.add_argument("--amplitudes", type=float, nargs="*", default=[5.0])
    ap.add_argument("--amp-threshold", type=float, default=5.0)
    ap.add_argument("--depth-band", type=float, nargs=2, default=[0.0, 600.0])
    ap.add_argument("--methods", nargs="*", default=[
        "cadence", "no_effect", "ma_cca", "ma_latent", "unit_ridge",
        "oracle", "ctrl_permuted_obs", "ctrl_scrambled_interv"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="icms_main")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    with args.cache.open("rb") as fh:
        ds = pickle.load(fh)["dataset"]
    print(f"{ds.name}: {len(ds.animals)} animals, {len(ds.sets)} sessions")
    cfg = build_cfg(args)
    print(f"holdout: {cfg.holdout.describe()}")
    run_loao(ds, cfg, animals=args.animals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
