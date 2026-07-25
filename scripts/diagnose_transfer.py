"""Diagnostic: where does zero-shot transfer lose accuracy?

Reports, for one leave-one-animal-out fold:
  * in-sample Delta-R^2 for every training animal (upper bound for a perfectly
    calibrated observation map),
  * zero-shot Delta-R^2 for the held-out animal,
  * the norm of the predicted vs measured effect (scale calibration),
  * the same after an explicit single-scalar rescaling of the prediction, which
    separates "wrong direction" from "wrong gain".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cadence import metrics as M
from cadence.eval.breakdown import group_teacher, per_condition_ceiling, per_condition_scores
from cadence.experiment import ExperimentConfig
from cadence.model import Cadence
from cadence.teacher import SyntheticConfig, build_synthetic_dataset
from cadence.training import (
    TrainConfig,
    build_config,
    calibrate_animal,
    fit,
    pack,
    predicted_delta,
    reset_animal,
)


def score(s, dpred):
    return M.evaluate_delta(s.y[:, s.t0 :], s.cond, s.perturbed, dpred)


def best_scale(s, dpred):
    """Optimal single gain applied to the prediction (diagnostic only)."""
    d_true, _ = M.measured_delta(s.y[:, s.t0 :], s.cond, s.perturbed)
    conds = sorted(set(d_true) & set(dpred))
    A = np.stack([d_true[c] for c in conds])
    B = np.stack([dpred[c] for c in conds])
    k = float((A * B).sum() / max((B * B).sum(), 1e-12))
    return k, M.delta_r2(A, B * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual-rank", type=int, default=3)
    ap.add_argument("--residual-l2", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=20)
    ap.add_argument("--n-animals", type=int, default=4)
    ap.add_argument("--trials-per-cond", type=int, default=150)
    ap.add_argument("--unperturbed", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--calib-epochs", type=int, default=120)
    ap.add_argument("--weight-delta", type=float, default=4.0)
    ap.add_argument("--calib-lr", type=float, default=6e-3)
    ap.add_argument("--held-out", default="rnn00")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teacher-steps", type=int, default=300)
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--out", type=Path, default=Path("results/diagnostics"))
    args = ap.parse_args()

    scfg = SyntheticConfig(
        regime="heterogeneous",
        n_animals=args.n_animals,
        n_units=64,
        trials_per_condition=args.trials_per_cond,
        unperturbed_trials=args.unperturbed,
        amplitudes=(1.5, 3.0, 4.5, 6.0),
        rate_scale=6.0,
        teacher_steps=args.teacher_steps,
        device=args.device,
        seed=0,
    )
    ds, _ = build_synthetic_dataset(scfg)

    ecfg = ExperimentConfig(
        latent_dim=args.latent_dim,
        residual_rank=args.residual_rank,
        residual_l2=args.residual_l2,
        device=args.device,
    )
    train_sets = [s for s in ds.sets if s.animal != args.held_out]
    test_sets = [s for s in ds.sets if s.animal == args.held_out]
    all_sets = train_sets + test_sets
    mcfg = build_config(all_sets, ds.n_u, ds.n_raw, ds.n_beh, **ecfg.model_kwargs())
    torch.manual_seed(0)
    model = Cadence(mcfg).to(args.device)
    tsets = pack(all_sets, args.device)
    train_keys = [s.key for s in train_sets]
    test_keys = [s.key for s in test_sets]

    tcfg = TrainConfig(
        epochs=args.epochs, steps_per_epoch=25, batch_size=128, log_every=40,
        device=args.device, weight_delta=args.weight_delta,
    )
    fit(model, {k: tsets[k] for k in train_keys}, tcfg, animal_keys=train_keys,
        use_perturbed=True, tag="shared")

    out: dict = {"args": vars(args), "in_sample": {}, "held_out": {}}
    for s in train_sets:
        d, _ = predicted_delta(model, tsets[s.key], init_from="unperturbed")
        sc = score(s, d)
        k, r2k = best_scale(s, d)
        grp = group_teacher(s, per_condition_scores(s, d))
        out["in_sample"][s.key] = {
            "delta_r2": sc["delta_r2"], "delta_corr": sc["delta_corr"],
            "norm_true": sc["effect_norm_true"], "norm_pred": sc["effect_norm_pred"],
            "best_gain": k, "delta_r2_rescaled": r2k, "groups": grp,
        }
        print(f"  in-sample {s.key}: dR2={sc['delta_r2']:+.3f} r={sc['delta_corr']:+.3f} "
              f"gain*={k:.3f} | cons={grp.get('group:conserved',{}).get('delta_r2',float('nan')):+.3f} "
              f"idio={grp.get('group:idiosyncratic',{}).get('delta_r2',float('nan')):+.3f} "
              f"| " + " ".join(f"{kk.split(':')[1][:6]}={vv['delta_r2']:+.2f}"
                               for kk, vv in grp.items() if kk.startswith('type:')), flush=True)

    ccfg = TrainConfig(
        epochs=args.calib_epochs, steps_per_epoch=25, batch_size=128, log_every=60,
        device=args.device, lr_animal=args.calib_lr, patience=60,
    )
    for k in test_keys:
        reset_animal(model, k, seed=1)
    calibrate_animal(model, tsets, test_keys, ccfg, tag="calib")

    for s in test_sets:
        d, _ = predicted_delta(model, tsets[s.key], init_from="unperturbed")
        sc = score(s, d)
        k, r2k = best_scale(s, d)
        ceil = M.noise_ceiling(s.y[:, s.t0 :], s.cond, s.perturbed, n_splits=200)
        pcs = per_condition_scores(s, d)
        grp = group_teacher(s, pcs)
        out["held_out"][s.key] = {
            "delta_r2": sc["delta_r2"], "delta_corr": sc["delta_corr"],
            "norm_true": sc["effect_norm_true"], "norm_pred": sc["effect_norm_pred"],
            "best_gain": k, "delta_r2_rescaled": r2k,
            "ceiling": ceil["delta_r2_ceiling"], "groups": grp,
            "per_cond": pcs, "per_cond_ceiling": per_condition_ceiling(s, n_splits=120),
        }
        print(f"  ZERO-SHOT {s.key}: dR2={sc['delta_r2']:+.3f} r={sc['delta_corr']:+.3f} "
              f"gain*={k:.3f} dR2*={r2k:+.3f} ceil={ceil['delta_r2_ceiling']:.3f} "
              f"| cons={grp.get('group:conserved',{}).get('delta_r2',float('nan')):+.3f} "
              f"idio={grp.get('group:idiosyncratic',{}).get('delta_r2',float('nan')):+.3f} "
              f"| " + " ".join(f"{kk.split(':')[1][:6]}={vv['delta_r2']:+.2f}"
                               for kk, vv in grp.items() if kk.startswith('type:')), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out / (args.tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
