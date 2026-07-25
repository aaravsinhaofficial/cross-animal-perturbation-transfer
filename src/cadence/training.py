"""Training and the cross-animal calibration protocol.

The protocol that defines the paper's claim is implemented in
:func:`calibrate_animal`: for a held-out animal, every shared parameter is
frozen and only that animal's private parameters (observation map, residual
dynamics, encoder read-in) are fitted -- using its **unperturbed trials only**.
No intervention trial from the held-out animal is ever touched before scoring.

To remove any asymmetry between training and test animals, animal-private
parameters are, by default, fitted on unperturbed batches for *every* animal
(``TrainConfig.animal_params_unperturbed_only``). Shared parameters see both
unperturbed and intervention data from the training animals.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .data.containers import AnimalTrials
from .model import Cadence, CadenceConfig


@dataclass
class TrainConfig:
    epochs: int = 200
    steps_per_epoch: int = 40
    # the rollout is launch-bound, so large batches cost almost nothing
    batch_size: int = 512
    lr_shared: float = 3e-3
    lr_animal: float = 6e-3
    weight_decay: float = 0.0
    weight_behavior: float = 1.0
    grad_clip: float = 1.0
    device: str = "cuda"
    seed: int = 0
    animal_params_unperturbed_only: bool = True
    perturbed_batch_frac: float = 0.5
    val_frac: float = 0.15
    patience: int = 40
    log_every: int = 25
    verbose: bool = True
    # weight on the condition-averaged causal-effect matching term, and how
    # often a training step targets it rather than the per-trial likelihood
    weight_delta: float = 4.0
    delta_batch_frac: float = 0.4
    delta_init_batch: int = 512
    eval_every: int = 2
    eval_delta_conds: int = 6
    eval_delta_batch: int = 256
    eval_max_trials: int = 768
    cosine_schedule: bool = True
    min_lr_frac: float = 0.05
    scale_obs_loss: bool = True
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# tensor packing
# ---------------------------------------------------------------------------
class TensorSet:
    def __init__(self, s: AnimalTrials, device: str):
        self.key = s.key
        self.animal = s.animal
        self.t0 = s.t0
        self.T = s.T
        self.n_obs = s.n_obs
        self.bin_s = s.bin_s
        d = device
        self.y = torch.as_tensor(s.y, dtype=torch.float32, device=d)
        self.u = None if s.u is None else torch.as_tensor(s.u, dtype=torch.float32, device=d)
        self.raw = torch.as_tensor(s.interv_raw, dtype=torch.float32, device=d)
        self.on = torch.as_tensor(s.interv_on, dtype=torch.float32, device=d)
        self.beh = (
            None if s.behavior is None else torch.as_tensor(s.behavior, dtype=torch.float32, device=d)
        )
        self.perturbed = torch.as_tensor(s.perturbed, dtype=torch.bool, device=d)
        self.cond = torch.as_tensor(s.cond, dtype=torch.long, device=d)
        self.pert_idx = torch.where(self.perturbed)[0]
        self.unp_idx = torch.where(~self.perturbed)[0]
        self._precompute_deltas()

    def _precompute_deltas(self):
        """Measured condition-averaged causal effects (targets for the
        delta-matching term). Computed once per observation set."""
        t0 = self.t0
        self.cond_ids: list[int] = []
        self.delta_y: dict[int, torch.Tensor] = {}
        self.delta_beh: dict[int, torch.Tensor] = {}
        self.cond_raw: dict[int, torch.Tensor] = {}
        self.cond_on: dict[int, torch.Tensor] = {}
        if len(self.unp_idx) == 0 or len(self.pert_idx) == 0:
            self.delta_scale = torch.ones((), device=self.y.device)
            self.delta_scale_beh = torch.ones((), device=self.y.device)
            return
        base_y = self.y[self.unp_idx, t0:].mean(0)
        base_b = None if self.beh is None else self.beh[self.unp_idx, t0:].mean(0)
        pert_conds = torch.unique(self.cond[self.pert_idx])
        sq, sqb = [], []
        for c in pert_conds.tolist():
            ids = torch.where(self.cond == c)[0]
            self.cond_ids.append(int(c))
            d = self.y[ids, t0:].mean(0) - base_y
            self.delta_y[int(c)] = d
            sq.append(d.pow(2).mean())
            if self.beh is not None:
                db = self.beh[ids, t0:].mean(0) - base_b
                self.delta_beh[int(c)] = db
                sqb.append(db.pow(2).mean())
            self.cond_raw[int(c)] = self.raw[ids, t0:].mean(0, keepdim=True)
            self.cond_on[int(c)] = (self.on[ids, t0:].mean(0, keepdim=True) > 0.5).float()
        self.delta_scale = torch.stack(sq).mean().clamp_min(1e-8)
        self.delta_scale_beh = (
            torch.stack(sqb).mean().clamp_min(1e-8) if sqb else torch.ones((), device=self.y.device)
        )

    def slice_batch(self, idx: torch.Tensor):
        t0 = self.t0
        y_pre = self.y[idx, :t0]
        y_post = self.y[idx, t0:]
        u_pre = None if self.u is None else self.u[idx, :t0]
        u_post = None if self.u is None else self.u[idx, t0:]
        raw = self.raw[idx, t0:]
        on = self.on[idx, t0:]
        beh = None if self.beh is None else self.beh[idx, t0:]
        return y_pre, y_post, u_pre, u_post, raw, on, beh


def pack(sets: list[AnimalTrials], device: str) -> dict[str, TensorSet]:
    return {s.key: TensorSet(s, device) for s in sets}


def build_config(
    sets: list[AnimalTrials],
    n_u: int,
    n_raw: int,
    n_beh: int,
    **kw,
) -> CadenceConfig:
    feat_dim = 0
    for s in sets:
        if s.unit_features is not None:
            feat_dim = int(s.unit_features.shape[1])
            break
    kw.setdefault("unit_feature_dim", feat_dim)
    return CadenceConfig(
        input_dim=n_u,
        interv_raw_dim=n_raw,
        behavior_dim=n_beh,
        animals=tuple(s.key for s in sets),
        obs_dims={s.key: s.n_obs for s in sets},
        **kw,
    )


def feature_tensors(sets: list[AnimalTrials], device: str) -> dict[str, torch.Tensor]:
    """Per-unit feature matrices, z-scored with statistics pooled across the
    supplied sets so that the shared embedding sees comparable inputs."""
    have = [s for s in sets if s.unit_features is not None]
    if not have:
        return {}
    allf = np.concatenate([s.unit_features for s in have], axis=0)
    mu = allf.mean(0, keepdims=True)
    sd = allf.std(0, keepdims=True) + 1e-6
    return {
        s.key: torch.as_tensor((s.unit_features - mu) / sd, dtype=torch.float32, device=device)
        for s in have
    }


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def batch_loss(model: Cadence, ts: TensorSet, idx: torch.Tensor, cfg: TrainConfig,
               use_intervention: bool = True):
    y_pre, y_post, u_pre, u_post, raw, on, beh = ts.slice_batch(idx)
    n_steps = y_post.shape[1]
    pred = model(
        ts.key, y_pre, u_pre, u_post,
        raw if use_intervention else None,
        on if use_intervention else None,
        n_steps,
        use_intervention=use_intervention,
    )
    loss = model.observation_nll(pred, y_post)
    parts = {"obs": float(loss.detach())}
    if beh is not None and model.behavior is not None and cfg.weight_behavior > 0:
        bl = model.behavior_nll(pred, beh)
        loss = loss + cfg.weight_behavior * bl
        parts["beh"] = float(bl.detach())
    reg = model.regularisation(ts.key)
    loss = loss + reg
    parts["reg"] = float(reg.detach())
    parts["total"] = float(loss.detach())
    return loss, parts


def delta_loss(
    model: Cadence,
    ts: TensorSet,
    cfg: TrainConfig,
    rng: np.random.Generator,
    unp_pool: np.ndarray,
    conds: list[int] | None = None,
):
    """Match the model's *predicted* condition-averaged causal effect to the
    measured one, using exactly the estimator used at evaluation time.

    Initial conditions are drawn from unperturbed trials; the intervention is
    then switched on or off from the same states, so this term supervises the
    shared causal operator directly rather than through the per-trial
    likelihood, where the effect is swamped by the baseline rate.
    """
    if not ts.cond_ids or len(unp_pool) == 0:
        return None, {}
    conds = conds or ts.cond_ids
    n_take = min(cfg.delta_init_batch, len(unp_pool))
    sel = rng.choice(unp_pool, size=n_take, replace=False)
    idx = torch.as_tensor(sel, device=ts.y.device)
    y_pre, _, u_pre, u_post, _, _, _ = ts.slice_batch(idx)
    n_steps = ts.T - ts.t0
    z0 = model.encode(ts.key, y_pre, u_pre)
    off = model.rollout(ts.key, z0, u_post, None, n_steps, use_intervention=False)

    c = int(conds[rng.integers(len(conds))])
    raw = ts.cond_raw[c].expand(n_take, -1, -1)
    on = ts.cond_on[c].expand(n_take, -1)
    a = model.interv(raw, on)
    onp = model.rollout(ts.key, z0, u_post, a, n_steps, use_intervention=True)

    fld = "rate" if model.cfg.obs_likelihood == "poisson" else "mean"
    d_pred = onp[fld].mean(0) - off[fld].mean(0)
    loss = (d_pred - ts.delta_y[c]).pow(2).mean() / ts.delta_scale
    parts = {"delta": float(loss.detach())}
    if model.behavior is not None and c in ts.delta_beh and cfg.weight_behavior > 0:
        db = onp["behavior"].mean(0) - off["behavior"].mean(0)
        lb = (db - ts.delta_beh[c]).pow(2).mean() / ts.delta_scale_beh
        loss = loss + cfg.weight_behavior * lb
        parts["delta_beh"] = float(lb.detach())
    return cfg.weight_delta * loss, parts


def _split_train_val(n: int, val_frac: float, rng: np.random.Generator):
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def eval_loss(model: Cadence, tsets: dict[str, TensorSet], val: dict[str, dict],
              cfg: TrainConfig, keys=None, use_perturbed: bool = True) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for k in keys or tsets:
        ts = tsets[k]
        for kind in ("unp", "pert"):
            if kind == "pert" and not use_perturbed:
                continue
            ids = val[k][kind]
            if len(ids) == 0:
                continue
            if len(ids) > cfg.eval_max_trials:
                ids = ids[: cfg.eval_max_trials]
            ids = torch.as_tensor(ids, device=ts.y.device)
            for j in range(0, len(ids), 512):
                sub = ids[j : j + 512]
                _, parts = batch_loss(model, ts, sub, cfg, use_intervention=(kind == "pert"))
                tot += parts["total"] * len(sub)
                cnt += len(sub)
    base = tot / max(cnt, 1)
    # add the causal-effect term so that early stopping tracks the quantity the
    # experiment actually scores
    if use_perturbed and cfg.weight_delta > 0:
        dl, dn = 0.0, 0
        ecfg = copy.copy(cfg)
        ecfg.delta_init_batch = cfg.eval_delta_batch
        for k in keys or tsets:
            ts = tsets[k]
            pool = val[k]["unp"]
            if len(pool) == 0 or not ts.cond_ids:
                continue
            rng = np.random.default_rng(0)
            # a fixed, evenly spaced subset of conditions keeps the validation
            # signal stable across epochs while bounding cost
            step = max(1, len(ts.cond_ids) // cfg.eval_delta_conds)
            for c in ts.cond_ids[::step][: cfg.eval_delta_conds]:
                loss, _ = delta_loss(model, ts, ecfg, rng, pool, conds=[c])
                if loss is not None:
                    dl += float(loss.detach())
                    dn += 1
        if dn:
            base = base + dl / dn
    model.train()
    return base


# ---------------------------------------------------------------------------
# main fitting routine
# ---------------------------------------------------------------------------
def fit(
    model: Cadence,
    tsets: dict[str, TensorSet],
    cfg: TrainConfig,
    train_shared: bool = True,
    animal_keys: list[str] | None = None,
    use_perturbed: bool = True,
    tag: str = "fit",
):
    """Fit the model. ``animal_keys`` lists the animals whose private parameters
    are optimised (all keys by default)."""
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device
    keys = list(tsets)
    animal_keys = keys if animal_keys is None else animal_keys

    # train / validation split, separately for unperturbed and perturbed trials
    split = {}
    for k in keys:
        ts = tsets[k]
        unp = ts.unp_idx.cpu().numpy()
        pert = ts.pert_idx.cpu().numpy()
        tr_u, va_u = _split_train_val(len(unp), cfg.val_frac, rng)
        tr_p, va_p = _split_train_val(len(pert), cfg.val_frac, rng) if len(pert) else ([], [])
        split[k] = {
            "train": {"unp": unp[tr_u], "pert": pert[tr_p] if len(pert) else np.array([], int)},
            "val": {"unp": unp[va_u], "pert": pert[va_p] if len(pert) else np.array([], int)},
        }
    train = {k: split[k]["train"] for k in keys}
    val = {k: split[k]["val"] for k in keys}

    shared_params = [p for p in model.shared_parameters()]
    animal_params = [p for k in animal_keys for p in model.animal_parameters(k)]
    opt_s = (
        torch.optim.AdamW(shared_params, lr=cfg.lr_shared, weight_decay=cfg.weight_decay)
        if train_shared and shared_params
        else None
    )
    opt_a = (
        torch.optim.AdamW(animal_params, lr=cfg.lr_animal, weight_decay=cfg.weight_decay)
        if animal_params
        else None
    )

    best = math.inf
    best_state = None
    bad = 0
    history = []
    total_epochs = cfg.epochs
    for ep in range(total_epochs):
        if cfg.cosine_schedule:
            frac = cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (
                1 + math.cos(math.pi * ep / max(total_epochs - 1, 1))
            )
            if opt_s is not None:
                for g in opt_s.param_groups:
                    g["lr"] = cfg.lr_shared * frac
            if opt_a is not None:
                for g in opt_a.param_groups:
                    g["lr"] = cfg.lr_animal * frac

        model.train()
        for _ in range(cfg.steps_per_epoch):
            k = keys[rng.integers(len(keys))]
            ts = tsets[k]
            pool_p = train[k]["pert"]
            pool_u = train[k]["unp"]
            # a delta step supervises the shared causal operator on the
            # condition-averaged effect; it uses intervention data, so (like any
            # perturbed batch) it never updates animal-private parameters
            want_delta = (
                use_perturbed
                and cfg.weight_delta > 0
                and len(pool_p) > 0
                and rng.random() < cfg.delta_batch_frac
            )
            if want_delta:
                loss, parts = delta_loss(model, ts, cfg, rng, pool_u)
                if loss is None:
                    continue
                want_pert = True
            else:
                want_pert = (
                    use_perturbed
                    and len(pool_p) > 0
                    and rng.random() < cfg.perturbed_batch_frac
                )
                pool = pool_p if want_pert else pool_u
                if len(pool) == 0:
                    continue
                sel = rng.choice(pool, size=min(cfg.batch_size, len(pool)), replace=False)
                idx = torch.as_tensor(sel, device=device)
                loss, parts = batch_loss(model, ts, idx, cfg, use_intervention=want_pert)

            if opt_s is not None:
                opt_s.zero_grad(set_to_none=True)
            if opt_a is not None:
                opt_a.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(shared_params + animal_params, cfg.grad_clip)
            if opt_s is not None:
                opt_s.step()
            # animal-private parameters are (by default) never updated from
            # intervention data, for any animal
            if opt_a is not None and not (want_pert and cfg.animal_params_unperturbed_only):
                if k in animal_keys:
                    opt_a.step()

        if ep % cfg.eval_every and ep != total_epochs - 1:
            continue
        vl = eval_loss(model, tsets, val, cfg, use_perturbed=use_perturbed)
        history.append(vl)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if cfg.verbose and (ep % cfg.log_every == 0 or ep == total_epochs - 1):
            print(f"    [{tag}] epoch {ep:4d} val {vl:.5f} best {best:.5f}", flush=True)
        if bad >= cfg.patience:
            if cfg.verbose:
                print(f"    [{tag}] early stop at epoch {ep}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val": best, "history": history, "split": split}


# ---------------------------------------------------------------------------
# the cross-animal protocol
# ---------------------------------------------------------------------------
def calibrate_animal(
    model: Cadence,
    tsets: dict[str, TensorSet],
    keys: list[str],
    cfg: TrainConfig,
    tag: str = "calib",
):
    """Fit a held-out animal's private parameters on its UNPERTURBED trials only.

    Shared parameters are frozen. Intervention trials from these keys are not
    used, and ``use_perturbed=False`` guarantees the intervention path is never
    even evaluated.
    """
    model.set_shared_grad(False)
    sub = {k: tsets[k] for k in keys}
    out = fit(
        model,
        sub,
        cfg,
        train_shared=False,
        animal_keys=keys,
        use_perturbed=False,
        tag=tag,
    )
    model.set_shared_grad(True)
    return out


def reset_animal(model: Cadence, key: str, seed: int = 0):
    """Re-initialise one animal's private parameters (used before calibration so
    that nothing from a previous fit can leak in). Per-unit features are static
    inputs, so they are preserved."""
    torch.manual_seed(seed)
    n_obs = model.cfg.obs_dims[key]
    dev = next(model.parameters()).device
    from .model import AnimalModule

    old = model.animals[key]
    feats = old.features if old.use_embed else None
    model.animals[key] = AnimalModule(model.cfg, n_obs, feats).to(dev)
    model._bind()


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(
    model: Cadence,
    ts: TensorSet,
    idx: np.ndarray | None = None,
    use_intervention: bool = True,
    batch: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    n = ts.y.shape[0]
    ids = np.arange(n) if idx is None else np.asarray(idx)
    outs: dict[str, list] = {}
    for j in range(0, len(ids), batch):
        sub = torch.as_tensor(ids[j : j + batch], device=ts.y.device)
        y_pre, _, u_pre, u_post, raw, on, _ = ts.slice_batch(sub)
        pred = model(
            ts.key, y_pre, u_pre, u_post,
            raw if use_intervention else None,
            on if use_intervention else None,
            y_pre.shape[1] * 0 + (ts.T - ts.t0),
            use_intervention=use_intervention,
        )
        for k, v in pred.items():
            outs.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


@torch.no_grad()
def condition_descriptor_tensors(ts: TensorSet, cond_id: int):
    """The (known) intervention waveform for a condition, as it will be applied.

    Only the *intervention parameters* are read -- never any neural or
    behavioural measurement from the intervention trials.
    """
    ids = torch.where(ts.cond == cond_id)[0]
    raw = ts.raw[ids, ts.t0 :].mean(0, keepdim=True)
    on = ts.on[ids, ts.t0 :].mean(0, keepdim=True)
    return raw, (on > 0.5).float()


@torch.no_grad()
def predicted_delta(
    model: Cadence,
    ts: TensorSet,
    field: str = "rate",
    init_from: str = "unperturbed",
    batch: int = 512,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Predicted causal effect per intervention condition.

    ``init_from='unperturbed'`` (default, and the protocol used for every headline
    number) draws initial conditions **exclusively from the held-out animal's
    unperturbed trials**. The intervention trials contribute nothing at all --
    not their spikes, not their behaviour, not even their pre-stimulus window.
    Only the intervention *parameters* (which are chosen by the experimenter and
    therefore known in advance) are used. This makes prediction and measurement
    statistically independent, so the split-half noise ceiling applies exactly.

    ``init_from='own'`` instead encodes each intervention trial's own
    pre-intervention window. Nothing about the response leaks, but prediction and
    measurement then share trial-level initial-condition noise, so this variant
    is reported only as a secondary, trial-resolved result.
    """
    cond = ts.cond.cpu().numpy()
    pert = ts.perturbed.cpu().numpy()
    unp_idx = np.where(~pert)[0]
    n_steps = ts.T - ts.t0

    if init_from == "unperturbed":
        # encode the unperturbed trials once; reuse those latent states for
        # every counterfactual intervention
        z0s = []
        for j in range(0, len(unp_idx), batch):
            sub = torch.as_tensor(unp_idx[j : j + batch], device=ts.y.device)
            y_pre, _, u_pre, _, _, _, _ = ts.slice_batch(sub)
            z0s.append(model.encode(ts.key, y_pre, u_pre))
        z0 = torch.cat(z0s)
        u_post = ts.u[unp_idx, ts.t0 :] if ts.u is not None else None
        base_pred = model.rollout(ts.key, z0, u_post, None, n_steps, use_intervention=False)
        base = base_pred[field].mean(0).cpu().numpy()
        out = {}
        for c in np.unique(cond[pert]):
            raw1, on1 = condition_descriptor_tensors(ts, int(c))
            raw = raw1.expand(z0.shape[0], -1, -1)
            on = on1.expand(z0.shape[0], -1)
            a = model.interv(raw, on)
            p = model.rollout(ts.key, z0, u_post, a, n_steps, use_intervention=True)
            out[int(c)] = p[field].mean(0).cpu().numpy() - base
        return out, base

    if init_from == "own":
        base = predict(model, ts, unp_idx, use_intervention=False)[field].mean(0)
        out = {}
        for c in np.unique(cond[pert]):
            ids = np.where(cond == c)[0]
            with_i = predict(model, ts, ids, use_intervention=True)[field].mean(0)
            without = predict(model, ts, ids, use_intervention=False)[field].mean(0)
            out[int(c)] = with_i - without
        return out, base

    raise ValueError(init_from)


@torch.no_grad()
def predicted_condition_mean(
    model: Cadence, ts: TensorSet, field: str = "rate", batch: int = 512
) -> dict[int, np.ndarray]:
    """Absolute (not differenced) predicted response per condition, using only
    unperturbed initial conditions."""
    cond = ts.cond.cpu().numpy()
    pert = ts.perturbed.cpu().numpy()
    unp_idx = np.where(~pert)[0]
    n_steps = ts.T - ts.t0
    z0s = []
    for j in range(0, len(unp_idx), batch):
        sub = torch.as_tensor(unp_idx[j : j + batch], device=ts.y.device)
        y_pre, _, u_pre, _, _, _, _ = ts.slice_batch(sub)
        z0s.append(model.encode(ts.key, y_pre, u_pre))
    z0 = torch.cat(z0s)
    u_post = ts.u[unp_idx, ts.t0 :] if ts.u is not None else None
    out = {}
    for c in np.unique(cond[pert]):
        raw1, on1 = condition_descriptor_tensors(ts, int(c))
        a = model.interv(raw1.expand(z0.shape[0], -1, -1), on1.expand(z0.shape[0], -1))
        p = model.rollout(ts.key, z0, u_post, a, n_steps, use_intervention=True)
        out[int(c)] = p[field].mean(0).cpu().numpy()
    return out
