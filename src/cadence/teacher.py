"""Teacher-RNN benchmark: synthetic animals with *known* shared dynamics,
known causal operator, known animal-specific observation maps and residual
dynamics, and a known partition of intervention directions into conserved and
idiosyncratic.

Two regimes are provided.

``shared``   One teacher RNN defines the species-invariant operator. Each animal
             inherits it, plus a small animal-specific low-rank perturbation of
             the recurrent weights (the residual dynamics) and its own random
             sub-sampled, gain-scaled Poisson observation map. Ground truth for
             every term in the CADENCE factorisation is available, so this
             regime measures *parameter recovery* and *identifiability*.

``heterogeneous``  Each animal is an *independently trained* RNN. No parameters
             are shared: only the task is. Interventions are specified
             functionally (along the readout direction, along the leading
             population PC, along the task-input direction) or idiosyncratically
             (along a random direction private to each network). This regime
             tests whether a causal operator estimated from other animals
             predicts intervention responses in a network whose circuit
             implementation was never seen -- the synthetic analogue of the
             biological claim.

``degenerate``  A deliberately symmetric teacher whose flow is invariant under
             rotations of a latent subspace. The symmetry group of the shared
             flow is non-trivial, so the observation map of a new animal is *not*
             identifiable from unperturbed activity and transfer must fail. This
             is the falsification control for the identifiability argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data.containers import AnimalTrials, Dataset
from .data.features import unit_features

# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------
# A delayed-response two-alternative task with graded sensory evidence.
#   inputs: [evidence_left, evidence_right, go_cue, fixation]
#   output: 2-dim choice readout, scored during the response epoch.


@dataclass
class TaskConfig:
    n_bins: int = 90
    sample: tuple[int, int] = (10, 30)
    delay: tuple[int, int] = (30, 60)
    response: tuple[int, int] = (60, 90)
    n_in: int = 4
    n_out: int = 2
    coherences: tuple[float, ...] = (0.15, 0.35, 0.7, 1.0)
    input_noise: float = 0.12


def make_task_inputs(cfg: TaskConfig, n_trials: int, rng: np.random.Generator):
    T = cfg.n_bins
    u = np.zeros((n_trials, T, cfg.n_in), dtype=np.float32)
    side = rng.integers(0, 2, n_trials)
    coh = rng.choice(cfg.coherences, n_trials)
    s0, s1 = cfg.sample
    for k in range(n_trials):
        u[k, s0:s1, side[k]] = coh[k]
        u[k, s0:s1, 1 - side[k]] = 0.0
    u[:, cfg.response[0] :, 2] = 1.0            # go cue
    u[:, : cfg.response[0], 3] = 1.0            # fixation
    u += rng.normal(0.0, cfg.input_noise, u.shape).astype(np.float32)
    target = np.zeros((n_trials, T, cfg.n_out), dtype=np.float32)
    for k in range(n_trials):
        target[k, cfg.response[0] :, side[k]] = 1.0
        target[k, cfg.response[0] :, 1 - side[k]] = -1.0
    mask = np.zeros((n_trials, T), dtype=np.float32)
    mask[:, cfg.response[0] :] = 1.0
    return u, target, mask, side, coh


# ---------------------------------------------------------------------------
# teacher RNN
# ---------------------------------------------------------------------------
class TeacherRNN(nn.Module):
    """x_{t+1} = x_t + (dt/tau) * (-x_t + W tanh(x_t) + B u_t + b + I_t + noise)."""

    def __init__(self, n_units: int = 64, n_in: int = 4, n_out: int = 2,
                 dt: float = 0.1, tau: float = 1.0, noise: float = 0.05, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_units, self.dt, self.tau, self.noise = n_units, dt, tau, noise
        self.W = nn.Parameter(torch.randn(n_units, n_units, generator=g) * (1.2 / n_units**0.5))
        self.B = nn.Parameter(torch.randn(n_units, n_in, generator=g) * (1.0 / n_in**0.5))
        self.b = nn.Parameter(torch.zeros(n_units))
        self.Wout = nn.Parameter(torch.randn(n_out, n_units, generator=g) * (1.0 / n_units**0.5))
        self.x0 = nn.Parameter(torch.zeros(n_units))

    def forward(self, u: torch.Tensor, inject: torch.Tensor | None = None,
                noise: bool | None = None):
        """u: (B, T, n_in); inject: (B, T, n_units) additive current."""
        B, T, _ = u.shape
        use_noise = self.noise if (noise is None or noise) else 0.0
        x = self.x0.expand(B, -1)
        xs, rs = [], []
        for t in range(T):
            r = torch.tanh(x)
            dx = -x + r @ self.W.t() + u[:, t] @ self.B.t() + self.b
            if inject is not None:
                dx = dx + inject[:, t]
            x = x + (self.dt / self.tau) * dx
            if use_noise:
                x = x + (2 * self.dt / self.tau) ** 0.5 * use_noise * torch.randn_like(x)
            xs.append(x)
            rs.append(torch.tanh(x))
        X = torch.stack(xs, 1)
        R = torch.stack(rs, 1)
        return X, R, R @ self.Wout.t()


def train_teacher(seed: int, task: TaskConfig, n_units: int = 64, steps: int = 900,
                  batch: int = 128, lr: float = 4e-3, device: str = "cuda",
                  verbose: bool = False) -> TeacherRNN:
    rng = np.random.default_rng(1000 + seed)
    net = TeacherRNN(n_units=n_units, n_in=task.n_in, n_out=task.n_out, seed=seed).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for it in range(steps):
        u, tgt, mask, _, _ = make_task_inputs(task, batch, rng)
        u_t = torch.as_tensor(u, device=device)
        tgt_t = torch.as_tensor(tgt, device=device)
        m_t = torch.as_tensor(mask, device=device).unsqueeze(-1)
        _, R, out = net(u_t)
        loss = ((out - tgt_t) ** 2 * m_t).sum() / m_t.sum() / tgt.shape[-1]
        loss = loss + 1e-4 * R.pow(2).mean() + 1e-4 * net.W.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if verbose and it % 200 == 0:
            print(f"  teacher {seed} step {it} loss {loss.item():.4f}", flush=True)
    return net.eval()


@torch.no_grad()
def teacher_accuracy(net: TeacherRNN, task: TaskConfig, n: int = 512, device: str = "cuda") -> float:
    rng = np.random.default_rng(7)
    u, _, _, side, _ = make_task_inputs(task, n, rng)
    _, _, out = net(torch.as_tensor(u, device=device), noise=True)
    resp = out[:, task.response[0] :].mean(1).cpu().numpy()
    return float((resp.argmax(1) == side).mean())


# ---------------------------------------------------------------------------
# intervention directions
# ---------------------------------------------------------------------------
INTERV_TYPES = ("readout", "pc1", "input", "idiosyncratic")


@torch.no_grad()
def intervention_directions(net: TeacherRNN, task: TaskConfig, rng: np.random.Generator,
                            device: str = "cuda") -> dict[str, np.ndarray]:
    """Functionally-defined (conserved) and random (idiosyncratic) directions.

    The conserved directions are defined by the *function* the network computes,
    so they are comparable across independently trained networks even though the
    networks share no parameters.
    """
    u, _, _, side, _ = make_task_inputs(task, 256, rng)
    X, R, _ = net(torch.as_tensor(u, device=device), noise=False)
    R = R.cpu().numpy()
    dirs = {}
    w = net.Wout.detach().cpu().numpy()
    dirs["readout"] = w[0] - w[1]
    flat = R[:, task.sample[0]:, :].reshape(-1, R.shape[-1])
    flat = flat - flat.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    dirs["pc1"] = vt[0]
    b = net.B.detach().cpu().numpy()
    dirs["input"] = b[:, 0] - b[:, 1]
    dirs["idiosyncratic"] = rng.normal(size=net.n_units)
    return {k: (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32) for k, v in dirs.items()}


# ---------------------------------------------------------------------------
# animals
# ---------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    regime: str = "heterogeneous"       # 'shared' | 'heterogeneous' | 'degenerate'
    n_animals: int = 8
    n_units: int = 64
    n_obs_range: tuple[int, int] = (28, 56)
    trials_per_condition: int = 24
    unperturbed_trials: int = 500
    amplitudes: tuple[float, ...] = (0.4, 0.8, 1.2, 1.6)
    interv_window: tuple[int, int] = (40, 55)   # bins, inside the delay epoch
    t0: int = 36                                # alignment: 4 bins before onset
    obs_gain: float = 1.4
    obs_bias: float = -0.4
    rate_scale: float = 1.0          # multiplies the Poisson rate (raises SNR)
    residual_scale: float = 0.06
    seed: int = 0
    device: str = "cuda"
    teacher_steps: int = 900


class DegenerateTeacher(nn.Module):
    """Teacher whose flow is invariant to rotations within 2-dimensional
    isotropic blocks. Used to falsify the identifiability claim."""

    def __init__(self, n_units: int = 64, n_in: int = 4, n_out: int = 2,
                 dt: float = 0.1, tau: float = 1.0, noise: float = 0.05, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        assert n_units % 2 == 0
        self.n_units, self.dt, self.tau, self.noise = n_units, dt, tau, noise
        # block-diagonal isotropic rotations: identical 2x2 blocks -> the flow
        # commutes with any rotation acting simultaneously in every block.
        blocks = []
        omega = 0.9
        for _ in range(n_units // 2):
            blocks.append(torch.tensor([[0.0, -omega], [omega, 0.0]]))
        W = torch.zeros(n_units, n_units)
        for i, blk in enumerate(blocks):
            W[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = blk
        self.register_buffer("Wfix", W)
        self.B = nn.Parameter(torch.randn(n_units, n_in, generator=g) * 0.0)
        self.b = nn.Parameter(torch.zeros(n_units))
        self.Wout = nn.Parameter(torch.randn(n_out, n_units, generator=g) * (1.0 / n_units**0.5))
        self.x0 = nn.Parameter(torch.randn(n_units, generator=g) * 0.5)

    def forward(self, u, inject=None, noise=None):
        B, T, _ = u.shape
        use_noise = self.noise if (noise is None or noise) else 0.0
        x = self.x0.expand(B, -1) + 0.3 * torch.randn(B, self.n_units, device=u.device)
        xs = []
        for t in range(T):
            dx = x @ self.Wfix.t() - 0.15 * x
            if inject is not None:
                dx = dx + inject[:, t]
            x = x + (self.dt / self.tau) * dx
            if use_noise:
                x = x + (2 * self.dt / self.tau) ** 0.5 * use_noise * torch.randn_like(x)
            xs.append(x)
        X = torch.stack(xs, 1)
        R = torch.tanh(X)
        return X, R, R @ self.Wout.t()


def _perturb_recurrent(net: TeacherRNN, scale: float, rng: np.random.Generator, rank: int = 3):
    """Animal-specific low-rank perturbation of the recurrent weights: the
    ground-truth residual dynamics."""
    n = net.n_units
    U = rng.normal(size=(n, rank)) / n**0.5
    V = rng.normal(size=(rank, n)) / n**0.5
    dW = scale * (U @ V)
    return torch.as_tensor(dW, dtype=torch.float32), (U, V)


def build_synthetic_dataset(cfg: SyntheticConfig) -> tuple[Dataset, dict]:
    """Simulate ``cfg.n_animals`` animals and return trial tensors plus a
    ground-truth bundle for the recovery analyses."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    task = TaskConfig()
    device = cfg.device

    # ---- teachers ----
    teachers: list[nn.Module] = []
    if cfg.regime == "shared":
        base = train_teacher(0, task, cfg.n_units, steps=cfg.teacher_steps, device=device)
        teachers = [base for _ in range(cfg.n_animals)]
    elif cfg.regime == "heterogeneous":
        for k in range(cfg.n_animals):
            teachers.append(
                train_teacher(k, task, cfg.n_units, steps=cfg.teacher_steps, device=device)
            )
    elif cfg.regime == "degenerate":
        base = DegenerateTeacher(cfg.n_units, task.n_in, task.n_out, seed=0).to(device).eval()
        teachers = [base for _ in range(cfg.n_animals)]
    else:
        raise ValueError(cfg.regime)

    acc = [teacher_accuracy(t, task, device=device) for t in teachers[: cfg.n_animals]]
    for t in teachers:
        for p in t.parameters():
            p.requires_grad_(False)

    sets: list[AnimalTrials] = []
    truth: dict = {"regime": cfg.regime, "task_accuracy": acc, "animals": {}}
    T = task.n_bins
    w0, w1 = cfg.interv_window

    for i in range(cfg.n_animals):
        akey = f"rnn{i:02d}"
        net = teachers[i]
        arng = np.random.default_rng(cfg.seed * 977 + i)

        # animal-specific residual dynamics (only in the 'shared' regime, where
        # all animals otherwise have identical recurrent weights)
        dW = None
        res_gt = None
        if cfg.regime == "shared" and cfg.residual_scale > 0:
            dW_t, res_gt = _perturb_recurrent(net, cfg.residual_scale, arng)
            dW = dW_t.to(device)

        class _Wrapped(nn.Module):
            def __init__(self, base, dW):
                super().__init__()
                self.base, self.dW = base, dW

            def forward(self, u, inject=None, noise=None):
                if self.dW is None:
                    return self.base(u, inject, noise)
                B, Tn, _ = u.shape
                use_noise = self.base.noise if (noise is None or noise) else 0.0
                W = self.base.W + self.dW
                x = self.base.x0.expand(B, -1)
                xs = []
                for t in range(Tn):
                    r = torch.tanh(x)
                    dx = -x + r @ W.t() + u[:, t] @ self.base.B.t() + self.base.b
                    if inject is not None:
                        dx = dx + inject[:, t]
                    x = x + (self.base.dt / self.base.tau) * dx
                    if use_noise:
                        x = x + (2 * self.base.dt / self.base.tau) ** 0.5 * use_noise * torch.randn_like(x)
                    xs.append(x)
                X = torch.stack(xs, 1)
                R = torch.tanh(X)
                return X, R, R @ self.base.Wout.t()

        sim = _Wrapped(net, dW).to(device).eval()

        # observation map: random sub-sampling with random gains (animal-specific)
        n_obs = min(int(arng.integers(*cfg.n_obs_range)), cfg.n_units)
        sel = arng.choice(cfg.n_units, size=n_obs, replace=False)
        gains = np.exp(arng.normal(0.0, 0.35, size=n_obs)).astype(np.float32)
        offs = arng.normal(cfg.obs_bias, 0.25, size=n_obs).astype(np.float32)
        C_gt = np.zeros((n_obs, cfg.n_units), dtype=np.float32)
        C_gt[np.arange(n_obs), sel] = gains * cfg.obs_gain

        dirs = intervention_directions(net, task, arng, device=device)

        # ---- assemble trial list ----
        conds: list[tuple[str, float]] = [("none", 0.0)]
        for tname in INTERV_TYPES:
            for amp in cfg.amplitudes:
                conds.append((tname, float(amp)))

        ys, us, raws, ons, behs, perts, condids = [], [], [], [], [], [], []
        locals_store: dict = {}
        for ci, (tname, amp) in enumerate(conds):
            n = cfg.unperturbed_trials if tname == "none" else cfg.trials_per_condition
            u_np, _, _, side, coh = make_task_inputs(task, n, arng)
            u_t = torch.as_tensor(u_np, device=device)
            inject = torch.zeros(n, T, cfg.n_units, device=device)
            if tname != "none":
                d = torch.as_tensor(dirs[tname], device=device)
                inject[:, w0:w1, :] = amp * d
            X, R, out = sim(u_t, inject=inject, noise=True)
            if tname == "none" and "R_unperturbed" not in locals_store:
                # keep the teacher's true latent rates for the first unperturbed
                # trials: ground truth for the identifiability analyses
                locals_store["R_unperturbed"] = R[:256].cpu().numpy().astype(np.float32)
                locals_store["X_unperturbed"] = X[:256].cpu().numpy().astype(np.float32)
            rate = torch.nn.functional.softplus(
                torch.as_tensor(C_gt, device=device) @ R.transpose(1, 2)
            ).transpose(1, 2)                                   # (n, T, n_obs)
            rate = (rate + torch.as_tensor(offs, device=device)).clamp_min(1e-3)
            rate = rate * cfg.rate_scale
            y = torch.poisson(rate)
            ys.append(y.cpu().numpy().astype(np.float32))
            us.append(u_np)
            raw = np.zeros((n, T, 1 + len(INTERV_TYPES)), dtype=np.float32)
            on = np.zeros((n, T), dtype=np.float32)
            if tname != "none":
                raw[:, w0:w1, 0] = amp
                raw[:, w0:w1, 1 + INTERV_TYPES.index(tname)] = 1.0
                on[:, w0:w1] = 1.0
            raws.append(raw)
            ons.append(on)
            behs.append(out.cpu().numpy().astype(np.float32))
            perts.append(np.full(n, tname != "none", dtype=bool))
            condids.append(np.full(n, ci, dtype=np.int64))

        y_all = np.concatenate(ys)
        pert_all = np.concatenate(perts)
        feats = unit_features(y_all[~pert_all], depth_um=None, cell_type=None)
        sets.append(
            AnimalTrials(
                unit_features=feats,
                key=akey,
                animal=akey,
                y=np.concatenate(ys),
                u=np.concatenate(us),
                interv_raw=np.concatenate(raws),
                interv_on=np.concatenate(ons),
                behavior=np.concatenate(behs),
                perturbed=np.concatenate(perts),
                t0=cfg.t0,
                bin_s=0.01,
                cond=np.concatenate(condids),
                meta={"conds": conds, "interv_types": INTERV_TYPES},
            )
        )
        truth["animals"][akey] = {
            "R_unperturbed": locals_store.get("R_unperturbed"),
            "X_unperturbed": locals_store.get("X_unperturbed"),
            "C": C_gt,
            "sel": sel,
            "gains": gains,
            "offs": offs,
            "dirs": dirs,
            "residual": res_gt,
            "n_obs": n_obs,
        }

    ds = Dataset(
        name=f"teacher-{cfg.regime}",
        sets=sets,
        n_u=task.n_in,
        n_raw=1 + len(INTERV_TYPES),
        n_beh=task.n_out,
        bin_s=0.01,
        interv_names=("amplitude",) + INTERV_TYPES,
        behavior_names=("choice_left", "choice_right"),
    )
    truth["conds"] = sets[0].meta["conds"]
    truth["interv_window"] = (w0, w1)
    truth["task"] = task
    return ds, truth
