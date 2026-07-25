"""Common container for multi-animal, intervention-annotated trial tensors.

Every dataset (teacher RNN, ICMS, ALM photoinhibition) is reduced to a list of
``AnimalTrials``. Time is discretised into fixed bins; every trial in a set
shares the same alignment index ``t0`` (the intervention onset bin), so that
``y[:, :t0]`` is guaranteed pre-intervention and ``y[:, t0:]`` is the window the
model has to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnimalTrials:
    """All trials from one animal (optionally one recording session)."""

    key: str                      # unique observation-map key, e.g. "sub-ICMS93/2023-09-14"
    animal: str                   # animal identity used for the leave-one-animal-out split
    y: np.ndarray                 # (n_trials, T, n_obs) spike counts or dF/F
    u: np.ndarray | None          # (n_trials, T, n_u) exogenous task input
    interv_raw: np.ndarray        # (n_trials, T, n_raw) physical intervention descriptor
    interv_on: np.ndarray         # (n_trials, T) gate in {0,1}
    behavior: np.ndarray | None   # (n_trials, T, n_beh)
    perturbed: np.ndarray         # (n_trials,) bool
    t0: int                       # alignment index (intervention onset bin)
    bin_s: float                  # bin width in seconds
    cond: np.ndarray | None = None  # (n_trials,) integer condition id for grouping
    meta: dict | None = None
    # (n_obs, n_feat) per-unit features, computed from UNPERTURBED data and static
    # metadata only; consumed by the shared unit embedding in the model.
    unit_features: np.ndarray | None = None

    def __post_init__(self):
        n = self.y.shape[0]
        assert self.interv_raw.shape[0] == n and self.interv_on.shape[0] == n
        assert self.perturbed.shape[0] == n
        if self.cond is None:
            self.cond = np.zeros(n, dtype=np.int64)

    @property
    def n_trials(self) -> int:
        return self.y.shape[0]

    @property
    def T(self) -> int:
        return self.y.shape[1]

    @property
    def n_obs(self) -> int:
        return self.y.shape[2]

    def subset(self, idx: np.ndarray) -> AnimalTrials:
        idx = np.asarray(idx)
        return AnimalTrials(
            key=self.key,
            animal=self.animal,
            y=self.y[idx],
            u=None if self.u is None else self.u[idx],
            interv_raw=self.interv_raw[idx],
            interv_on=self.interv_on[idx],
            behavior=None if self.behavior is None else self.behavior[idx],
            perturbed=self.perturbed[idx],
            t0=self.t0,
            bin_s=self.bin_s,
            cond=self.cond[idx],
            meta=self.meta,
            unit_features=self.unit_features,
        )

    @property
    def unperturbed(self) -> AnimalTrials:
        return self.subset(np.where(~self.perturbed)[0])

    @property
    def perturbed_only(self) -> AnimalTrials:
        return self.subset(np.where(self.perturbed)[0])


@dataclass
class Dataset:
    """A collection of ``AnimalTrials`` plus dataset-level metadata."""

    name: str
    sets: list[AnimalTrials]
    n_u: int
    n_raw: int
    n_beh: int
    bin_s: float
    interv_names: tuple[str, ...] = ()
    behavior_names: tuple[str, ...] = ()

    @property
    def animals(self) -> list[str]:
        seen, out = set(), []
        for s in self.sets:
            if s.animal not in seen:
                seen.add(s.animal)
                out.append(s.animal)
        return out

    def for_animals(self, animals) -> list[AnimalTrials]:
        animals = set(animals)
        return [s for s in self.sets if s.animal in animals]

    def summary(self) -> str:
        lines = [f"{self.name}: {len(self.animals)} animals, {len(self.sets)} observation sets"]
        for s in self.sets:
            lines.append(
                f"  {s.key:38s} animal={s.animal:14s} n_obs={s.n_obs:4d} "
                f"trials={s.n_trials:5d} (pert={int(s.perturbed.sum()):5d}) T={s.T} t0={s.t0}"
            )
        return "\n".join(lines)
