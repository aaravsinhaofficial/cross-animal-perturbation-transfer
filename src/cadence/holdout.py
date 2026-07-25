"""Intervention holdouts: making "a previously unseen intervention" precise.

A held-out animal contributes no intervention data at all, so every response we
predict in it is unobserved. But the *intervention setting* itself (say 6 uA on a
contact 240 um below the surface) might still have been seen in other animals. To
test the stronger claim -- an intervention setting that appears nowhere in
training, evaluated in an animal that appears nowhere in training -- the setting
is additionally deleted from every training animal.

Holdout kinds
-------------
``none``               every intervention setting is available during training
``amplitude``          named amplitudes deleted from all training animals
``amplitude_extrap``   all amplitudes above a threshold deleted (extrapolation
                       beyond the trained dose range, not interpolation)
``depth``              stimulation contacts in a depth band deleted
``type``               named intervention types deleted (teacher benchmark)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.containers import AnimalTrials, Dataset


@dataclass
class InterventionHoldout:
    kind: str = "none"
    amplitudes: tuple[float, ...] = ()
    amp_threshold: float | None = None
    depth_band_um: tuple[float, float] | None = None
    types: tuple[str, ...] = ()
    tol: float = 1e-6

    @property
    def active(self) -> bool:
        return self.kind != "none"

    def describe(self) -> str:
        if self.kind == "none":
            return "none"
        if self.kind == "amplitude":
            return f"amplitude in {self.amplitudes}"
        if self.kind == "amplitude_extrap":
            return f"amplitude > {self.amp_threshold}"
        if self.kind == "depth":
            return f"depth in {self.depth_band_um} um"
        if self.kind == "type":
            return f"type in {self.types}"
        return self.kind


def _cond_attrs(s: AnimalTrials, cond_id: int) -> dict:
    """Physical attributes of a condition, whichever dataset it came from."""
    meta = s.meta or {}
    out: dict = {}
    if "cond_amp" in meta:
        amp = meta["cond_amp"]
        out["amplitude"] = float(amp.get(cond_id, amp.get(str(cond_id), np.nan)))
    if "cond_depth_um" in meta:
        dep = meta["cond_depth_um"]
        out["depth_um"] = float(dep.get(cond_id, dep.get(str(cond_id), np.nan)))
    if "conds" in meta:  # teacher benchmark
        name, amp = meta["conds"][cond_id]
        out["type"] = name
        out["amplitude"] = float(amp)
    return out


def is_held_out(s: AnimalTrials, cond_id: int, spec: InterventionHoldout) -> bool:
    if not spec.active:
        return False
    a = _cond_attrs(s, cond_id)
    if spec.kind == "amplitude":
        v = a.get("amplitude")
        return v is not None and any(abs(v - x) <= spec.tol for x in spec.amplitudes)
    if spec.kind == "amplitude_extrap":
        v = a.get("amplitude")
        return v is not None and spec.amp_threshold is not None and v > spec.amp_threshold + spec.tol
    if spec.kind == "depth":
        v = a.get("depth_um")
        lo, hi = spec.depth_band_um
        return v is not None and np.isfinite(v) and lo <= v <= hi
    if spec.kind == "type":
        return a.get("type") in spec.types
    raise ValueError(spec.kind)


def strip_training_conditions(s: AnimalTrials, spec: InterventionHoldout) -> AnimalTrials | None:
    """Delete held-out intervention conditions from a *training* animal."""
    if not spec.active:
        return s
    drop = np.zeros(s.n_trials, dtype=bool)
    for c in np.unique(s.cond[s.perturbed]):
        if is_held_out(s, int(c), spec):
            drop |= s.cond == c
    keep = np.where(~drop)[0]
    if len(keep) == 0:
        return None
    out = s.subset(keep)
    if not out.perturbed.any():
        return None
    return out


def eval_conditions(s: AnimalTrials, spec: InterventionHoldout) -> list[int]:
    """Conditions of a *test* animal that the holdout says to score."""
    conds = [int(c) for c in np.unique(s.cond[s.perturbed])]
    if not spec.active:
        return conds
    return [c for c in conds if is_held_out(s, c, spec)]


def summarise_holdout(ds: Dataset, spec: InterventionHoldout) -> dict:
    per_animal = {}
    for s in ds.sets:
        held = eval_conditions(s, spec)
        avail = [int(c) for c in np.unique(s.cond[s.perturbed])]
        per_animal[s.key] = {
            "n_conditions": len(avail),
            "n_held_out": len(held),
            "held_out": held,
            "attrs": {c: _cond_attrs(s, c) for c in avail},
        }
    return {"spec": spec.describe(), "per_set": per_animal}
