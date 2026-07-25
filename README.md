# Do individuals share a causal dynamical operator?

**Zero-shot transfer of intervention responses between animals.**

Latent neural dynamics look alike across individuals doing the same thing, and many
methods now align recordings across animals into a shared representation. Those
results are *descriptive*: they say observed activity occupies a common geometry.
They do not say that different individuals share a common **causal** operator — that
the same physical intervention would move each brain the same way.

This repository turns that stronger claim into a prediction problem with a hard
protocol:

> After observing **only normal, unperturbed activity** from a new animal, predict
> the **full time-resolved neural and behavioural response** to an intervention
> that animal has never received.

No intervention trial from the held-out animal is used at any point — not its
spikes, not its behaviour, not even its pre-stimulus window. Only the intervention
*parameters*, which the experimenter chooses, are known in advance.

---

## Headline results

Real data: **6 mice**, **48 sessions**, **949 single units**, **16,532 stimulation
trials**, parameterised intracortical microstimulation of mouse S1 during a
detection task ([DANDI:001868](https://dandiarchive.org/dandiset/001868)).
Leave-one-animal-out. `ΔR² = 0` is exactly the "the intervention does nothing"
model; `1` is perfect.

| readout | in-sample | new session | **new animal** | new animal + unseen amplitude | ceiling |
|---|---|---|---|---|---|
| single units | +0.329 | +0.172 | **+0.003** | +0.011 | 0.914 |
| depth bands | +0.572 | +0.359 | **+0.115** | +0.072 | 0.938 |
| population rate | +0.965 | +0.598 | **+0.147** | +0.164 | 0.953 |
| wheel speed | +0.833 | −0.113 | **+0.045** | +0.016 | 0.370 |
| **detection probability** | +0.970 | +0.194 | **+0.577** | **+0.321** | 0.968 |

The answer is **granularity-dependent**, and that is the finding:

- **Behaviour transfers.** The time-resolved probability that the animal has
  reported detection by time *t* is predicted in a completely new animal at
  **ΔR² = 0.577** (95% CI [0.471, 0.669]), **60% of the noise ceiling**,
  *p* = 7×10⁻¹⁵ (paired permutation vs. no-effect; Wilcoxon *p* = 6×10⁻¹¹).
  Positive in **all 6 animals** and 43/48 sessions. Still positive
  (**ΔR² = 0.321**) when the stimulation amplitude is *also* deleted from every
  training animal — a new individual *and* a new intervention setting.
- **Coarse-grained neural activity transfers partially.** Population rate:
  ΔR² = 0.147 with a high shape correlation *r* = 0.669 — the time course and dose
  dependence are conserved, the overall gain varies between animals.
- **Single units do not transfer.** ΔR² = 0.003, indistinguishable from the
  no-effect model (*p* = 0.92), despite a 0.914 ceiling and despite being
  well predicted within a session (0.329) and partly within animal across
  sessions (0.172).

The bottleneck is **not** the causal operator. It is the animal-specific map
between individual neurons and the conserved latent state, which is not
recoverable from spontaneous activity in ~20 simultaneously recorded units. Four
qualitatively different models (latent CADENCE, an observed-space linear-response
model, a physical-feature encoding model, a manifold-alignment group average) all
plateau at *r* ≈ 0.28 across animals.

---

## Why single units fail: an identifiability result

The model factorises the dynamics as

```
z_{t+1} = z_t + Δt [ F_shared(z_t, u_t) + G_shared(z_t) a_t + F_res_i(z_t) ]
y_t^(i) = Poisson( softplus( C_i z_t + b_i ) )
b_t     = D_shared(z_t)
```

**Proposition.** Fix `F` and `G`. If animal *i*'s unperturbed law is matched by
both `(C_i, F_res_i)` and `(C_i', F_res_i')`, then the two differ by some
`T ∈ Sym(F) = {T : T F(z,u) = F(Tz,u) ∀ z,u}`, and their predicted intervention
responses differ by `C_i T⁻¹ (T G(Tz) a − G(z) a)`. So the predicted response is
**unique iff `Sym(F)` is trivial**, and otherwise ambiguous by exactly that group.

Two consequences, both verified:

1. **Rich, asymmetric shared dynamics make transfer identifiable; degenerate
   (isotropic/rotationally symmetric) dynamics make it impossible.** The
   teacher-RNN benchmark includes a deliberately symmetric teacher as a
   falsification control.
2. **Spontaneous cortical activity sits near a fixed point**, where the flow is
   weak and nearly isotropic — effectively the degenerate case. This predicts
   difficulty exactly where we measure it.

An observed-space **linear-response** formulation avoids latent frames entirely:
estimate each animal's propagator `A_i` from its own unperturbed activity and share
only the *drive*, so `Δ_i(t) = Σ_k A_i^k u_{t−k}` — a discrete-time
fluctuation–response statement, fitted in closed form
(`src/cadence/linear_response.py`).

---

## Rigour

Everything the claim depends on is checked, and the checks are in the repo.

- **Validated noise ceilings.** The split-half ceiling estimator is tested against
  simulations with known signal and noise across six noise/trial-count regimes
  (agreement within 0.06): `tests/test_metrics.py`.
- **Covariate-matched unperturbed data.** Stimulation trials are delivered under a
  quiescence criterion, so uniformly sampled inter-trial windows have up to **3×**
  the pre-stimulus population rate. That would bias every measured effect, since
  predictions draw initial conditions from unperturbed trials. Candidate windows
  are matched to stimulation trials on a joint quantile grid of pre-stimulus
  population rate and wheel speed, then mean-trimmed; sessions with residual
  mismatch > 1.25× are **excluded**. The audit reports **0** remaining problems.
- **Leakage audit** on every session: the intervention gate must be identically
  zero before the alignment index and on all unperturbed trials.
- **Bit-for-bit reproducible.** Seeds are derived with `zlib.crc32`, not Python's
  per-process-salted `hash`; two independent builds agree exactly.
- **Negative controls.** Permuted unit identities, scrambled intervention labels.
- **Statistics.** Bootstrap CIs over sessions, paired permutation tests (with the
  exact 2^−(n−1) floor) and Wilcoxon signed-rank.
- **Numbers cannot drift.** `scripts/make_paper_numbers.py` emits every figure
  quoted in the paper as a LaTeX macro straight from the result JSONs.

---

## Reproducing

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python -e . \
    numpy scipy matplotlib scikit-learn pandas h5py pynwb pyyaml seaborn statsmodels tqdm
VIRTUAL_ENV=.venv uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128

# 1. data (7.5 GB, public, CC-BY-4.0) + checksummed manifest
.venv/bin/python scripts/download_dandi.py --out data/raw/dandi001868

# 2. analysis tensors + leakage audit  (writes results/tables/icms_audit.json)
.venv/bin/python scripts/build_icms_cache.py

# 3. the generalisation ladder  (Table 2 of the paper)
.venv/bin/python scripts/run_icms_ladder.py

# 4. teacher-RNN benchmark with ground truth  (Table 1)
.venv/bin/python scripts/run_teacher.py --regime shared        --identifiability
.venv/bin/python scripts/run_teacher.py --regime heterogeneous --identifiability
.venv/bin/python scripts/run_teacher.py --regime degenerate

# 5. figures, numbers, paper
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_paper_numbers.py
cd paper && pdflatex main && pdflatex main
```

Diagnostics that localise the difficulty:

```bash
.venv/bin/python scripts/probe_transfer_levels.py     # within-session / cross-session / cross-animal
.venv/bin/python scripts/probe_granularity.py         # readout granularity
.venv/bin/python scripts/probe_behavior_transfer.py   # behaviour, per channel
.venv/bin/python scripts/probe_physical_transfer.py   # physical-feature baseline
```

Tests: `.venv/bin/python -m pytest tests/ -q`

---

## Layout

```
src/cadence/
  model.py             CADENCE: shared F, shared state-dependent G, shared unit
                       embedding, animal-specific residual dynamics
  linear_response.py   observed-space fluctuation-response model (closed form)
  training.py          training + the calibration protocol (unperturbed only)
  experiment.py        leave-one-animal-out engine, baselines, controls, oracle
  holdout.py           "previously unseen intervention" made precise
  metrics.py           ΔR², validated split-half ceilings, bootstrap, permutation
  baselines.py         manifold alignment, aligned group mean, unit encoding model
  teacher.py           teacher-RNN generator: shared / heterogeneous / degenerate
  figures.py           publication figures
  data/                NWB loaders, containers, per-unit features
scripts/               download, build, run, probe, figures, paper numbers
paper/                 LaTeX source, auto-generated numbers.tex, figures
results/               JSON results, tables, figures
tests/                 metric and pipeline tests
```

## Data

[DANDI:001868](https://dandiarchive.org/dandiset/001868) — *Chronic
electrophysiology and two-photon calcium imaging of mouse primary somatosensory
cortex during intracortical microstimulation learning* (CC-BY-4.0). Raw data are
never committed; `scripts/download_dandi.py` reconstructs them and writes a
SHA-256 manifest.

## Limitations

One species, one task, one perturbation modality. Six animals bound the precision
of the confidence intervals. Simultaneously recorded populations are small (8–33
units) — our diagnosis predicts that unit-level transfer should improve with
population size, which is the sharp, testable consequence of this work. Electrical
microstimulation is coarse and not cell-type-specific.

## License

MIT (code). The dataset is CC-BY-4.0 and belongs to its authors.
