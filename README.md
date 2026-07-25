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

### The bottleneck is the readout, not the operator

Hold the shared causal operator **completely fixed** at the value fitted on the other
animals, and grant the held-out animal only a rescaling of the predicted response,
fitted on its intervention trials. This is not a prediction — it uses data the
protocol forbids — but it is a tightly constrained decomposition: a per-unit scalar
cannot invent structure in time or across conditions, because each unit's predicted
time course and dose dependence are fixed by the shared operator.

| readout freedom granted | parameters | ΔR² | sessions >0 |
|---|---|---|---|
| none (honest zero-shot) | 0 | +0.003 | 31/48 |
| **one gain per unit** | *Nᵢ* | **+0.405** | **48/48** |
| gain and offset per unit | 2*Nᵢ* | +0.494 | 48/48 |
| gain per unit + shared timecourse | *Nᵢ*+*T* | +0.465 | 48/48 |

One scalar per unit lifts ΔR² from +0.003 to **+0.405** (paired difference +0.402,
*p* = 7×10⁻¹⁵, all 6 animals positive). **The shared causal operator is already
substantially correct** — it predicts *which pattern in time and across conditions*
each unit will show — and gets only the per-unit amplitude wrong. That amplitude is
not recoverable from spontaneous activity in ~20 simultaneously recorded units. Four
qualitatively different models (latent CADENCE, an observed-space linear-response
model, a physical-feature encoding model, a manifold-alignment group average) all
plateau at *r* ≈ 0.28 across animals.

And we know why, for this preparation: pooled over all sessions, a unit's response is
essentially **uncorrelated with its distance from the stimulating contact**
(*r*(|Δdepth|, Δrate) = −0.013). Low-amplitude ICMS recruits a sparse, spatially
diffuse, implant-specific set of neurons, so there is no conserved neuron-level
spatial rule for any model to learn.

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

Two consequences:

1. **With a free per-animal observation map, transfer needs the shared flow to be
   asymmetric.** Degenerate (isotropic, rotationally symmetric) dynamics leave a
   reparameterisation the readout can absorb. But the freedom can be removed by
   *construction*: generating observation maps from a shared function of per-unit
   features is exactly what stops the readout absorbing a rotation. This is why our
   deliberately symmetric teacher still transfers — the degeneracy is broken by the
   parameterisation, not by the data — and it is why tying observation maps across
   animals is what makes transfer possible at all.
2. **Spontaneous cortical activity sits near a fixed point**, where the flow is weak
   and nearly isotropic. So the freedom left in a new animal's readout is large, and
   transfer is then only as good as the per-unit features are informative about that
   unit's causal coupling. In this preparation they are not (see above), which
   predicts difficulty exactly where we measure it.

Teacher-RNN simulations (5 animals per regime, so these are existence checks rather
than precise measurements) bear both branches out: zero-shot transfer is positive
where animals genuinely share dynamics (ΔR² = +0.230, 5/5 folds when teachers are
identical; +0.159, 3/5 when they also carry animal-specific weight perturbations),
negative between independently trained networks sharing only the task (−0.104, 1/5),
and every negative control is below zero. The oracle is far above zero-shot in all
three regimes (+0.29 to +0.74) — the model can express these responses; calibrating
the held-out animal is what limits it. Informatively, a *deliberately symmetric*
teacher still transfers: generating observation maps from a shared function of
per-unit features removes exactly the freedom the symmetry would exploit, so the
degeneracy is broken by the parameterisation rather than by the data.

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
