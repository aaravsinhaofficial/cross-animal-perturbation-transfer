# Brains share the rule, but not the wiring

Show a model several animals being perturbed. Then show it a new animal doing nothing
in particular. From that ordinary activity, plus the settings of a perturbation about
to be delivered, the model has to say what will happen next in an animal it has never
seen perturbed.

We ran that test on two cohorts of mice and got two different answers. The difference
between them is the result.

## Two ways of pushing on a cortex

**Light in frontal cortex.** 20 mice, 109 sessions, anterior lateral motor cortex
silenced with light during the delay period of a decision task
([DANDI:000009](https://dandiarchive.org/dandiset/000009)).

**Current in somatosensory cortex.** 6 mice, 48 sessions, a chronically implanted probe
delivering microstimulation the animal has learned to report
([DANDI:001868](https://dandiarchive.org/dandiset/001868)).

Both perturbations are described by numbers the experimenter chose, so an unseen
intervention has a meaning the model can be handed.

## Why the second number is the interesting one

A score on the whole response flatters everybody. A perturbation moves a population in
a broadly stereotyped way, so a model can look good while handing every neuron the same
answer. So we split the measured effect into the part the population shares and the
part specific to one neuron,

```
Delta_n(t)  =  mean over neurons of Delta(t)  +  delta_n(t)
```

and score models on `delta` alone. A stereotype borrowed from other animals scores
exactly zero here, not approximately zero, because its prediction does not vary across
neurons. Anything above zero is prediction of individual structure in an animal whose
perturbation trials were never read.

## What we found

**On the whole response, under light, the model wins.** It scores +0.134 against +0.039
for the average of the other mice, a paired difference of +0.095 at p = 3.5e-4 over 20
animals, and on the best measured mice it reaches 0.6.

**On the individual part, it still wins, but by much less.** A shared operator acting on
each neuron's own ordinary activity lands above zero in 15 of 20 mice (exact sign test,
p = 0.041), against a measurable maximum of 0.57. So a few per cent of what is
individual about a neuron's response transfers, and most of it does not.

**Under current, nothing transfers at either level**, and this is not a measurement
problem. The measurable maximum for the individual part there is 0.90, higher than in
the light cohort, so it is sitting there clearly resolved and no rule we fit reaches it.

## The mistake we nearly made

The measured effect is the perturbed mean minus the control mean, and the model is
handed each neuron's control activity as an input. If the same trials produce both,
their noise enters the input and the target with opposite signs, and a model scores
above zero knowing nothing at all.

We caught this after it had already produced a result we liked. Splitting the control
trials, so one half builds every input and the other half defines the target, cut the
headline number roughly in half. Every number here is from the split version, and the
whole analysis rerun with the perturbed trials replaced by a second group of control
trials, where the answer is known to be zero, returns zero.

## How many animals an operator needs

Fitting the shared operator on random subsets of the training animals gives an orderly
curve. One animal scores −1.09, which is far worse than predicting nothing, because a
single animal's idiosyncrasies get mistaken for a rule. Five animals still sit below
zero. It first becomes useful at around eight, and by nineteen it reaches +0.05, still
rising, tracking the log of the cohort size at r = +0.86.

A study with five animals would have concluded that nothing transfers.

## A simulated cortex

Turning the recruitment from a smooth rule of position into a scattered set redrawn for
each animal costs about 40% of the recoverable individual part. It does not drive it to
zero, because in the simulation a cell's ongoing rate reports how strongly the network
drives it and therefore how much a perturbation will move it, and that rule survives
private recruitment.

The microstimulation cohort sits below the worst case the simulator can produce. So
private recruitment is part of the story there and not all of it: for electrical
stimulation, a neuron's ongoing rate does not say how much the stimulus will move it,
and under light it does.

## Animals are the sample size

Several sessions from one mouse are still one mouse, so every headline number is
inferred at the animal level, with a bootstrap that resamples animals and an exact sign
flip permutation over them. With six animals the smallest p value such a test can return
is 0.031; with twenty it is 1.9e-6.

An earlier version of this work reported session level statistics as the headline, which
overstated the evidence. Correcting it changed the behavioural p value from 7e-15 to
0.031.

## Reproducing

```bash
make env        # python 3.12 venv and dependencies
make cache      # download both dandisets, build tensors, audit
make analysis   # the results table, with animal-level statistics
make operator   # leave-one-animal-out training of the shared operator
make individual # the decomposition into shared and individual parts
make cortex     # the simulated cortex sweeps
make paper      # figures, numbers and the PDF
make test       # tests
```

Raw files are never committed; the download scripts rebuild them with a SHA-256
manifest.

## What is checked

The split-half ceiling estimator is validated against simulations with known signal and
noise across six regimes, agreeing with the attainable score to within 0.06. The noise
ceiling for the individual part is computed two independent ways, analytically from
trial counts and by splitting trials, and they agree.

Stimulation trials in the microstimulation cohort are delivered under a quiescence
criterion, so randomly sampled inter-trial windows carry up to three times the
pre-stimulus firing rate of a real trial. Windows are matched to stimulation trials on a
joint quantile grid of pre-stimulus rate and wheel speed, then trimmed to match on the
mean, and any session still mismatched by more than a factor of 1.25 is dropped.

Seeds use a stable checksum rather than Python's per-process hash, so two independent
builds agree exactly and the results table reproduces bit for bit.

## Layout

```
src/cadence/
  operator2.py         the shared operator, written as a correction to the stereotype
  individuality.py     the split into shared and individual parts, and the ceiling
  model.py             the hierarchical model: shared dynamics, shared stimulus
                       operator, animal-specific observation maps
  linear_response.py   each animal's propagator from its own resting activity,
                       convolved with a shared drive
  synthetic_cortex.py  the simulated cortex used to test the mechanism
  metrics.py           scoring, validated split-half ceilings, animal-level statistics
  data/                loaders for both cohorts, containers, per-unit features
scripts/               download, build, analysis, probes, figures, paper
paper/                 LaTeX source with auto-generated numbers and tables
results/               JSON results and tables
```

## License

MIT for the code. The datasets are CC-BY-4.0 and belong to their authors.
