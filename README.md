# Brains share the rule, but not the wiring

Show a model several animals being perturbed. Then show it a new animal doing nothing
in particular. From that ordinary activity, plus the settings of a perturbation about
to be delivered, the model has to say what will happen next in an animal it has never
seen perturbed.

We ran that test on two cohorts of mice and got two different answers. The difference
between them is the result.

## Two ways of pushing on a cortex

**Light in frontal cortex.** 39 mice, 193 sessions, 3,004 neurons, anterior lateral
motor cortex silenced with light during the delay period of a decision task
([DANDI:000009](https://dandiarchive.org/dandiset/000009),
[000010](https://dandiarchive.org/dandiset/000010),
[000011](https://dandiarchive.org/dandiset/000011)). The last two describe the light as
a continuous laser trace with onset events rather than per-trial columns, so
`src/cadence/data/alm_wide.py` reconstructs the dose, the site and the epoch from
that.

**Current in somatosensory cortex.** 6 mice, 48 sessions, a chronically implanted probe
delivering microstimulation the animal has learned to report
([DANDI:001868](https://dandiarchive.org/dandiset/001868)).

Both perturbations are described by numbers the experimenter chose, so an unseen
intervention has a meaning the model can be handed.

## What the shared rule is

The operator transfers, so something about a neuron's ordinary activity must say how a
perturbation will move it, and the same something must hold in every animal. We asked
what it is without fitting anything: inside each recording, correlate the individual
part of each neuron's response with properties read off its control trials.

| property of the neuron | correlation | animals negative | p |
|---|---|---|---|
| how choice selective it is | −0.171 | 32/38 | 2.4e-5 |
| how fast it fires | −0.221 | 28/38 | 0.005 |
| how much it ramps | −0.003 | 19/38 | 1.00 |
| how fast it fires, **under current** | −0.015 | 4/6 | 0.69 |

The cohort is three separate releases collected years apart, and the selectivity result
replicates independently in each half of it: 17 of the first release's 19 animals
(p = 7e-4) and 15 of the later releases' 19 (p = 0.019).

**The light takes most from the cells that have most**, by the same rule in nearly every
animal. The ramp is a negative control inside the same analysis. Under current no such
rule exists, which is exactly why nothing individual transfers there.

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
animals, positive in 17 of them, and on the four animals it does best in it reaches
+0.36 to +0.54 against a stereotype managing +0.06 to +0.31. No single
animal carries it: dropping each in turn, the weakest version of the test is p = 7.1e-4.

It replicates on the full 39-animal cohort, which adds two releases whose recordings are
much smaller: +0.086 against +0.020, better in 26 of 39 animals, p = 0.013. The average
does not rise when animals with a dozen neurons are added, for the reason the
measurement-quality result below makes explicit.

**On the individual part, it still wins, and how much depends on the recording.** A
shared operator acting on each neuron's own ordinary activity lands above zero in 15 of
20 mice (exact sign test, p = 0.041), against a measurable maximum of 0.57. Split the
animals in half by that maximum, which is fixed by trial counts and firing rates before
any model is fitted, and in the better measured half the operator recovers **22% of
everything individual that is measurable, in 9 of 10 animals** (p = 0.021). In the worse
measured half it recovers nothing.

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
curve. One animal is not merely useless but far worse than predicting nothing, because a
single animal's idiosyncrasies get mistaken for a rule and then applied confidently to
somebody else. Five animals are still deeply negative. The average across animals only
reaches zero at 38, tracking the log of the cohort size at r = +0.77 and still rising.

That average is a mean of ratios, and one animal with almost no measurable effect holds
it down, so the number to read is how many animals the operator helps in: at 38 it is
above zero in 29 of 39 mice (p = 0.0034), and at five animals in fewer than half.

A study with five animals would not have found this and would have reported that
nothing transfers.

## What behaviour does and does not tell you

The behavioural consequence of the light transfers well: the average of the other mice
predicts a held-out mouse's change in choice with a median ΔR² of 0.51. But adding what
the model knows about that mouse's own neurons does not improve it at all.

That is the thesis from the other side. Behaviour is a population-level readout, and
population-level effects are stereotyped enough across animals that the individual adds
nothing. So predicting an animal's change of mind is no evidence that a model has
captured that animal.

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
