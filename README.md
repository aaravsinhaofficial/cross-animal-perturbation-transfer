# Brains share the rule, but not the wiring

We trained a model on how electrical stimulation moves the brains of five mice, then
showed it a sixth mouse doing nothing special, just resting. From that resting
activity, plus the settings of the stimulus we were about to deliver, the model had to
say what would happen next in an animal it had never seen perturbed.

## What came through

**What the mouse would do.** When it would notice the stimulus and report it, and how
that changed with stimulation strength. This worked, and it worked in every animal.

**How its neural population as a whole would respond.** The timing, the rise and fall,
and the dependence on current all came across.

**But not what any individual neuron would do.**

## Why the last one fails

This is the part we found most interesting. The shared rule actually gets each neuron's
response shape right, meaning when it fires, how it ramps and decays, and how it scales
with current. The only thing it gets wrong is how strongly each neuron responds, which
is one number per cell.

Hand the model that one number per cell and change nothing else, and the neural score
jumps from 0.035 to 0.422, in every one of the six mice. A single scalar cannot invent
structure in time or across currents, because each neuron's predicted time course is
already fixed by the shared rule. So the rule was right and the amplitude was wrong.

That amplitude turned out to be essentially unguessable. We tried predicting it from
everything we could measure without stimulating, including each neuron's depth, its
distance from the electrode, its firing statistics and its spontaneous coupling to the
cells near the contact. Across all 949 neurons the correlation was 0.174, and the
improvement did not survive a test over animals.

The reason is that low current stimulation does not light up a neat sphere of nearby
cells. It grabs a sparse, scattered set that depends on where that particular electrode
happened to land in that particular brain. Pooled over every session, how much a neuron
responds is essentially unrelated to how far it sits from the contact (r = −0.013).
There is no species level rule to learn at that resolution.

## A simulated cortex, and a correction to what we expected

We built a cortex where we control the thing we think matters. When the electrode
drives neurons by a rule shared across animals, single neuron transfer works at every
population size we tried, and it improves very regularly as more neurons are recorded
(correlation of 0.99 between the score and the log of the population size). When the
electrode instead drives a scattered set private to each implant, the average score
falls from 0.68 to 0.41, the trend with population size becomes erratic, and the spread
across animals roughly triples.

So private recruitment does clearly damage single neuron transfer and makes it much less
reliable from animal to animal. It does not abolish it in simulation, which means it is
one contributing cause in the real mice rather than the whole story. Small
simultaneously recorded populations are likely another: our sessions have between 8 and
33 neurons, at the bottom of the range we simulated.

## Six animals is the sample size

The claim is about animals. Several sessions from one mouse are still one mouse, so
every headline number is inferred at the animal level, with a
bootstrap that resamples animals and an exact sign flip permutation over the six. With
six animals the smallest p value such a test can return is 0.031, reached when all six
fall on the same side, and we say so wherever it appears.

An earlier version of this work reported session level statistics as the headline,
which overstated the evidence. Correcting it changed the behavioural p value from
7×10⁻¹⁵ to 0.031 and removed any claim that population level effect sizes transfer
reliably.

## The honest comparison

Averaging the other five mice predicts the sixth about as well as our dynamical model
does. Adding the new mouse's resting activity made the behavioural prediction worse.
The behavioural consequence of stimulating this part of cortex is conserved enough
across individuals that it does not need a model of the individual, which is itself
worth knowing.

## What we could not settle

Six animals is the binding constraint. One species, one cortical area, one task, one
way of perturbing. Electrical microstimulation is coarse and does not respect cell
type, so an optogenetic perturbation aimed at a defined population may have more
conserved structure at the single cell level than we found.

## Reproducing

```bash
make env        # python 3.12 venv and dependencies
make cache      # download DANDI:001868 (7.5 GB, CC-BY-4.0), build tensors, audit
make analysis   # the results table, with animal-level statistics
make cortex     # the simulated cortex sweep
make paper      # figures, numbers and the PDF
make test       # tests
```

Data: [DANDI:001868](https://dandiarchive.org/dandiset/001868), chronic
electrophysiology in mouse somatosensory cortex during intracortical microstimulation
learning (CC-BY-4.0). Raw files are never committed; the download script rebuilds them
with a SHA-256 manifest.

## What is checked

The split-half ceiling estimator is validated against simulations with known signal and
noise across six regimes, agreeing with the attainable score to within 0.06.

Stimulation trials are delivered under a quiescence criterion, so randomly sampled
inter-trial windows carry up to three times the pre-stimulus firing rate of a real
trial. Because predictions start from unperturbed initial conditions, that would have
inflated every effect. Windows are matched to stimulation trials on a joint quantile
grid of pre-stimulus rate and wheel speed, then trimmed to match on the mean, and any
session still mismatched by more than a factor of 1.25 is dropped.

Seeds use a stable checksum rather than Python's per-process hash, so two independent
builds agree exactly and the results table reproduces bit for bit.

## Layout

```
src/cadence/
  model.py             the hierarchical model: shared dynamics, shared stimulus
                       operator, animal-specific observation maps
  linear_response.py   each animal's propagator from its own resting activity,
                       convolved with a shared drive
  synthetic_cortex.py  the simulated cortex used to test the mechanism
  metrics.py           scoring, validated split-half ceilings, animal-level statistics
  dose.py              the shared stimulus operator for low-dimensional readouts
  data/                loaders, containers, per-unit features
scripts/               download, build, analysis, probes, figures, paper
paper/                 LaTeX source with auto-generated numbers and tables
results/               JSON results and tables
```

## License

MIT for the code. The dataset is CC-BY-4.0 and belongs to its authors.
