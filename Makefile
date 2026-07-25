# Reproduce everything, in order. `make all` from a clean checkout.
PY := .venv/bin/python

.PHONY: all env data cache analysis operator individual cohort cortex ladder teacher probes figures paper test lint clean

all: cache analysis operator individual cohort cortex figures paper

env:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=.venv uv pip install --python $(PY) -e . numpy scipy matplotlib \
	  scikit-learn pandas h5py pyarrow pynwb pyyaml seaborn statsmodels tqdm pytest ruff
	VIRTUAL_ENV=.venv uv pip install --python $(PY) torch \
	  --index-url https://download.pytorch.org/whl/cu128

# public NWB (CC-BY-4.0) plus a SHA-256 manifest: 7.5 GB of microstimulation and
# 13 GB of optogenetic silencing
data:
	$(PY) scripts/download_dandi.py --out data/raw/dandi001868
	$(PY) scripts/download_dandiset.py --dandiset 000009 --out data/raw/dandi000009
	$(PY) scripts/download_dandiset.py --dandiset 000010 --out data/raw/dandi000010
	$(PY) scripts/download_dandiset.py --dandiset 000011 --out data/raw/dandi000011

# analysis tensors + the leakage / covariate-matching audit
cache: data
	$(PY) scripts/build_icms_cache.py
	$(PY) scripts/build_alm_cache.py
	$(PY) scripts/build_alm_wide_cache.py

# the results table, with animal-level statistics
analysis:
	$(PY) scripts/run_final_analysis.py
	$(PY) scripts/probe_readout_oracle.py
	$(PY) scripts/probe_unit_gain.py

# leave-one-animal-out training of the shared operator, both cohorts
operator:
	$(PY) scripts/train_operator2.py --cache data/proc/alm.pkl  --tag alm5  --seeds 0 1 2 3 4
	$(PY) scripts/train_operator2.py --cache data/proc/icms.pkl --tag icms5 --seeds 0 1 2 3 4
	$(PY) scripts/train_operator2.py --cache data/proc/alm.pkl data/proc/alm_wide.pkl \
	  --tag almall --seeds 0 1 2
	$(PY) scripts/compare_cohort_size.py

# the split into what the population shares and what belongs to one neuron
# every cohort through the same procedure, one ridge fixed in advance, plus the
# per-animal-selection variant on the full cohort for the comparison in the text
individual:
	$(PY) scripts/analyse_individuality.py --cache data/proc/alm.pkl --tag alm_fixed \
	  --ridge 1.0 --preds results/preds_alm5.npz
	$(PY) scripts/analyse_individuality.py --cache data/proc/icms.pkl --tag icms_fixed \
	  --ridge 1.0 --preds results/preds_icms5.npz
	$(PY) scripts/analyse_individuality.py --cache data/proc/alm.pkl data/proc/alm_wide.pkl \
	  --tag almall_fixed --ridge 1.0
	$(PY) scripts/analyse_individuality.py --cache data/proc/alm.pkl data/proc/alm_wide.pkl \
	  --tag almall
	$(PY) scripts/run_alignment_baseline.py
	$(PY) scripts/analyse_rule.py
	$(PY) scripts/analyse_rule.py --cache data/proc/icms.pkl --tag icms
	$(PY) scripts/run_behaviour_transfer.py --cache data/proc/alm.pkl \
	  --preds results/preds_alm5.npz --tag alm

# how the operator improves as animals are added
cohort:
	$(PY) scripts/run_cohort_scaling.py --cache data/proc/alm.pkl  --tag alm
	$(PY) scripts/run_cohort_scaling.py --cache data/proc/alm.pkl data/proc/alm_wide.pkl \
	  --tag almall --draws 8
	$(PY) scripts/run_cohort_scaling.py --cache data/proc/icms.pkl --tag icms --draws 8

# the simulated cortex: does private recruitment explain the failure?
cortex:
	$(PY) scripts/run_cortex_sweep.py
	$(PY) scripts/run_decomposition_sweep.py --n-animals 8

# the earlier session-level table, kept for reference
ladder:
	$(PY) scripts/run_icms_ladder.py

# teacher-RNN benchmark with known ground truth (Table 1)
teacher:
	$(PY) scripts/run_teacher.py --regime shared        --n-animals 5
	$(PY) scripts/run_teacher.py --regime heterogeneous --n-animals 5
	$(PY) scripts/run_teacher.py --regime degenerate    --n-animals 5

# diagnostics that localise where transfer breaks down
probes:
	$(PY) scripts/probe_transfer_levels.py
	$(PY) scripts/probe_granularity.py
	$(PY) scripts/probe_behavior_transfer.py
	$(PY) scripts/probe_physical_transfer.py

figures:
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_figures2.py

paper: figures
	$(PY) scripts/make_paper_numbers2.py
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null
	cp paper/main.pdf docs/paper.pdf
	cp paper/figures/*.png docs/

test:
	$(PY) -m pytest tests/ -q

lint:
	.venv/bin/ruff check src scripts tests

clean:
	rm -f paper/*.aux paper/*.log paper/*.out
