# Reproduce everything, in order. `make all` from a clean checkout.
PY := .venv/bin/python

.PHONY: all env data cache ladder teacher probes figures paper test lint clean

all: cache ladder teacher figures paper

env:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=.venv uv pip install --python $(PY) -e . numpy scipy matplotlib \
	  scikit-learn pandas h5py pyarrow pynwb pyyaml seaborn statsmodels tqdm pytest ruff
	VIRTUAL_ENV=.venv uv pip install --python $(PY) torch \
	  --index-url https://download.pytorch.org/whl/cu128

# 7.5 GB of public NWB (CC-BY-4.0) plus a SHA-256 manifest
data:
	$(PY) scripts/download_dandi.py --out data/raw/dandi001868

# analysis tensors + the leakage / covariate-matching audit
cache: data
	$(PY) scripts/build_icms_cache.py

# the results table (Table 2 of the paper)
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

paper: figures
	$(PY) scripts/make_paper_numbers.py
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
