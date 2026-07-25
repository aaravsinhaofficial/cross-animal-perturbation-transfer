#!/usr/bin/env bash
# The CPU-bound analyses, run one at a time and with a thread budget, so they do not
# starve the leave-one-animal-out training jobs on the GPUs.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12

run () { echo "=== $* ==="; .venv/bin/python -u "$@" 2>&1 | grep -viE "^\s*(warnings|  warn)"; }

run scripts/analyse_individuality.py --cache data/proc/icms.pkl --tag icms \
    --preds results/preds_icms3_test.npz
run scripts/run_cohort_scaling.py --cache data/proc/alm.pkl --tag alm
run scripts/run_decomposition_sweep.py --n-animals 8 --n-obs 32 --het 0.0 0.25 0.5
run scripts/run_cohort_scaling.py --cache data/proc/icms.pkl --tag icms --draws 8
