#!/bin/bash
# Environment for lensing_fisher on hive. Sourced by the job templates, and
# meant to be sourced by hand on a login node before running anything
# interactively:
#
#   source cluster/env.sh
#
# Everything in this file is a hive-specific fact. If a path is wrong, fix it
# here rather than in a config — the YAML configs reference these variables and
# lensing_fisher.config raises on an unset one rather than silently substituting
# an empty string.

source /home/gplynch/.bash_profile

module load openmpi
conda activate pbh

# lensing_fisher is cloned *inside* mnu_hunter, matching the local layout. The
# code is its own git repo; the data it reads and every result it writes live in
# mnu_hunter/{src,data}, which is not version controlled.
export MNU_HUNTER_ROOT=/home/gplynch/projects/mnu_hunter
export LENSING_FISHER_ROOT=$MNU_HUNTER_ROOT/lensing_fisher

# Planck 2018 plc_3.0 baseline, as used by src/lensing_sensitivity_fisher.py.
# No trailing slash: the configs append "/low_l/...", "/hi_l/...".
export PLANCK_CLIK_BASELINE=/home/gplynch/opt/cobaya_packages/data/planck_2018/baseline/plc_3.0

# One MPI rank per task, threads within a task. OPENBLAS_NUM_THREADS=1 stops
# BLAS from oversubscribing the cores OpenMP is already using.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
