#!/bin/bash
#SBATCH -N 1
#SBATCH -J {{JOB_NAME}}
#SBATCH -t 2-11:00
#SBATCH -A leknoxgrp
#SBATCH --partition=high
#SBATCH --mem-per-cpu=4000M
#SBATCH --ntasks=15
#SBATCH --qos=leknoxgrp-high-qos
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/{{JOB_NAME}}.%j.out
#SBATCH --error=logs/{{JOB_NAME}}.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gplynch@ucdavis.edu

# One Fisher iteration. {{INPUT_FILE}} is a lensing_fisher YAML config, e.g.
# examples/fisher_spa.yaml.
#
#   ./cluster/submit_job.sh -f examples/fisher_spa.yaml -t cluster/job_template_fisher.sh
#
# RESOURCES — two changes from the old lensing_sensitivity template:
#
#   mem-per-cpu 2000M -> 4000M. Theory lmax went 2500 -> 6500 at
#   lens_potential_accuracy 8, and every rank holds its own CAMB results object
#   plus a small cache of them (bins.camb_cache_size, default 4). After the first
#   run check what was actually used:  seff $SLURM_JOB_ID   and trim if generous.
#
#   ntasks 15 is unchanged and known to fit on a node. Fisher work is
#   embarrassingly parallel over matrix elements, so more ranks is strictly
#   faster in wall-clock — but each rank carries its own CAMB cache, so raising
#   ntasks raises memory roughly linearly. Raise it only after seff says there is
#   headroom.
#
# WALLTIME is a guess carried over from a much smaller job (27 bins at lmax
# 2500). Benchmark before trusting it — see cluster/RUNBOOK.md.

source /home/gplynch/projects/lensing_fisher/cluster/env.sh

cd $LENSING_FISHER_ROOT

echo "========= Job started at `date` =========="
echo "config    : {{INPUT_FILE}}"
echo "ranks     : $SLURM_NTASKS x $SLURM_CPUS_PER_TASK threads"
echo "mpirun    : $(mpirun --version | head -1)"
echo "python    : $(which python)"

srun python -m lensing_fisher.cli -i {{INPUT_FILE}}

echo "========= Job finished at `date` =========="
