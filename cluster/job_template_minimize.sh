#!/bin/bash
#SBATCH -N 1
#SBATCH -J {{JOB_NAME}}
#SBATCH -t 0-08:00
#SBATCH -A leknoxgrp
#SBATCH --partition=high
#SBATCH --mem-per-cpu=2000M
#SBATCH --ntasks=16
#SBATCH --qos=leknoxgrp-high-qos
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/{{JOB_NAME}}.%j.out
#SBATCH --error=logs/{{JOB_NAME}}.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gplynch@ucdavis.edu

# Best fit for a converged chain, for use as the C_L^pp fiducial template.
# {{INPUT_FILE}} is the *same* Cobaya YAML the chain ran, e.g.
# examples/chain_up_planck.yaml — not a copy. Run it only after that chain has
# converged; the minimizer reads its samples.
#
#   ./cluster/submit_job.sh -f examples/chain_up_planck.yaml -t cluster/job_template_minimize.sh
#
# Output goes to <output prefix>.minimize.* and <output prefix>.minimum[.txt].
# Cobaya inserts a "minimize" infix, so nothing here overwrites the chain.
#
# WHY 16 RANKS. cobaya's minimizer runs ceil(best_of / n_ranks) starts per rank,
# so with the default best_of=2 the rank count *is* the number of independent
# minimizations. Starting points are drawn in rank order: the first four take the
# MAP of each of the four chain files, and every rank past that draws a random
# point from the `ref` distributions. So 16 ranks buys 4 local refinements plus
# 12 independent global starts, and their agreement is the evidence that the
# minimum is the right one. Raising ntasks raises that count one-for-one.
#
# This matters here specifically: a minimiser on this likelihood has gone to a
# spurious optimum before (tau=0.186, Alens=1.86, from emulator NaNs — see
# cluster/RUNBOOK.md). A single start would not have caught it.
#
# 2 threads rather than 4: the unlensed configs cut at ell <= 1000 and run CAMB
# at lmax 3000, so a likelihood call is cheap and the ranks are worth more than
# the threads. Each rank runs its start serially, so walltime is the cost of one
# minimization, not 16.

source /home/gplynch/projects/mnu_hunter/lensing_fisher/cluster/env.sh

cd $LENSING_FISHER_ROOT

echo "========= Job started at `date` =========="
echo "config    : {{INPUT_FILE}}"
echo "starts    : $SLURM_NTASKS x $SLURM_CPUS_PER_TASK threads"
echo "mpirun    : $(mpirun --version | head -1)"
echo "python    : $(which python)"

srun cobaya-run {{INPUT_FILE}} --minimize

echo "========= Job finished at `date` =========="
