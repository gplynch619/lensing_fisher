#!/bin/bash
#SBATCH -N 1
#SBATCH -J {{JOB_NAME}}
#SBATCH -t 6-00:00
#SBATCH -A leknoxgrp
#SBATCH --partition=high
#SBATCH --mem-per-cpu=2000M
#SBATCH --ntasks=4
#SBATCH --qos=leknoxgrp-high-qos
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/{{JOB_NAME}}.%j.out
#SBATCH --error=logs/{{JOB_NAME}}.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gplynch@ucdavis.edu

# One Cobaya chain. {{INPUT_FILE}} is a Cobaya YAML, e.g.
# examples/chain_up_planck.yaml or examples/chain_up_planck_act_spt.yaml.
#
#   ./cluster/submit_job.sh -f examples/chain_up_planck.yaml -t cluster/job_template_chain.sh
#
# Resources match the old MCMC template: 4 ranks = 4 chains, which is what
# Rminus1_stop assumes. Both unlensed sets are cut at ell <= 1000 and CAMB runs
# at lmax 3000, so this is far cheaper than the Fisher job.
#
# -r resumes. Cobaya writes to the `output:` path in the config, so resubmitting
# the same file continues rather than restarts; delete the output prefix to start
# clean.

source /home/gplynch/projects/lensing_fisher/cluster/env.sh

cd $LENSING_FISHER_ROOT

echo "========= Job started at `date` =========="
echo "config    : {{INPUT_FILE}}"
echo "chains    : $SLURM_NTASKS x $SLURM_CPUS_PER_TASK threads"
echo "mpirun    : $(mpirun --version | head -1)"
echo "python    : $(which python)"

srun cobaya-run {{INPUT_FILE}} -r

echo "========= Job finished at `date` =========="
