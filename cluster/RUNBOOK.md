# Runbook — unlensed chains and the Fisher binning iteration

For whoever (human or agent) is driving this on `hive`. It covers the parts that
need judgment. The mechanical parts — environment, paths, submission — are in
`cluster/env.sh` and the job templates next to this file.

## Submitting

Same pattern as the rest of the project: a template plus an input file.

```bash
source cluster/env.sh
./cluster/submit_job.sh -f examples/chain_up_planck.yaml -t cluster/job_template_chain.sh
./cluster/submit_job.sh -f examples/fisher_spa.yaml      -t cluster/job_template_fisher.sh
```

`submit_job.sh` is a copy of the one in `mnu_hunter/`, unchanged: it substitutes
`{{JOB_NAME}}` (the input filename, extension stripped) and `{{INPUT_FILE}}`
(absolutised) into the template, writes the result to `submission_scripts/`, and
`sbatch`es it. Note the flag is `-f`, not `-i`. Create `logs/` first or SLURM
will drop the job.

The templates hardcode `-A leknoxgrp`, `--partition=high`,
`--qos=leknoxgrp-high-qos` and the conda env `pbh`, matching the existing
`job_template.sh`. `cluster/env.sh` holds every path.

There are two independent tracks. They do not depend on each other and should be
started together.

- **Track A — the unlensed chains.** Two Cobaya runs producing the lensing power
  *predicted* by lensing-free CMB data. Output: a best-fit `C_L^pp` template.
- **Track B — the Fisher binning iteration.** Repeated Fisher runs on the lensed
  (SPA) data, refining the `C_L^pp` bin grid until it stops moving. Output: the
  converged bin edges and `L_eff` with its 68% band.

Track B does *not* wait for Track A. It runs against a fiducial-cosmology
template, and the difference is corrected exactly at analysis time by the
Jacobian `r_j = C_tem(L_j)/C_fid(L_j)`. See "Why the tracks are independent".

---

## Before anything: three checks

Run these on a login node. Each takes seconds and each has caught a real problem.

```bash
source cluster/env.sh
python -c "import candl, clipy, cobaya, camb, mpi4py; print(candl.__version__, clipy.__version__, cobaya.__version__, camb.__version__, mpi4py.__version__)"
python -m pytest -q            # from the repo root; 67 tests, all should pass
python -m lensing_fisher.cli --help
```

**1. The three clik files are installed.** Confirm they are readable:

```bash
ls -l $PLANCK_CLIK_BASELINE/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik \
      $PLANCK_CLIK_BASELINE/low_l/commander/commander_dx12_v3_2_29.clik \
      $PLANCK_CLIK_BASELINE/low_l/simall/simall_100x143_sroll2_v3_EE_Aplanck.clik
```

Note sroll2 lives in `low_l/simall/`, alongside (or in place of) the 2018
`simall_100x143_offlike5_EE_Aplanck_B.clik` — the directory is named for the
likelihood, not the reprocessing. Getting the wrong one of those two is a silent
error: both load, both constrain tau, and they differ.

This file is load-bearing, not cosmetic: every config uses a low-ell EE
*likelihood* to constrain tau, and drops the internal tau priors that ACT and
SPT carry on exactly that basis (see check 2). Without it, tau would have no
constraint at all in the Fisher.

**A native (non-clik) sroll2 is also installed**, at
`/home/gplynch/opt/cobaya_packages/data/planck_sroll2_lowE_native`. Do not wire
it into the configs — the chain and the Fisher must go through the same clipy
likelihood or they stop being comparable, which is the whole point of the shared
dataset files. It is useful as an independent cross-check: run one short UP-P
chain against each and confirm the tau posteriors agree.

**2. Do the likelihoods build and evaluate?**

```bash
python - <<'PY'
from lensing_fisher.config import load_raw
from lensing_fisher.likelihoods import build_likelihoods
for ds in ("up_planck", "up_planck_act_spt", "spa"):
    names, likes, nuis = build_likelihoods(
        load_raw(f"examples/datasets/{ds}.yaml")["likelihoods"])
    print(ds, names, sorted(nuis))
PY
```

Report the nuisance lists. They must match the `params:` blocks of the chain
configs — UP-P should be `A_planck` only; UP-PAS adds the ACT and SPT
calibrations. A nuisance the likelihood requires but the chain does not declare
is a silent failure mode: Cobaya will use the likelihood's default and never say
so.

The log must also show, for the two sets that include ACT and SPT:

```
candl_data.ACT_DR6_TTTEEE: dropped internal prior(s) on ['tau']
spt_candl_data.SPT3G_D1_TnE_lite: dropped internal prior(s) on ['tau']
```

Both datasets ship a stand-in tau prior (ACT `0.0566 +/- 0.0058`, SPT
`0.051 +/- 0.006`) applied inside `log_like`, intended as an alternative to a
low-ell EE likelihood. Keeping them alongside sroll2 counts tau three times and
takes sigma(tau) from ~0.007 to ~0.0035; because tau is degenerate with A_s that
lands directly on the lensing amplitude. If those lines are missing, the run is
wrong — do not proceed. (The pre-2026 run had this, plus a fourth explicit tau
prior in the driver.)

**3. Does the theory reach the data?** `driver.run` calls
`windows.check_theory_lmax` and *raises* if bandpower windows extend above the
CAMB lmax. If it raises on the SPA config, do not lower the tolerance — raise
`camb.set_for_lmax.lmax`. The pre-2026 reference run fed zero theory to ACT
bandpowers above ell=2500 and nothing in its output recorded that.

---

## Track A — the two unlensed chains

Both are standard LCDM with self-consistent CAMB lensing and **no `A_template`**.
Everything sits below ell=1000, so they are cheap.

| | Config | Dataset |
|---|---|---|
| **A1** | `examples/chain_up_planck.yaml` | Planck alone, TT<=1000, TE/EE<=600 |
| **A2** | `examples/chain_up_planck_act_spt.yaml` | A1 plus ACT TT/TE/EE 600-1000 and SPT TT<=1000, TE/EE 600-1000 |

A2 is the production template. A1 exists so that the shift between the two
isolates what the added ACT and SPT data does to the prediction.

Both sets of cuts are settled — submit as written. Verified against the actual
bandpowers, the candl likelihoods retain:

```
UP-PAS   act_dr6  TT/TE/EE   8 bins each   600.5 ..  950.5
         spt3g_D1 TT        12 bins        424.7 ..  974.5
         spt3g_D1 TE/EE      8 bins each   624.4 ..  974.5
```

Two overlaps are accepted deliberately and must not be "corrected": ACT TT
600-1000 on top of Planck TT (the DR6 prescription neglects the correlation
there, Planck being tighter), and SPT against everything (separate survey, small
deep field, covariance neglected outright). The reasoning is recorded in the
dataset file.

If you do change a cut, note that dropping a spectrum from a candl `ell_cuts`
block removes it **entirely** rather than leaving it uncut, and that the Planck
cuts are clipy `crop` strings against a different indexing than candl's
`ell_cuts` on bin *effective* ells — "1000" is not precisely the same cut on both
sides.

### After each chain converges

`Rminus1_stop: 0.02` is set in the configs. Then minimize and write the template:

```bash
cobaya-run examples/chain_up_planck_act_spt.yaml --minimize
```

Take the best fit and write `{L, CL_pp_fid}` in the CAMB convention
**`[L(L+1)]^2 C_L^phiphi / 2pi`** — that is what `get_lens_potential_cls(...)[:, 0]`
returns and what `get_lensed_cls_with_spectrum` expects. Getting this wrong is
silent: the Fisher still runs, `r` is then wrong by `[L(L+1)]^2`, and `L_eff`
comes out plausible but incorrect.

```python
import camb, pickle, numpy as np
pars = camb.set_params(**best_fit)          # lens_potential_accuracy >= 4
pars.set_for_lmax(3000, lens_potential_accuracy=4)
res = camb.get_results(pars)
clpp = res.get_lens_potential_cls(lmax=3000)[:, 0]
L = np.arange(clpp.size)
pickle.dump({"L": L, "CL_pp_fid": clpp}, open("up_pas_bestfit_pp.pkl", "wb"))
```

> **Use CAMB, not the `cmb_alens_spt_*` emulators**, anywhere near these chains.
> Their tau training range is [0.012, 0.096]; extrapolating returns NaN from
> simall, which silently drops low-l EE from the gradient and walks a minimiser
> to spurious optima (tau=0.186, Alens=1.86 was the observed failure). Both
> configs already specify CAMB — do not switch them.

---

## Track B — the Fisher binning iteration

Config: `examples/fisher_spa.yaml`. Dataset: SPA at full range (Planck + ACT +
SPT), theory lmax 6500. Grid: 50 bins over L=2..2000 plus an automatic catch-all
bin to CAMB's lmax, log-spaced at iteration 0.

**The user drives this loop. Run one pass, report, and wait.** Do not chain
iterations unattended.

### Benchmark before the first full pass

The Fisher template's walltime (`2-11:00`) and rank count (15) are carried over
from a much smaller job — 27 bins at theory lmax 2500. This run is 51 bins at
lmax 6500 with `lens_potential_accuracy: 8`. Time one CAMB call at those settings
on a login node before committing two and a half days:

```bash
source cluster/env.sh
python - <<'PY'
import time, camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.37, ombh2=0.02233, omch2=0.1198, tau=0.054, mnu=0.06, omk=0)
pars.InitPower.set_params(As=2.1e-9, ns=0.9652)
pars.set_for_lmax(6500, lens_potential_accuracy=8, lens_margin=2050)
pars.set_matter_power(kmax=10, k_per_logint=130, nonlinear=True)
pars.set_accuracy(AccuracyBoost=1.0, lSampleBoost=1.0, lAccuracyBoost=1.2,
                  DoLateRadTruncation=False, min_l_logl_sampling=6000)
t = time.time(); res = camb.get_results(pars); print(f"get_results {time.time()-t:.1f}s")
PY
```

Roughly: a 62-parameter Fisher has ~1950 unique matrix elements, each needing a
few likelihood evaluations, but only ~15% perturb a cosmological parameter at all
— the rest hit the CAMB cache. So expect wall-clock of order
`1950 * 0.15 * 4 * t_camb / 15 ranks`. If that lands anywhere near the walltime,
raise ranks or split the job before submitting. Report the number.

### One pass

```bash
./cluster/submit_job.sh -f examples/fisher_spa.yaml -t cluster/job_template_fisher.sh
# -> $MNU_HUNTER_ROOT/data/full_fishers/spa_iterN.pkl

lensing-fisher-rebin $MNU_HUNTER_ROOT/data/full_fishers/spa_iterN.pkl \
    -n 50 --min-width 3 -o edges_N+1.yaml
```

Then **stop and report** to the user:

- `L_eff` and its 68% band, alongside the previous pass's values
- how far the edges moved, and whether `edges_converged` returned true
- **how many bins sit at the min-width floor** — this is the diagnostic that
  matters most. If most of them do, the grid has stopped being
  equal-information and `L_eff` is being set by the floor rather than by the
  data. At 50 bins over L=2..2000 a floor of 3 leaves the target attainable; a
  floor of 8 put 46/50 bins on the floor with 71% scatter in per-bin
  information. The CLI prints a NOTE when this happens — do not ignore it.
- any negative eigenvalues, or a non-invertible marginalized `clpp` block

Wait for the user before starting the next pass.

### Setting up the next pass

Paste the new edges into `bins.edges` and switch the step sizes to track the
previous Fisher:

```yaml
bins:
  edges: [2, 5.1, 8.7, ...]        # from edges_N+1.yaml
  step_size:
    from_fisher: ${MNU_HUNTER_ROOT}/data/full_fishers/spa_iterN.pkl
    target_sigma_frac: 0.3
    min: 0.02
    max: 0.5
output:
  filename: spa_iterN+1.pkl
```

A scalar `step_size` is fine only for iteration 0. On an equal-information grid
the narrow bins hold less lensing power and are more weakly constrained, so a
single fractional step gives them a poor finite-difference signal.

**Keep every iteration's pickle.** The sequence of `L_eff` values is itself the
evidence of convergence and belongs in the notes.

### Stopping

Stop when both hold:

- edges shift by less than the min width, and
- `L_eff` and its 68% bounds move by less than 2% of the band width.

Expect 2-4 passes. If it has not converged by 5, stop and report rather than
continuing — non-convergence means something about the kernel or the step sizes
is wrong, and more passes will not fix it.

---

## Why the tracks are independent

Two different fiducials are in play and they decouple.

**The frozen basis array `C_fid`.** The parametrization is
`C_L^phiphi(theta, q) = C_fid(L) * (1 + sum_i q_i B_i(L))` with `C_fid` a frozen
array loaded once — cosmology enters only through the unlensed spectra, never
through the lensing kernel. Choosing `C_fid != C_tem` is corrected *exactly* by
`r_j = dq_j/dA_tem = C_tem(L_j)/C_fid(L_j)` applied at analysis time. Nothing has
to be recomputed when the template changes. The only residual is the within-bin
*shape* mismatch of `C_tem/C_fid`, which matters solely for the widest high-L
bins, and even that goes away if `r_j` is computed as a `C_fid`-weighted average
across the bin rather than at the bin centre.

**The fiducial cosmology at which F is evaluated.** Under a frozen basis this
affects only the unlensed spectra and the nuisance sector, not the lensing
response — second order for a kernel that is a *relative* weighting across L.

So: run Track B now against the existing template,
`$MNU_HUNTER_ROOT/src/unlensed_bestfit_pp.pkl`, which is what `fisher_spa.yaml`
points at. It is the *pre-2026* unlensed best fit, so `r != 1` and the analysis
step applies the correction — that is expected, not a problem, and the converged
bin grid does not depend on it.

Once A2 lands, repoint `bins.template.file` at its best-fit pickle and rerun once
at the converged edges. Then `r == 1` identically and the Fisher is *exactly* the
`A_template` Fisher. `L_eff` should move by much less than its 68% width. If it
does not, that is a result worth writing down, not a bug.

---

## Things that have already gone wrong once

- **`mpirun -n >1` silently wrote all-zero Fisher matrices** — the MPI result
  path discarded the assembled array. Fixed, and `FisherMatrix.save` now refuses
  an all-zero matrix. If you ever see that exception, the fix regressed; do not
  work around it.
- **candl's `ell_max` ignores `data_selection`.** A likelihood cropped to
  ell<=1000 still advertises 8501. Trust `windows.effective_ell_max`, not
  `like.ell_max`.
- **clipy exposes no window functions**, so `effective_ell_max` falls back to its
  advertised value (2508 for plik-lite, even cropped). That is deliberately
  conservative. Set `theory_lmax` explicitly only if you have checked it.
- **The Cobaya wrapper had a wiring bug** that made both chains fail on the first
  line of `initialize()`. Fixed and covered by `tests/test_cobaya_likelihood.py`.
  If `python -m pytest -q` is green, this is not your problem.
- **candl swallows unknown constructor kwargs.** `candl.Like` takes `**kwargs`
  and ignores what it does not recognise, so `clear_internal_priors=True` passed
  to the *constructor* does nothing and says nothing. It is an option of candl's
  Cobaya interface, not of `Like`. Priors must instead be edited on the built
  object, and *before its first `log_like` call* — candl caches the compiled
  function on first evaluation and ignores later edits. `make_candl` does this at
  construction; nothing else should touch `like.priors`.
- **ACT DR6 ships bandpowers from ell=600 in all of TT/TE/EE and applies no
  default cut**, so every limit is an analysis choice the config must state. Note
  the TT overlap with Planck over 600 < ell < 1000 is *intended* — the DR6
  prescription neglects the correlation there because Planck is the tighter of
  the two. Do not "fix" it.
- **ACT's advertised `ell_max` of 8501 is not where its data is.** The last
  bandpower sits at ell = 6125.5 and the windows carry zero weight above 6500,
  which is why the CAMB lmax is 6500. `lmax_theory: 9500`, as used in the Cobaya
  configs elsewhere in this project, is safe but buys nothing and costs real time
  at `lens_potential_accuracy: 8`.
