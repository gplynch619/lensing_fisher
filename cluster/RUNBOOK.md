# Runbook — unlensed chains and the Fisher binning iteration

For whoever (human or agent) is driving this on `hive`. It covers the parts that
need judgment. The mechanical parts — environment, paths, submission — are in
`cluster/env.sh` and the job templates next to this file.

## Install

Clone *inside* the existing `mnu_hunter`, matching the local layout:

```bash
cd /home/gplynch/projects/mnu_hunter
git clone git@github.com:gplynch619/lensing_fisher.git
cd lensing_fisher
source cluster/env.sh
pip install -e . --no-deps      # candl/clipy/cobaya/camb come from the env
mkdir -p logs $MNU_HUNTER_ROOT/data/full_fishers
```

The split of responsibilities:

| | |
|---|---|
| `mnu_hunter/lensing_fisher/` | the code — its own git repo, no data |
| `mnu_hunter/src/unlensed_bestfit_pp.pkl` | the `C_fid` template the Fisher reads |
| `mnu_hunter/data/` | every output: Fisher pickles and chains |

Outputs deliberately land outside the repo, where the previous runs already live,
so multi-GB chains never enter git. `--no-deps` is deliberate too: the cosmology
stack is installed in the `pbh` env and pip must not try to replace it.

If `mnu_hunter` is itself version-controlled on hive (it is not locally), add
`lensing_fisher/` to its `.gitignore` so the nested repo is not swept into it.

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
python -m pytest -q            # from the repo root; 88 tests, all should pass
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

`Rminus1_stop: 0.02` is set in the configs. Then minimize and write the template.
Submit the minimizer against the *same* config the chain ran — not a copy, so the
two cannot drift:

```bash
./cluster/submit_job.sh -f examples/chain_up_planck_act_spt.yaml \
                        -t cluster/job_template_minimize.sh
```

Cobaya inserts a `minimize` infix, so this writes `<prefix>.minimum[.txt]`
alongside the chain and overwrites nothing. **The rank count is the number of
independent minimizations** — cobaya runs `ceil(best_of / n_ranks)` starts per
rank, so at the default `best_of: 2` the 16 ranks in the template give 16 starts:
the first four seeded from the MAP of each chain file, the remaining twelve from
random `ref` draws. That mixture is the point. A single start would not detect
the spurious optimum this likelihood has produced before.

Check three things before trusting the result:

- **all starts converged** — `grep -c 'Run 1/1 converged'` should equal the rank
  count, and there should be no `Cannot reproduce log minimum` warning
- **it beats the chain** — the minimum's `-logpost` must be below the best
  sample the chain drew
- **it is actually a minimum** — step each parameter by +/-0.1 sigma and confirm
  `-logpost` rises both ways. The asymmetry of that pair also measures how far
  off-centre the minimum still is: `d = h (f+ - f-) / (2 (f+ + f-))`.

For UP-P on 2026-08-04 that gave 16/16 converged, an improvement of 0.183 over
the best sample, every excursion positive, and a residual offset <= 0.04 sigma
(`cosmomc_theta`; all others <= 0.013 sigma). `bobyqa`'s default `rhoend: 0.05`
is a deliberately relaxed criterion for noisy likelihoods and lands right there —
do not tighten it without checking that it is not just chasing CAMB noise.

That precision is far more than this needs: a 0.05 sigma error in any single
parameter moves `C_L^pp` by at most **0.25%** anywhere in 2 <= L <= 2000 (worst
case `omch2`, at L=2000), against a 2.6% `sigma(A_template)` from the SPA Fisher.
At the offsets actually achieved the template error is ~0.03%.

One reporting trap: the `-log(Like)` and `chi-sq` in the header of `.minimum` are
the *posterior* values. The likelihood chi2 is the separate `chi2__...` entry —
939.81 against the header's 897.36 for UP-P, the 21.2-nat difference being the
flat priors' normalization. The *location* is unaffected, since flat priors put
the MAP and the maximum-likelihood point at the same place.

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

### Cost: measured, 2026-08-04

The first full pass took **13 minutes** on 15 ranks, not the 2.5 days the
template reserved, and the walltime is now `0-03:00`. After the load-balance fix
below it should be nearer 5. Nothing was skipped — the estimate in this section
used to be wrong, by about 14x.

Cost is set by the number of **unique cosmologies**, not by the element count. A
mixed derivative in one cosmological and one non-cosmological parameter reuses
the same `+/-h` CAMB solves as every other element perturbing that parameter, so
for `n_c` cosmological parameters the whole matrix needs

    4*n_c  (single-parameter stencils)
  + 4*n_c*(n_c-1)/2  (the corners of the cosmology-pair elements)
  + 1  (the fiducial)

= **85 solves at `n_c = 6`**, against roughly 16 s each at lmax 6500 /
`lens_potential_accuracy: 8` / 4 threads. The old estimate here counted
`1950 * 0.15 * 4` calls and ignored the sharing.

Those 85 used to land almost entirely on rank 0: `FisherMatrix._tasks` restarted
its round robin inside each bucket, and the 15 cosmology-pair buckets hold one
element each, so `n=0` dealt every one of them to rank 0 while the other ranks
idled at ~25 solves (42% CPU efficiency overall). The counter now carries across
buckets and all 15 ranks take 25.

To re-measure after a change, time one call at the production settings:

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

Expect wall-clock of order `85 * t_camb / n_ranks` plus about 0.3 s per matrix
element for the likelihood evaluations themselves (1891 elements at 61
parameters, spread over the ranks). Report the number.

### Accuracy settings: measured, 2026-08-04 — no change needed

`camb.accuracy` in the config is inherited from the pre-2026 run
(`AccuracyBoost 1.0`, `lSampleBoost 1.0`, `lAccuracyBoost 1.2`). A Fisher matrix
is more demanding of these than a chain is: it is a second derivative taken by
finite differences, so noise `eps` in `C_ell` enters the answer as `~eps/h^2`,
an amplification of order `1e4` at `h ~ 0.01`. Absolute accuracy largely cancels
between the `+h` and `-h` evaluations; *smoothness in the parameters* does not.

That reasoning is sound but the answer came back clean, and directly: every
`clpp` diagonal of the real iteration-0 Fisher is stable to better than **0.1%
across `h = 0.01 .. 5.0`**, a 500x range, and the off-diagonals to 0.1-2%. The
finite differences sit on a wide plateau at the current settings. `step_size:
0.05` is comfortably inside it for every bin, weak ones included.

So do not raise the boosts, and do not read the singular matrix as a symptom of
them — see "The matrix is rank-deficient, and that is fine" below. Verified at
the same time: `set_for_lmax(lens_potential_accuracy=8)` does survive the later
`set_matter_power` and `set_accuracy` calls (`max_eta_k` = 144000).

The check below is kept for when the settings or the parameter set change. It
measures the stability of a second difference at current and boosted settings:

```bash
python - <<'PY'
import numpy as np, camb, time

def d2_logl_proxy(boost, h=0.01, H0=67.37):
    """Second difference of the TT spectrum in H0, at one accuracy setting."""
    out = []
    for f in (1 - 2*h, 1 - h, 1.0, 1 + h, 1 + 2*h):
        p = camb.CAMBparams()
        p.set_cosmology(H0=H0*f, ombh2=0.02233, omch2=0.1198, tau=0.054, mnu=0.06,
                        bbn_predictor='PArthENoPE_880.2_standard.dat',
                        nnu=3.044, num_massive_neutrinos=1)
        p.InitPower.set_params(As=1e-10*np.exp(3.043), ns=0.9652)
        p.set_for_lmax(6500, lens_potential_accuracy=8, lens_margin=2050)
        p.set_matter_power(kmax=10, k_per_logint=130, nonlinear=True)
        p.set_accuracy(AccuracyBoost=boost[0], lSampleBoost=boost[1],
                       lAccuracyBoost=boost[2], DoLateRadTruncation=False,
                       min_l_logl_sampling=6000)
        out.append(camb.get_results(p).get_total_cls(6500, CMB_unit='muK')[:, 0])
    a, b, c, d, e = out
    # the 4th-order stencil the Fisher actually uses
    return (-a + 16*b - 30*c + 16*d - e) / (12 * (H0*h)**2)

t = time.time(); base = d2_logl_proxy((1.0, 1.0, 1.2)); t_base = time.time()-t
t = time.time(); high = d2_logl_proxy((2.0, 2.0, 2.0)); t_high = time.time()-t
ell = np.arange(base.size)
m = (ell >= 600) & (ell <= 6300)
rel = np.abs(high[m] - base[m]) / np.maximum(np.abs(high[m]), 1e-30)
print(f"time  base {t_base:6.1f}s   boosted {t_high:6.1f}s   ratio {t_high/t_base:.1f}x")
print(f"d2C_TT/dH0^2 relative change, 600<ell<6300: "
      f"median {np.median(rel):.2e}  90th {np.percentile(rel,90):.2e}")
PY
```

Interpretation: if the second difference is stable to well under a percent, the
current settings are fine and the boosts only cost time. If it moves at the
percent level or worse, raise `lSampleBoost` first — it is the direct lever,
setting how densely `ell` is sampled before spline interpolation, which is where
parameter-dependent wiggle enters. Report both numbers and the cost ratio before
changing anything; at lmax 6500 the boosted settings can be several times slower,
which interacts with the walltime above.

**lmax stays 6500, not 9000.** ACT's window support ends at 6308 with zero weight
above 6500 (measured, see check 3), and `lens_margin: 2050` already computes the
unlensed spectra to 8550 for the lensing convolution. 9000 roughly doubles the
cost of every CAMB call for bandpowers that do not exist.

**`recombination_model: CosmoRec` is not set, deliberately.** It is a real physics
change — a few times 0.1% in the damping tail, where ACT and SPT live — but the
Cobaya chain configs do not set it either, and adopting it in one and not the
other is worse than adopting it in neither. It also requires CosmoRec to be built
and visible to CAMB. If you want it, it has to go into both, and both should then
be rerun. Its effect on a *curvature* is second order in any case.

### One pass

```bash
./cluster/submit_job.sh -f examples/fisher_spa.yaml -t cluster/job_template_fisher.sh
# -> $MNU_HUNTER_ROOT/data/full_fishers/spa_iterN.pkl

lensing-fisher-rebin $MNU_HUNTER_ROOT/data/full_fishers/spa_iterN.pkl \
    -n 50 --min-width 3 -o edges_N+1.yaml
```

Then **stop and report** to the user:

- `L_eff` (mean), the **median**, and the 68% band, alongside the previous pass's
  values. Prefer the median: see "L_eff is the fragile statistic" below. The CLI
  prints a NOTE when mean and median differ by more than a quarter of the 68%
  width, which is the signal that the weight is skewed enough for the mean to be
  misleading.
- **the moment window.** The moments now stop below the last bin, so the window
  is that bin's lower edge — and that edge *moves between iterations*. Two passes
  are only comparable at a common cap; compare at the smaller of the two.
- how far the edges moved, and whether `edges_converged` returned true
- **how many bins sit at the min-width floor** — this is the diagnostic that
  matters most. If most of them do, the grid has stopped being
  equal-information and `L_eff` is being set by the floor rather than by the
  data. At 50 bins over L=2..2000 a floor of 3 leaves the target attainable; a
  floor of 8 put 46/50 bins on the floor with 71% scatter in per-bin
  information. The CLI prints a NOTE when this happens — do not ignore it.

  The floor is not cosmetic. With `steepness: 2` a bin basis function is a
  difference of two sigmoids that each take about +/-1 in L to switch, so a bin
  narrower than ~3 never reaches 1 and is mostly a smeared copy of its
  neighbours. Iteration 0's log-spaced grid put 18 bins below L=24, several of
  them containing *no integer multipole at all*: bin 2 spanned [2.296, 2.637)
  and its largest effect, 0.16, was at L=2 — inside bin 1. `--min-width 3` is
  what keeps the parametrization meaning what it says.

Wait for the user before starting the next pass.

### The matrix is rank-deficient, and that is fine

Iteration 0 reported 23 non-positive eigenvalues and NaN per-bin sigmas. That is
the expected output, not a fault, and it is *not* the small scales dropping out:
bins 43-51 (L > 660) carry 0.2% of the non-positive subspace and are the best
determined in the matrix. The bad directions are alternating-sign combinations of
*adjacent* bins around L = 30-330.

The lensed CMB responds to `C_L^pp` through a broad smoothing, so it resolves
only a handful of combinations — the marginalized `clpp` block's eigenvalues fall
below 1e-3 of maximum at mode 7 and below 1e-6 at mode 14, and only ~4 modes have
sigma < 1. The other ~44 directions have a true eigenvalue of zero, which finite
differences scatter to either side; 23 landed negative, the worst at 1.5e-12 of
the largest eigenvalue.

The distinction that matters:

| question | needs | status |
|---|---|---|
| "how well is bin *j* measured on its own?" | `inv(F)` | undefined, permanently |
| "where does the information on `A_template` live?" | `F @ r` | fine |

`w_j = r_j (F r)_j` never touches the null space, so `L_eff`, its band and
`sigma(A_template)` are unaffected — iteration 0 gave `L_eff` = 187.1, band
[60.3, 240.9], `sigma(A_tem)` = 0.0263, with all 51 `w_j` non-negative. Since the
second row is what this track is for, a fine grid is fine. Reducing the bin count
would only be necessary if you wanted per-bin `C_L^pp` constraints, which would
need n <~ 7 and a rank-aware analysis.

`summary()` now reports the resolved-mode count and prints `--` rather than NaN.

### Setting up the next pass

Paste the new edges into `bins.edges` and bump the output filename. That is all:

```yaml
bins:
  edges: [2, 5.1, 8.7, ...]        # from edges_N+1.yaml
  step_size: 0.05                  # leave scalar — see below
output:
  filename: spa_iterN+1.pkl
```

**Do not switch to `step_size: {from_fisher: ...}`.** This runbook used to
instruct that, on the reasoning that narrow bins are weakly constrained and need
proportionally larger steps. Both halves turned out to be wrong:

- It cannot work. Sizing a step off `sigma(q_j)` requires inverting the
  marginalized `clpp` block, which is rank-deficient by construction. On the
  iteration-0 grid 21 of 51 sigmas came back NaN, and the code logged
  `nan..nan` and carried on — the NaNs would have reached the next Fisher
  silently. `clpp_step_sizes` now raises instead.
- It is unnecessary. The finite differences are on a 500x-wide plateau (see
  "Accuracy settings" above), so a single scalar serves every bin.

**Keep every iteration's pickle.** The sequence of `L_eff` values is itself the
evidence of convergence and belongs in the notes.

### L_eff is the fragile statistic

`weight_density` spreads each bin's information uniformly across the bin. That is
harmless for a bin a few multipoles wide and badly wrong for the last one, which
runs to CAMB's `max_l` — its width is set by the theory lmax, not by the data,
and the real information inside it falls steeply toward the low edge. A mean
takes the full lever arm of that width.

Measured on the SPA runs. Iteration 0's last bin was [2000, 8550] holding 0.02%
of the information, so it barely mattered; iteration 1's was [999, 8550] holding
2.1%, and including it moved `L_eff` from 164.2 to **261.2** — past its own 68%
upper bound of 242.8. A mean outside its own central interval is the tell. The
median moved 0.5 and the band moved 1.6 across the same change.

So `summarize` excludes the last bin from the moments (`exclude_catchall=True`,
the default). It still contributes to `w_bins` and `sigma_A_template`, which are
unaffected — those are sums, with no lever arm.

### Stopping

Stop when both hold:

- edges shift by less than the min width, **among the edges that carry
  information**, and
- the median and the 68% bounds move by less than 2% of the band width, compared
  **at a common moment cap**.

Both qualifications are load-bearing, and iteration 1 is why. Its raw verdict was
"max edge movement 375.93, not converged" — but 44 of 51 edges moved less than
the min width, every edge below L=300 moved by <=16, and the 375.93 was the
catch-all's lower edge going 999.3 -> 1375.0. That edge sits where ~2% of the
information is, so the equal-information CDF is nearly flat there and the
quantile is barely determined. It is not a data-determined number and waiting for
it to settle is waiting on noise.

Likewise the moments: compared at their own windows, iteration 0 -> 1 moved the
mean by 22 and `L_plus` by 11. Compared at a common cap of 999 — the range both
grids actually resolve with real bins — the shifts are mean +2.0, median +0.4,
`L_minus` +0.2, `L_plus` +0.7, against a 2% tolerance of 3.4. Converged.

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
- **`set_cosmology` silently reverted the BBN predictor.** CAMB rederives `YHe`
  from `bbn_predictor` on *every* `set_cosmology` call, and
  `BinnedLensingTheory._camb_results` called it without the config's
  `camb.set_cosmology` block. So the first cache miss dropped PArthENoPE for
  CAMB 1.6.4's PRIMAT default and stayed there — `YHe` 0.24537943 -> 0.24583841,
  0.19%, landing in the damping tail where ACT and SPT carry their weight. The
  whole iteration-0 run was affected. It was self-consistent internally, so
  nothing looked wrong; it just described a different model from the chains,
  which is the one failure this package exists to prevent. Fixed by passing
  `cosmology_kwargs` through, covered by
  `tests/test_parametrization.py::test_bbn_predictor_survives_every_cache_miss`.
  The same call also hardcoded `mnu`/`omk` defaults, so a parameter promoted to
  the free vector would have been silently pinned; also fixed.
- **A NaN likelihood became a *perfect* likelihood.** Cobaya evaluates the
  posterior with `make_finite=True`, which is `np.nan_to_num` — and that maps NaN
  to **0.0**, not to a bad value. A log-likelihood of zero beats every real point
  in the space, so a minimizer runs straight at it and a chain that proposes one
  can never leave. This sank the first UP-PAS minimization (2026-08-05): two of
  sixteen starts reached the corner of the prior box and reported
  `-logpost = -25.3613`, which is *exactly* minus the log-prior volume — the
  signature of a likelihood contributing precisely zero. Cobaya's own "Cannot
  reproduce log minimum" check caught it and failed the job, which is the only
  reason it was not silently adopted. `CandlClipyCombined.logp` now returns
  `-inf` if any component is non-finite; `make_finite` maps that to -1.8e308.
  The culprit is **`planck_lowl_ee` (simall sroll2) at tau ~ 0.14**, with CAMB's
  spectra entirely finite — so this is *not* the emulator tau-range problem noted
  below, and using CAMB does not protect you from it.
- **clipy's `commander` self-check fails**, `got -166.796 expected -11.6257`, in
  every job that loads it. Investigated: at the fiducial cosmology commander
  returns -166.839 and responds sensibly to `logA` (dlogL = -0.82, -2.33, +1.41
  for dlogA = +0.02, +0.05, -0.05), so the shape and derivatives are sound and
  this looks like a missing ~155-nat normalization constant. Harmless for the
  Fisher (cancels in a second difference) and for MCMC (a constant offset), but
  it invalidates any absolute chi2 or evidence. Confirm against cobaya's native
  clik before quoting either.
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
