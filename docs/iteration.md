# The binning iteration

What the loop is for, what converges, and how to tell when it has. For the
mechanics of submitting the jobs on hive, see `cluster/RUNBOOK.md`; for what to
do with the resulting pickle, see [outputs.md](outputs.md).

## The model

The lensing potential is a frozen template times bin-wise fractional
perturbations:

```
C_L^phiphi(theta, q)  =  C_fid(L) * (1 + sum_i q_i B_i(L))
```

- `C_fid(L)` is a **frozen array**, loaded once from a `{L, CL_pp_fid}` pickle.
  Cosmology never touches it.
- `B_i(L)` is a smooth top-hat over bin `i` — a difference of two sigmoids, with
  the outermost two one-sided so the basis sums to exactly 1 everywhere.
- `q_i = 0` is the fiducial, and `q_i` is a *fractional* perturbation.

Cosmology enters only through the unlensed spectra. That is the whole design:
because the lensing kernel is held fixed, a uniform `q_i = a` is *exactly*
`A_template = 1 + a`, so this is the Fisher matrix of the model the template
chain samples. `tests/test_template_equivalence.py` asserts that spectrum by
spectrum.

## Why iterate

We want an **equal-information grid** — every bin carrying the same share of the
constraint, so none is wasted and none dominates. But you cannot place such a
grid without already knowing where the information is, and that is exactly what
the Fisher matrix tells you. So it bootstraps:

```
     bin grid  ──>  Fisher matrix  ──>  w(L), information density
        ^                                        │
        └─────  edges at equal-information quantiles  ─────┘
```

**Iteration 0** is a deliberately uninformed starting guess (log-spaced).
**Iterations 1+** are equal-information grids. The loop has converged when
feeding a grid in reproduces it on the way out.

Note what is *not* iterating: the physics. `sigma(A_template)` and the 68% band
are stable from iteration 0 onward. What iterates is the *coordinate system* used
to describe where the information sits.

## The catch-all bin

The last bin always runs to CAMB's `max_l` (8550 = `lmax` 6500 + `lens_margin`
2050), appended automatically by `driver.add_catchall_bin`. It exists so that
every multipole belongs to some bin, which is what makes a uniform `q` exactly
`A_template`.

**It is not a data-driven bin.** Its width is set by the theory lmax, not by the
data, and it typically holds ~0.02% of the information spread over thousands of
multipoles. Two consequences, both learned the hard way:

- **Hold it fixed across iterations.** Pass `--l-max 2000` to
  `lensing-fisher-rebin` so the placement covers only the informative range and
  the catch-all is re-appended at run time. Without it the placement spans the
  input grid's *full* range: with ~98% of the weight below L~1000, all 50 bins
  landed under 999 and the 50th became `[999, 8550]`. The resolved range
  collapsed from 2000 to 999, a bin was spent on the tail, and the catch-all's
  lower edge became a free parameter — which then dominated both the "edge
  movement" verdict and `L_eff`.
- **Keep it out of the moments.** `analysis.summarize(exclude_catchall=True)`,
  the default. See [outputs.md](outputs.md#l_eff-is-the-fragile-one).

## One pass

```bash
./cluster/submit_job.sh -f examples/fisher_spa.yaml -t cluster/job_template_fisher.sh
lensing-fisher-rebin $MNU_HUNTER_ROOT/data/full_fishers/spa_iterN.pkl \
    -n 50 --min-width 3 --l-max 2000 -o edges_N+1.yaml
```

Paste the fragment into `bins.edges`, bump `output.filename`, resubmit. **Leave
`bins.step_size` a scalar** — do not switch to the per-bin `from_fisher` form;
it needs `inv(F_lens)`, which does not exist (see below), and the finite
differences are on a 500x-wide plateau anyway, so a scalar serves every bin.

`--min-width 3` is not cosmetic. With `steepness: 2` a bin's basis function is a
difference of sigmoids that each take about +/-1 in L to switch, so a bin
narrower than ~3 never reaches 1 and is mostly a smeared copy of its neighbours.
Iteration 0's log grid put 18 bins below L=24, several containing *no integer
multipole at all*; bin 2 spanned [2.296, 2.637) and its largest effect, 0.16, was
at L=2 — inside bin 1. At width 3 a basis function peaks at 0.995.

## Stopping

Stop when both hold:

- edges shift by less than the min width, **among the edges that carry
  information**, and
- the median and 68% bounds move by less than 2% of the band width, compared **at
  a common moment cap**.

Both qualifications matter. An edge sitting where the information is thin has a
nearly flat equal-information CDF under it, so its quantile is barely determined
and waiting for it to settle is waiting on noise. And because the moment window
is the catch-all's lower edge, two passes are only comparable at the smaller of
their two caps.

Expect 2-4 passes. If it has not settled by 5, stop and report — something about
the kernel or the step sizes is wrong and more passes will not fix it.

## Why the matrix is singular, and why that is fine

The marginalized `clpp` block is rank-deficient by construction. The lensed CMB
responds to `C_L^pp` through a broad smoothing, so it resolves only a handful of
combinations: on a 51-bin grid the eigenvalues fall below 1e-3 of maximum at mode
7 and below 1e-6 at mode 14, and only ~4 modes have `sigma < 1`. The remaining
~44 directions have a true eigenvalue of zero, which finite differences scatter
to either side of it — so a rank-deficient matrix with some small negative
eigenvalues is the **expected** output, not a symptom of bad step sizes.

The distinction that matters:

| question | needs | status |
|---|---|---|
| "how well is bin *j* measured on its own?" | `inv(F)` | undefined, permanently |
| "where does the information on `A_template` live?" | `F @ r` | fine |

Everything the analysis reports is the second row, so none of it is affected.
Reducing the bin count would only help the first row, and would need `n <~ 7`.

## History

| pass | grid | median | 68% band | sigma(A_tem) |
|---|---|---|---|---|
| 0 | log, 50 bins over L=2..2000 + catch-all | 126.0 | [60.3, 241.1] | 0.0263 |
| 1 | equal-info, 50 bins over L=2..2000 + catch-all | 126.6 | [60.6, 242.8] | 0.0263 |

Iteration 1 -> 2 proposes a maximum edge shift of 42.4, with 44 of 51 edges
moving less than 3 and every mover above L=259.

`spa_iter0_primat_yhe.pkl` and `spa_iter1_fullrange_rebin.pkl` are kept as the
pre-fix counterparts of the two passes (wrong `YHe`, and the full-range rebin
respectively). Neither should be used for results.
