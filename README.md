# lensing_fisher

Tools for an apples-to-apples comparison of CMB lensing power measured from the
2pt function (peak smoothing) and the 4pt function (lensing reconstruction).

The central object is an **effective bandpower window** for the 2pt measurement.
A Fisher analysis over binned perturbations to `Cl_pp` says which angular scales
the primary CMB actually responds to, giving an effective `L` with horizontal
error bars that can be plotted against reconstruction bandpowers.

## Layout

| Module | Role |
|---|---|
| `config` | YAML loading: `${VAR}` expansion and `!include file#key` |
| `likelihoods` | Build candl/clipy likelihoods from a YAML block; per-spectrum ell cuts |
| `windows` | How high in ell a likelihood really reaches, and the guards that depend on it |
| `clpp_template` | Load and interpolate a frozen `{L, CL_pp_fid}` template |
| `local_lens` | `BinnedLensingTheory` — the binned `Cl_pp` parametrization plus the CAMB results cache |
| `fisher` | `FisherMatrix` — finite differences, MPI distribution, output |
| `driver` | Config → saved Fisher matrix, as a sequence of named steps |
| `analysis` | Marginalize → weights → `L_eff` → next bin grid |
| `cobaya_likelihood` | Cobaya likelihood built from the *same* YAML the Fisher uses |
| `template_lensing` | `TemplateLensingCAMB`, the Cobaya theory for the `A_template` chain |
| `cli` | `lensing-fisher` and `lensing-fisher-rebin` |
| `mpi_log` | Rank-aware printing, so setup messages appear once and not once per rank |

Two entry points matter for the science. `driver.run(cfg, comm)` turns a config
into a saved Fisher matrix; `analysis.summarize(pickle)` turns that matrix into
the effective `L` and its 68% band. Everything else supports one of those two.

## Documentation

| | |
|---|---|
| [docs/iteration.md](docs/iteration.md) | What the binning loop is doing, what converges, when to stop |
| [docs/outputs.md](docs/outputs.md) | What is in the output pickle and the four ways to read it |
| [cluster/RUNBOOK.md](cluster/RUNBOOK.md) | Running the whole analysis on hive — jobs, pre-flight checks, and everything that has gone wrong once |

Start with `docs/outputs.md` if you have a pickle and want numbers out of it;
with `docs/iteration.md` if you want to know why there is more than one.

## Install

```bash
pip install -e .
pip install -e '.[mpi,cobaya,dev]'
```

`candl` and `clipy` come from your cosmology environment, not pip.

## The parametrization

```
C_L^phiphi(theta, q) = C_fid(L) * (1 + sum_i q_i B_i(L))
```

`C_fid` is a **frozen array**, loaded once. Cosmology enters only through the
unlensed spectra; the lensing kernel is held fixed. That is what makes this the
Fisher matrix *of the A_template model* — `TemplateLensingCAMB` lenses with
`A_template * <frozen template>` in exactly the same way, so a uniform `q_i = a`
is precisely `A_template = 1 + a`. `tests/test_template_equivalence.py` asserts
that correspondence spectrum by spectrum; if it breaks, `L_eff` is describing a
different measurement than the chain reports.

Point `bins.template.file` at the unlensed-dataset best fit and the Jacobian
`r_j = C_tem/C_fid` is identically 1. Use `from_fiducial_cosmology: true` for
iterations that run before that best fit exists, and correct with `r` afterwards.

## Running a Fisher matrix

```bash
export PLANCK_CLIK_BASELINE=/path/to/plc_3.0
export MNU_HUNTER_ROOT=/path/to/mnu_hunter
mpirun -n 32 lensing-fisher -i examples/fisher_spa.yaml
```

Datasets live in `examples/datasets/` and are pulled into both the Fisher config
and the Cobaya chain with `!include datasets/spa.yaml#likelihoods`, so the two
cannot drift apart.

### Binning iteration

```bash
lensing-fisher-rebin data/full_fishers/spa_iter0.pkl \
    -n 50 --min-width 3 --l-max 2000 -o edges_1.yaml
```

Prints the median, mean and 68% band, proposes an equal-information grid, and
says whether the edges have settled. Paste the fragment into the next config's
`bins.edges`, bump `output.filename`, and resubmit. Full walkthrough in
[docs/iteration.md](docs/iteration.md).

**`--l-max` is required in practice.** It bounds the placement to the informative
range so the catch-all bin — whose width is set by the theory lmax, not the data
— stays fixed between passes and is re-appended at run time. Without it the
placement spans the catch-all too, and since ~98% of the weight sits below
L~1000 it will collapse the resolved range and make its own boundary the largest
"edge movement" in the report.

**Leave `bins.step_size` a scalar.** The per-bin `from_fisher` form needs
`sigma(q_j)`, hence `inv(F_lens)`, which does not exist — the block is
rank-deficient by construction, and `driver.clpp_step_sizes` now raises rather
than propagating NaNs. It is also unnecessary: the `clpp` diagonals are stable to
better than 0.1% across `h = 0.01 .. 5.0`.

Watch the min-width warning. If most bins sit at the floor, the equal-information
target is not being met and `--min-width` is too large for the bin count — with a
kernel shaped like the published one, 50 bins over L=2..2000 wants a floor near 3,
not 8. Below ~3 the bin basis functions stop reaching 1 and start duplicating
their neighbours.

## Running a chain

```yaml
likelihood:
  lensing_fisher.cobaya_likelihood.CandlClipyCombined:
    dataset_file: /path/to/examples/datasets/up_planck_act_spt.yaml
    clear_internal_priors: false     # keep candl's priors, as the Fisher does

theory:
  lensing_fisher.template_lensing.TemplateLensingCAMB:
    clpp_template_file: /path/to/unlensed_bestfit_pp.pkl
```

Add `A_template` to `params:` for the template chain; omit both the theory
override and `A_template` for the plain unlensed chain.

### The three datasets

| File | What it is |
|---|---|
| `datasets/up_planck.yaml` | **UP-P** — Planck alone, TT<=1000, TE/EE<=600, plus low-l |
| `datasets/up_planck_act_spt.yaml` | **UP-PAS** — UP-P plus the lensing-free ACT TE/EE 600-1000 and SPT<=1000 |
| `datasets/spa.yaml` | **SPA** — the same experiments at full range; the *lensed* set |

Each dataset file also carries `tied_parameters` (`A_planck: A_act`, one shared
Planck/ACT calibration) and any `drop_priors` needed to stop a likelihood's
internal stand-in prior from double-counting one the set already includes — ACT
and SPT-3G both ship a `tau` prior meant as a substitute for low-ell EE.

The two unlensed sets are built on one stack and share their Planck block
verbatim, so the difference between their best-fit `C_L^pp` templates is
attributable to the added ACT and SPT data rather than to a change of likelihood
implementation. `tests/test_config.py` asserts that they differ by exactly the
two added entries. `examples/chain_up_planck.yaml` and
`examples/chain_up_planck_act_spt.yaml` are the corresponding chains; UP-PAS
produces the production template.

## Traps this package guards against

- **candl `ell_max` ignores `data_selection`.** A likelihood cropped to ell<=1000
  still advertises 8501 (ACT DR6) or 4095 (SPT-3G D1). `windows.effective_ell_max`
  finds where the retained windows actually stop; `windows.check_theory_lmax`
  refuses to start when real bandpowers would meet zero-padded theory. The
  pre-2026 reference run used theory to lmax 2500 against ACT bandpowers reaching
  ell~6126, and nothing in the output file said so.
- **clipy exposes no window functions**, so it falls back to its advertised
  `ell_max` (2508 for plik-lite, even cropped). Set `theory_lmax` explicitly if
  you have checked it yourself.
- **`FisherMatrix.save` refuses an all-zero matrix.** Before the MPI result-path
  fix, `mpirun -n >1` silently wrote zeros; only serial runs were correct.
- **Bin edges and `clpp_fid` are stored in the output pickle**, so downstream
  analysis never reconstructs bin geometry by hand.
- **Non-uniform bins need a weight *density*.** `analysis.weight_density` divides
  per-bin information by bin width; assigning the per-bin total to every L (fine
  for a uniform grid) would drag `L_eff` toward the wide high-L bins.
- **MPI work is bucketed by cosmology, not round-robin**, so the CAMB results
  cache actually hits. Roughly 85% of matrix elements perturb no cosmological
  parameter at all. The deal counter runs *across* buckets: with a per-bucket
  counter the single-element cosmology-pair buckets all landed on rank 0, which
  then carried 85 CAMB solves against ~25 elsewhere.
- **`set_cosmology` re-derives `YHe` on every call**, so the config's
  `bbn_predictor` has to be re-supplied each time. It was not, and the first
  cache miss silently reverted PArthENoPE to CAMB's PRIMAT default for the whole
  run — a 0.19% shift in `YHe`, landing in the damping tail where ACT and SPT
  carry their weight, and self-consistent enough that nothing looked wrong.
  `BinnedLensingTheory` now takes `cosmology_kwargs`.
- **A NaN log-likelihood becomes a *perfect* one.** Cobaya evaluates the
  posterior with `make_finite=True`, i.e. `np.nan_to_num`, which maps NaN to
  **0.0** — better than any real point in the space. `CandlClipyCombined.logp`
  returns `-inf` if any component is non-finite. Without it, a minimizer walks
  straight into the NaN region and a chain that proposes one can never leave.
- **The marginalized `clpp` block is rank-deficient by construction**, so
  `inv(F_lens)` is NaN and per-bin `sigma(q_j)` does not exist. That is physics,
  not numerics — see [docs/outputs.md](docs/outputs.md#do-not-invert-f_lens).

## Tests

```bash
pytest
```

CAMB-dependent tests skip cleanly without it. The load-bearing ones are the
uniform-`q` ↔ `TemplateLensingCAMB` equivalence, the frozen-`C_fid` guard, and
`FisherMatrix` against an analytic Hessian at 1–4 ranks.
