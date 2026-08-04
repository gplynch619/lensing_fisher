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
`L_eff` and its 68% band. Everything else supports one of those two.

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
lensing-fisher-rebin data/full_fishers/spa_iter0.pkl -n 50 --min-width 3 -o edges_1.yaml
```

Prints `L_eff` and its 68% band, proposes an equal-information grid, and says
whether the edges have settled. Paste the fragment into the next config, point
`bins.step_size.from_fisher` at the previous pickle so each bin's step tracks its
own `sigma(q_j)`, and resubmit.

Watch the min-width warning. If most bins sit at the floor, the equal-information
target is not being met and `--min-width` is too large for the bin count — with a
kernel shaped like the published one, 50 bins over L=2..2000 wants a floor near 3,
not 8.

## Running a chain

```yaml
likelihood:
  lensing_fisher.cobaya_likelihood.CandlClipyCombined:
    dataset_file: /path/to/examples/datasets/unlensed_planck.yaml
    clear_internal_priors: false     # keep candl's priors, as the Fisher does

theory:
  lensing_fisher.template_lensing.TemplateLensingCAMB:
    clpp_template_file: /path/to/unlensed_bestfit_pp.pkl
```

Add `A_template` to `params:` for the template chain; omit both the theory
override and `A_template` for the plain unlensed chain (see
`examples/chain_unlensed_planck.yaml`).

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
  parameter at all.

## Tests

```bash
pytest
```

CAMB-dependent tests skip cleanly without it. The load-bearing ones are the
uniform-`q` ↔ `TemplateLensingCAMB` equivalence, the frozen-`C_fid` guard, and
`FisherMatrix` against an analytic Hessian at 1–4 ranks.
