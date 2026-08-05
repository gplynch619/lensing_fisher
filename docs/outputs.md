# Reading a Fisher pickle

What `lensing-fisher` writes, and the four ways to use it. For why the iteration
exists at all, see [iteration.md](iteration.md).

The pickle is **self-contained** — it embeds the bin geometry, the frozen
`C_fid`, and every accuracy setting, so downstream analysis never reconstructs
anything by hand. It needs only **numpy and scipy** to use: no CAMB, candl, clipy
or cobaya. That is deliberate, so the analysis runs on a laptop.

```bash
rsync -av hive:$MNU_HUNTER_ROOT/data/full_fishers/spa_iter1.pkl .   # ~85 KB
```

## Contents

| key | shape | what |
|---|---|---|
| `fisher_matrix` | (61, 61) | the matrix, **not** marginalized |
| `param_names` | 61 | order: 6 cosmology, 4 nuisance, then `clpp_1..clpp_51` |
| `bin_edges` | (52,) | the grid; the last entry is CAMB's `max_l` |
| `clpp_fid` | (8551,) | the frozen `C_fid` actually used, on CAMB's ell grid |
| `step_sizes` | (61,) | the finite-difference step per parameter |
| `fiducial_params` | (61,) | the point the expansion is about |
| `camb` | dict | `set_cosmology`, `set_for_lmax`, `matter_power`, `accuracy` |
| `cosmo_names`, `nuisance_names` | lists | the block boundaries |
| `tied_parameters` | dict | e.g. `{'A_planck': 'A_act'}` |
| `fiducial_cosmology`, `fixed_cosmology` | dicts | the cosmology it was evaluated at |
| `likelihood_names` | list | which data went in |
| `steepness` | float | the bin basis sigmoid steepness |

## Level 1 — the whole chain in one call

```python
from lensing_fisher import analysis
s = analysis.summarize("spa_iter1.pkl")
```

| key | what |
|---|---|
| `L_median`, `L_minus`, `L_plus` | **the numbers to quote** — 126.6, [60.6, 242.8] |
| `sigma_A_template` | `1/sqrt(sum w_j)` = 0.0263 |
| `w_bins` (51,) | each bin's share of the `A_template` information |
| `w` (8549,), `L_grid` (8549,) | that share as a **density in L**, and the axis |
| `F_lens` (51, 51) | the marginalized `clpp` block |
| `r` (51,) | Jacobian to `A_template`; all 1 when `C_fid` *is* the template |
| `L_eff` | the mean — fragile, see below |
| `moment_L_max`, `excluded_catchall`, `excluded_information` | which range the moments used, and how much was left out |

Pass `template_file=` only when the Fisher's `C_fid` differs from the template
you actually care about; then `r_j = C_tem/C_fid` is computed per bin and applied.
When `bins.template.file` already points at that template, omit it — `r` is
identically 1.

## Level 2 — the building blocks

Each step is a separate function, so you can inspect or replace any of them.

| function | does |
|---|---|
| `load_fisher(path)` | read the pickle |
| `clpp_indices(param_names)` | the bin rows/columns, ordered **numerically** (lexical order would put `clpp_10` before `clpp_2` and silently scramble the grid) |
| `marginalize(F, names)` | Schur complement over cosmology + nuisance |
| `information_per_bin(F_lens, r)` | `w_j = r_j (F r)_j` |
| `weight_density(w_bins, edges, L_grid)` | `w_j / dL_j` |
| `effective_L(w, L_grid)` | `(mean, 16th, 84th)` |
| `weight_quantile(w, L_grid, q)` | any quantile; `q=0.5` is the median |
| `bin_averaged_ratio(edges, C_tem, C_fid, ells)` | `r_j`, `C_fid`-weighted across each bin |
| `next_bin_edges(w, L_grid, n, min_width, L_range)` | the equal-information placement |
| `edges_converged(old, new, tol)` | the convergence test |

## Level 3 — raw

`pickle.load` and index it yourself. You own the marginalization; see the traps.

## Level 4 — the CLI

```bash
lensing-fisher-rebin spa_iter1.pkl -n 50 --min-width 3 --l-max 2000 -o edges_2.yaml
```

Prints the summary, proposes the next grid as a YAML fragment for `bins.edges`,
and reports edge movement against the grid that produced the input.

## Plotting where the data are sensitive

The one-liner, since `w` is already a density:

```python
s = analysis.summarize("spa_iter0.pkl")
plt.plot(s["L_grid"], s["w"]); plt.xscale("log")
```

Or from the pieces, dropping the catch-all and stepping on the real edges:

```python
d       = analysis.load_fisher("spa_iter0.pkl")
F       = analysis.marginalize(d["fisher_matrix"], d["param_names"])
edges   = np.asarray(d["bin_edges"])
widths  = np.diff(edges)
centres = np.sqrt(edges[:-1] * edges[1:])          # geometric — the axis is log L
w       = analysis.information_per_bin(F, np.ones(len(F)))
plt.stairs(w[:-1] / widths[:-1], edges[:-1])       # [:-1] drops the catch-all
```

`stairs` on the actual edges rather than a line through `centres`, because bin
widths span three orders of magnitude and a line implies a resolution the grid
does not have.

Use **iteration 0** for this plot. Its log grid is roughly uniform in log L and
shows the shape directly. On an equal-information grid `w_j` is flat *by
construction* — that is its success criterion — so plot the density there or you
will see a flat line and think it is broken.

## Traps

### `diag(F)` is not the information density

`F_ii` is the information on `q_i`, a *fractional* perturbation of the whole bin,
so a wider bin holds more lensing power and earns a larger `F_ii` for free. On a
log grid with widths from 0.3 to 6550, the raw diagonal is largely reporting the
binning. Concretely, on iteration 0:

```
raw diag(F)   peaks at L = 135   <- and the catch-all bin is SECOND LARGEST
diag(F)/dL    peaks at L = 118
w_j/dL        peaks at L =  89
```

The catch-all spans L=2000..8550 and carries ~0% of the information, but it is
6550 wide, so its raw `F_ii` (2.09) beats everything above L~250. Plot the raw
diagonal and sensitivity appears to *rise* into the tail.

Also decide which "information" you mean: `diag(F_lens)` is **conditional** —
how well a bin is measured with all others held fixed — while `w_j = (F r)_j` is
that bin's **share of the constraint on the overall amplitude**, accounting for
the strong anticorrelation between neighbours. `L_eff` is a moment of the second.

### Marginalize, do not slice

Slicing `fisher_matrix` down to the `clpp` block *conditions* on cosmology being
known exactly. `analysis.marginalize` takes the Schur complement. (It does this
as `inv(inv(F))[block]`; that agrees with a direct Schur complement to 4e-9 here,
so the extra inversion is not a problem in practice.)

### Do not invert `F_lens`

It is rank-deficient by construction — the lensed CMB resolves ~4 modes out of 51
— so `inv` returns NaN and `sqrt(diag(inv(F)))` is NaN for the degenerate
directions. This is expected, not corruption, and no amount of accuracy or step
tuning changes it. Everything above uses `F @ r`, never `inv(F)`.

`FisherMatrix.summary()` reports the resolved-mode count and prints `--` rather
than NaN for those parameters; `driver.clpp_step_sizes` raises rather than
feeding NaNs into the next run.

### `L_eff` is the fragile one

`weight_density` spreads each bin's information uniformly across the bin. That is
harmless for a bin a few multipoles wide and badly wrong for one thousands wide,
where the real information falls steeply toward the low edge — and a mean takes
the full lever arm of that width.

Measured: with the catch-all at [999, 8550] holding 2.1% of the information,
including it moved `L_eff` from 164.2 to **261.2**, past its own 68% upper bound
of 242.8. A mean outside its own central interval is the tell. The median moved
0.5 and the band 1.6 across the same change.

So `summarize` excludes the last bin from the moments by default, and the CLI
prints a NOTE when mean and median differ by more than a quarter of the 68%
width. **Prefer the median.** `w_bins` and `sigma_A_template` are unaffected
either way — they are sums, with no lever arm.
