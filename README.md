# lensing_fisher

A small package for CMB 2pt lensing sensitivity analyses. Two pieces:

1. **`FisherMatrix`** — a generic, MPI-aware Fisher matrix calculator (4th-order central differences, forward differences for parameters with fiducial = 0, optional adaptive step selection, pickle output).
2. **YAML-driven Fisher driver** for the "local-bump Cl_pp" parametrization used to derive an effective L for the 2pt lensing measurement, against any combination of candl/clipy likelihoods.
3. **`TemplateLensingCAMB`** — a Cobaya CAMB theory subclass that lenses the primary CMB with a fixed template Cl_pp scaled by an `A_template` parameter (used for the template-amplitude chain).

## Install

```bash
git clone <this repo>
cd lensing_fisher
pip install -e .
# optional extras
pip install -e '.[mpi,cobaya,likelihoods,dev]'
```

The `likelihoods` extra is a placeholder — `candl` and `clipy` typically come from your cosmology environment, not pip, so install them separately.

## Run a Fisher matrix

```bash
mpirun -n 8 lensing-fisher -i examples/lensing_sensitivity.yaml
```

The example YAML reproduces the run that produced
`data/full_fishers/lensing_sensitivity_reference.pkl` in the parent project.

## Use `TemplateLensingCAMB` from Cobaya

In your cobaya YAML:

```yaml
theory:
  lensing_fisher.theory.template_lensing.TemplateLensingCAMB:
    clpp_template_file: /path/to/unlensed_bestfit_pp.pkl
```

Then add `A_template` to your `params:` block.
