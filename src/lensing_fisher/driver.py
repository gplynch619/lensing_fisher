"""Config -> saved Fisher matrix.

``run(cfg, comm)`` reads as the sequence of steps the calculation actually
performs; each step is a small function below that can be read, called and
tested on its own.

    1. build the likelihoods           build_likelihoods
    2. build CAMB parameters           build_camb_params
    3. check the theory reaches the data   windows.check_theory_lmax
    4. lay out the Cl_pp bins          resolve_bin_edges (+ the catch-all bin)
    5. load the frozen C_fid           load_clpp_fid
    6. assemble parameters and steps   assemble_parameters
    7. differentiate                   FisherMatrix.compute
    8. save with provenance            FisherMatrix.save
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import clpp_template, windows
from .mpi_log import info, set_rank
from .fisher import FisherMatrix
from .likelihoods import build_likelihoods, combined_loglike
from .local_lens import COSMO_KEYS, BinnedLensingTheory


# ----------------------------------------------------------------------
# Step 2: CAMB
# ----------------------------------------------------------------------

def build_camb_params(camb_cfg: dict, cosmology: dict):
    """A ``CAMBparams`` at the fiducial cosmology with the configured accuracy."""
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=cosmology["H0"],
        ombh2=cosmology["ombh2"],
        omch2=cosmology["omch2"],
        tau=cosmology["tau"],
        mnu=cosmology.get("mnu", 0.06),
        omk=cosmology.get("omk", 0.0),
    )
    pars.InitPower.set_params(As=1e-10 * np.exp(cosmology["logA"]), ns=cosmology["ns"])
    pars.set_for_lmax(**camb_cfg["set_for_lmax"])
    pars.set_matter_power(**camb_cfg["matter_power"])
    pars.set_accuracy(**camb_cfg["accuracy"])
    pars.WantTensors = False
    pars.WantCls = True
    pars.WantScalars = True
    return pars


# ----------------------------------------------------------------------
# Step 4: bins
# ----------------------------------------------------------------------

def resolve_bin_edges(bins_cfg: dict) -> np.ndarray:
    """Bin edges from either an explicit list or a generator spec.

    ``edges: [2, 12, 30, ...]``          explicit, arbitrary widths
    ``edges: {start, stop, n, spacing}`` generated, ``spacing`` log or linear
    """
    spec = bins_cfg["edges"]
    if not isinstance(spec, dict):
        return np.asarray(spec, dtype=float)

    start, stop, n = float(spec["start"]), float(spec["stop"]), int(spec["n"])
    spacing = spec.get("spacing", "log")
    if spacing == "log":
        return np.geomspace(max(start, 1e-6), stop, n + 1)
    if spacing == "linear":
        return np.linspace(start, stop, n + 1)
    raise ValueError(f"bins.edges.spacing must be 'log' or 'linear'; got {spacing!r}")


def add_catchall_bin(bin_edges: np.ndarray, camb_lmax: int) -> np.ndarray:
    """Extend the grid to ``camb_lmax`` so every lensed multipole belongs to a bin.

    The outermost basis functions are one-sided, so the last bin scales lensing
    all the way to ``camb_lmax`` whatever its nominal top edge says. Making that
    explicit keeps the bin width honest for the analysis step, and makes a
    uniform q exactly A_template rather than "A_template below the top edge".
    """
    if bin_edges[-1] >= camb_lmax - 1:
        return bin_edges
    return np.concatenate([bin_edges, [float(camb_lmax)]])


# ----------------------------------------------------------------------
# Step 5: the frozen fiducial lensing spectrum
# ----------------------------------------------------------------------

def load_clpp_fid(template_cfg: dict, camb_pars) -> np.ndarray:
    """The frozen C_fid, on CAMB's ell grid.

    ``{file: <{L, CL_pp_fid} pickle>}`` is the normal case — point it at the
    unlensed-dataset best fit and the Jacobian to A_template is identically 1.
    ``{from_fiducial_cosmology: true}`` derives it from the fiducial cosmology
    instead, for iterations that run before that best fit exists.
    """
    lmax = camb_pars.max_l
    if template_cfg.get("from_fiducial_cosmology"):
        info(f"C_fid from the fiducial cosmology, lmax={lmax}")
        return clpp_template.clpp_fid_from_camb(camb_pars, lmax)

    path = template_cfg["file"]
    L, CL = clpp_template.load_template(path)
    info(f"C_fid from {path} (L={L[0]:.0f}..{L[-1]:.0f} -> lmax={lmax})")
    return clpp_template.interpolate_to_ells(L, CL, np.arange(lmax + 1))


# ----------------------------------------------------------------------
# Step 6: parameters and step sizes
# ----------------------------------------------------------------------

@dataclass
class ParameterSet:
    """The vector the Fisher matrix is taken over, with its finite-difference steps."""

    fiducial: dict = field(default_factory=dict)
    steps: np.ndarray = field(default_factory=lambda: np.array([]))
    cosmo_names: list = field(default_factory=list)
    nuisance_names: list = field(default_factory=list)
    bin_names: list = field(default_factory=list)

    @property
    def names(self) -> list:
        return list(self.fiducial)


def clpp_step_sizes(step_cfg, bin_edges: np.ndarray) -> np.ndarray:
    """Finite-difference step for each bin amplitude.

    A plain number applies one fractional step to every bin. That is fine for a
    uniform grid but poor for an equal-information one, where narrow bins hold
    less lensing power and so are more weakly constrained. The dict form

        step_size: {from_fisher: prev.pkl, target_sigma_frac: 0.3}

    sizes each step as a fraction of that bin's marginal ``sigma(q_j)`` from the
    previous iteration, interpolated in log L onto the new grid, so every bin
    keeps a comparable finite-difference signal as the bins move around.
    """
    n_bins = len(bin_edges) - 1
    if not isinstance(step_cfg, dict):
        return np.full(n_bins, float(step_cfg))

    from .analysis import load_fisher, marginalize

    previous = load_fisher(step_cfg["from_fisher"])
    sigma_prev = np.sqrt(np.diag(np.linalg.inv(
        marginalize(previous["fisher_matrix"], previous["param_names"])
    )))

    prev_edges = np.asarray(previous["bin_edges"], dtype=float)
    prev_centers = 0.5 * (prev_edges[:-1] + prev_edges[1:])
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sigma = np.exp(np.interp(np.log(centers), np.log(prev_centers), np.log(sigma_prev)))

    fraction = float(step_cfg.get("target_sigma_frac", 0.3))
    steps = np.clip(fraction * sigma,
                    float(step_cfg.get("min", 0.02)),
                    float(step_cfg.get("max", 1.0)))
    info(f"clpp steps = {fraction} x sigma(q) from {step_cfg['from_fisher']}: "
         f"{steps.min():.3f}..{steps.max():.3f}")
    return steps


def assemble_parameters(cosmo_cfg: dict, bins_cfg: dict,
                        nuisance_fid: dict, bin_edges: np.ndarray) -> ParameterSet:
    """Collect cosmology + nuisance + bin amplitudes into one ordered vector.

    Order is cosmology, then nuisance, then ``clpp_1 .. clpp_n``; it sets the
    row/column order of the Fisher matrix and the block boundaries the analysis
    step relies on.
    """
    cosmo_names = list(cosmo_cfg["fiducial"])
    nuisance_names = [n for n in nuisance_fid if n not in cosmo_names]
    bin_names = [f"clpp_{i + 1}" for i in range(len(bin_edges) - 1)]

    fiducial = {name: float(cosmo_cfg["fiducial"][name]) for name in cosmo_names}
    fiducial.update({name: float(nuisance_fid[name]) for name in nuisance_names})
    fiducial.update({name: 0.0 for name in bin_names})

    relative = cosmo_cfg.get("step_sizes_relative", {})
    absolute = cosmo_cfg.get("step_sizes_absolute", {})
    nuisance_relative = float(cosmo_cfg.get("nuisance_step_relative", 0.01))

    steps = []
    for name in cosmo_names:
        if name in absolute:
            steps.append(float(absolute[name]))
        elif name in relative:
            steps.append(float(relative[name]) * fiducial[name])
        else:
            raise ValueError(
                f"cosmology parameter {name!r} has no entry in step_sizes_relative "
                f"or step_sizes_absolute"
            )
    steps += [nuisance_relative * fiducial[name] for name in nuisance_names]
    steps += list(clpp_step_sizes(bins_cfg.get("step_size", 0.05), bin_edges))

    return ParameterSet(
        fiducial=fiducial,
        steps=np.asarray(steps, dtype=float),
        cosmo_names=cosmo_names,
        nuisance_names=nuisance_names,
        bin_names=bin_names,
    )


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run(cfg: dict, comm: Any = None) -> str:
    """Run the Fisher computation described by ``cfg``; return the saved path."""
    set_rank(comm.Get_rank() if comm is not None else 0)

    cosmo_cfg, camb_cfg, bins_cfg = cfg["cosmology"], cfg["camb"], cfg["bins"]

    names, likelihoods, nuisance_fid = build_likelihoods(cfg["likelihoods"])

    fixed = {k: float(v) for k, v in cosmo_cfg.get("fixed", {}).items()}
    camb_pars = build_camb_params(camb_cfg, {**cosmo_cfg["fiducial"], **fixed})

    # Every rank checks: this must abort the whole job, not just rank 0.
    windows.check_theory_lmax(
        names, likelihoods, camb_pars.max_l,
        tol=float(camb_cfg.get("window_support_tol", 1e-3)),
    )

    bin_edges = resolve_bin_edges(bins_cfg)
    if bins_cfg.get("extend_to_lmax", True):
        bin_edges = add_catchall_bin(bin_edges, camb_pars.max_l)

    clpp_fid = load_clpp_fid(bins_cfg["template"], camb_pars)

    theory = BinnedLensingTheory(
        camb_pars, bin_edges, clpp_fid,
        steepness=float(bins_cfg.get("steepness", 2.0)),
        cache_size=int(bins_cfg.get("camb_cache_size", 4)),
    )

    params = assemble_parameters(cosmo_cfg, bins_cfg, nuisance_fid, bin_edges)
    loglike = combined_loglike(likelihoods, theory, cosmo_cfg.get("tau_prior"))

    widths = np.diff(bin_edges)
    info(f"{len(params.fiducial)} parameters: {len(params.cosmo_names)} cosmology, "
         f"{len(params.nuisance_names)} nuisance, {len(params.bin_names)} bins")
    info(f"bins: L={bin_edges[0]:.0f}..{bin_edges[-1]:.0f}, "
         f"widths {widths.min():.1f}..{widths.max():.0f}")

    fisher = FisherMatrix(
        loglike, params.fiducial, params.steps, comm=comm,
        cached_params=[p for p in params.cosmo_names if p in COSMO_KEYS],
    )
    fisher.compute()

    hits, misses = theory.cache_stats["hits"], theory.cache_stats["misses"]
    info(f"CAMB cache on rank 0: {hits}/{hits + misses} hits "
         f"({100.0 * hits / max(hits + misses, 1):.1f}%)")
    if fisher.rank == 0:
        fisher.summary()

    path = fisher.save(
        cfg["output"]["directory"], cfg["output"]["filename"],
        metadata=_provenance(cfg, params, bin_edges, clpp_fid, names),
    )
    info(f"saved to {path}")
    return path


def _provenance(cfg, params: ParameterSet, bin_edges, clpp_fid, names) -> dict:
    """Everything the analysis step needs, so bin geometry is never re-derived."""
    return {
        "bin_edges": bin_edges,
        "clpp_fid": clpp_fid,
        "steepness": float(cfg["bins"].get("steepness", 2.0)),
        "cosmo_names": params.cosmo_names,
        "nuisance_names": params.nuisance_names,
        "fiducial_cosmology": {k: params.fiducial[k] for k in params.cosmo_names},
        "fixed_cosmology": cfg["cosmology"].get("fixed", {}),
        "likelihood_names": names,
        "camb": cfg["camb"],
    }
