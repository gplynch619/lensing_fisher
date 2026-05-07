"""Fisher matrix driver: turns a parsed YAML config into a saved Fisher pickle.

Use ``compute_fisher_matrix(cfg, comm)``; the CLI in ``cli.py`` wraps this with
argparse + MPI.COMM_WORLD.
"""

from typing import Any, Optional

import numpy as np

from .fisher import FisherMatrix
from .likelihoods import build_likelihoods
from .theory.local_lens import build_local_lens_theory


def _build_camb_pars(camb_cfg: dict, fid: dict):
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=fid["H0"],
        ombh2=fid["ombh2"],
        omch2=fid["omch2"],
        tau=fid["tau"],
        mnu=fid.get("mnu", 0.06),
        omk=fid.get("omk", 0.0),
    )
    pars.InitPower.set_params(As=1e-10 * np.exp(fid["logA"]), ns=fid["ns"])

    pars.set_for_lmax(**camb_cfg["set_for_lmax"])
    pars.WantTensors = False
    pars.WantCls = True
    pars.WantScalars = True

    pars.set_matter_power(**camb_cfg["matter_power"])
    pars.set_accuracy(**camb_cfg["accuracy"])
    return pars


def _resolve_L_centers(spec) -> np.ndarray:
    """Accept either a list of L values or a {start, stop, step} dict."""
    if isinstance(spec, dict):
        return np.arange(spec["start"], spec["stop"] + 1, step=spec["step"])
    return np.asarray(spec)


def _build_combined_loglike(likelihoods, pars_to_theory_specs, tau_prior_cfg: Optional[dict]):
    import candl.tools  # type: ignore

    like_funcs = [
        candl.tools.get_params_to_logl_func(like, pars_to_theory_specs)
        for like in likelihoods
    ]

    if tau_prior_cfg is not None:
        tau_mean = float(tau_prior_cfg["mean"])
        tau_sigma = float(tau_prior_cfg["sigma"])
    else:
        tau_mean = tau_sigma = None

    def combined(pars: dict) -> float:
        val = 0.0
        for lf in like_funcs:
            val += lf(pars)
        if tau_mean is not None and "tau" in pars:
            val -= ((tau_mean - pars["tau"]) / tau_sigma) ** 2.0
        return val

    return combined


def compute_fisher_matrix(cfg: dict, comm: Any = None) -> str:
    """Run the Fisher matrix computation specified by ``cfg``.

    Returns the saved pickle path on rank 0; empty string on non-zero ranks.
    """
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1

    if rank == 0:
        print(f"[lensing_fisher] starting Fisher computation on {size} MPI rank(s)")

    cosmo_cfg = cfg["cosmology"]
    fid: dict[str, float] = {k: float(v) for k, v in cosmo_cfg["fiducial"].items()}
    fixed = {k: float(v) for k, v in cosmo_cfg.get("fixed", {}).items()}

    if rank == 0:
        print("[lensing_fisher] building likelihoods...")
    likelihoods, nuisance_fid = build_likelihoods(cfg)
    if comm is not None:
        comm.Barrier()

    for k, v in nuisance_fid.items():
        fid.setdefault(k, v)

    camb_pars = _build_camb_pars(cfg["camb"], {**fid, **fixed})

    bins_cfg = cfg["bins"]
    if bins_cfg.get("parametrization", "local_lens") != "local_lens":
        raise ValueError(
            f"only 'local_lens' parametrization is supported; got "
            f"{bins_cfg.get('parametrization')!r}"
        )
    L_centers = _resolve_L_centers(bins_cfg["L_centers"])
    Nlens = len(L_centers)

    pars_to_theory_specs = build_local_lens_theory(
        camb_pars,
        L_centers,
        width=float(bins_cfg["width"]),
        steepness=float(bins_cfg.get("steepness", 1.0)),
        amp_unit=float(bins_cfg.get("amp_unit", 1e-7)),
    )

    Ncosmo = len(fid)  # cosmology + nuisance, before adding clpp_*
    for i in range(Nlens):
        fid[f"clpp_{i+1}"] = 0.0

    cosmo_param_names = list(cosmo_cfg["fiducial"].keys())

    step_sizes_rel = cosmo_cfg.get("step_sizes_relative", {})
    step_sizes_abs = cosmo_cfg.get("step_sizes_absolute", {})
    nuisance_step_default_rel = float(cosmo_cfg.get("nuisance_step_relative", 0.01))
    clpp_step_abs = float(bins_cfg["step_size_abs"])

    step_sizes = []
    for name in fid:
        if name.startswith("clpp"):
            step_sizes.append(clpp_step_abs)
        elif name in step_sizes_abs:
            step_sizes.append(float(step_sizes_abs[name]))
        elif name in step_sizes_rel:
            step_sizes.append(float(step_sizes_rel[name]) * fid[name])
        elif name in cosmo_param_names:
            raise ValueError(
                f"cosmology parameter {name!r} has no step size in "
                f"step_sizes_relative or step_sizes_absolute"
            )
        else:
            step_sizes.append(nuisance_step_default_rel * fid[name])
    step_sizes = np.asarray(step_sizes, dtype=float)

    fd_scheme = np.array(Ncosmo * [True] + Nlens * [False])

    like_fn = _build_combined_loglike(
        likelihoods, pars_to_theory_specs, cosmo_cfg.get("tau_prior")
    )

    if rank == 0:
        print(f"[lensing_fisher] {len(fid)} parameters total "
              f"({Ncosmo} cosmo+nuisance, {Nlens} clpp bins)")

    fm = FisherMatrix(like_fn, fid, use_central_differences=fd_scheme, comm=comm)
    fm.compute_fisher_matrix(step_size=step_sizes, adaptive=False, relative=False)

    if comm is not None:
        comm.Barrier()

    out_cfg = cfg["output"]
    saved_path = fm.save_to_pickle(out_cfg["directory"], out_cfg["filename"])
    if rank == 0:
        print(f"[lensing_fisher] saved Fisher matrix to {saved_path}")
    return saved_path
