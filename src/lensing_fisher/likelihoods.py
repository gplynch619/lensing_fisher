"""Build candl / clipy likelihoods from a YAML ``likelihoods:`` block.

The block is a mapping of instance name -> spec, each spec carrying a
``backend:`` key plus backend-specific arguments::

    likelihoods:
      planck_highl_ttteee:
        backend: clipy
        clik_path: .../plik_lite_v22_TTTEEE.clik
        crop: ["crop TT 0 1000 strict"]
      act_dr6:
        backend: candl
        dataset: candl_data.ACT_DR6_TTTEEE
        ell_cuts: {TE: [600, 1000], EE: [600, 1000]}

The same block drives the Fisher driver and the Cobaya chain (via
:mod:`lensing_fisher.cobaya_likelihood`), so the two cannot see different data.
"""

import importlib

import numpy as np

from .mpi_log import info


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------

def _resolve_dotted(path: str):
    """Resolve ``"spt_candl_data.SPT3G_D1_TnE_lite"`` to the object it names."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"dataset must be a dotted path, got {path!r}")
    return getattr(importlib.import_module(module_name), attr)


def ell_cut_mask(like, ell_cuts: dict) -> list:
    """Per-bandpower boolean mask implementing per-spectrum ell ranges.

    ``ell_cuts`` maps a candl spectrum type to an inclusive ``[lmin, lmax]``,
    tested against bin effective ells. **A spectrum absent from ``ell_cuts`` is
    dropped entirely** — that is how ACT is restricted to TE/EE only, leaving
    Planck to supply TT below ell=1000.
    """
    eff = np.asarray(like.effective_ells)
    mask = np.zeros(int(like.bins_stop_ix[-1]), dtype=bool)
    for spec, start, stop in zip(like.spec_types, like.bins_start_ix, like.bins_stop_ix):
        if spec not in ell_cuts:
            continue
        lo, hi = ell_cuts[spec]
        mask[start:stop] = (eff[start:stop] >= lo) & (eff[start:stop] <= hi)
    return mask.tolist()


def make_candl(*, dataset: str, ell_cuts=None, **kwargs):
    """``candl.Like`` from a dotted dataset path, optionally ell-cut.

    Note that candl's ``ell_min``/``ell_max`` are *not* reduced by the resulting
    ``data_selection``; a cut likelihood still requests theory over its original
    range. :mod:`lensing_fisher.windows` deals with the consequences.
    """
    import candl

    data = _resolve_dotted(dataset)
    if ell_cuts is not None:
        ell_cuts = {str(k): [float(v[0]), float(v[1])] for k, v in ell_cuts.items()}
        probe = candl.Like(data, feedback=False)
        dropped = [s for s in dict.fromkeys(probe.spec_types) if s not in ell_cuts]
        mask = ell_cut_mask(probe, ell_cuts)
        if not any(mask):
            raise ValueError(f"{dataset}: ell_cuts {ell_cuts} kept no bandpowers")
        info(f"{dataset}: keeps {sum(mask)}/{len(mask)} bandpowers"
             + (f"; drops spectra {dropped}" if dropped else ""))
        kwargs["data_selection"] = mask
    return candl.Like(data, **kwargs)


def make_clipy(*, clik_path: str, all_priors: bool = True, crop=None, **kwargs):
    """``clipy.clik_candl`` from a .clik path, optionally cropped.

    ``crop`` takes clipy's own strings, e.g. ``["crop TT 0 1000 strict"]``.
    """
    import clipy

    if crop is not None:
        kwargs["crop"] = list(crop)
    return clipy.clik_candl(clik_path, all_priors=all_priors, **kwargs)


BACKENDS = {"candl": make_candl, "clipy": make_clipy}


# ----------------------------------------------------------------------
# Nuisance fiducials
# ----------------------------------------------------------------------

def _clipy_nuisance(like) -> dict:
    """Nuisance fiducials from a clipy likelihood's ``default_par``."""
    return {k: float(v) for k, v in like.default_par.items() if k != "Dl"}


def _candl_nuisance(like) -> dict:
    """Nuisance fiducials from a candl likelihood's prior central values.

    Pure calibration parameters (``Ecal``, ``P_act``) are required but carry no
    prior, so they default to 1.
    """
    out = {}
    for name in like.required_nuisance_parameters:
        for prior in like.priors:
            if name in prior.par_names:
                out[name] = float(prior.central_value[prior.par_names.index(name)])
                break
        else:
            out[name] = 1.0
    return out


_NUISANCE_EXTRACTORS = {"candl": _candl_nuisance, "clipy": _clipy_nuisance}


# ----------------------------------------------------------------------

def build_likelihoods(spec_block: dict):
    """Build every likelihood in a ``likelihoods:`` block.

    Returns ``(names, likelihoods, nuisance_fiducials)``, the first two in YAML
    order. On a name clash between likelihoods the earlier one wins.
    """
    if not spec_block:
        raise ValueError("empty 'likelihoods:' block")

    names, likelihoods, nuisance = [], [], {}
    for name, spec in spec_block.items():
        spec = dict(spec)
        backend = spec.pop("backend")
        if backend not in BACKENDS:
            raise ValueError(
                f"likelihood {name!r}: unknown backend {backend!r}; "
                f"expected one of {sorted(BACKENDS)}"
            )
        like = BACKENDS[backend](**spec)
        names.append(name)
        likelihoods.append(like)
        for key, value in _NUISANCE_EXTRACTORS[backend](like).items():
            nuisance.setdefault(key, value)

    return names, likelihoods, nuisance


def combined_loglike(likelihoods, theory, tau_prior=None):
    """Sum of ``log_like`` over likelihoods, as a function of a parameter dict.

    ``theory`` is a ``pars_to_theory_specs(pars, ell_high, ell_low)`` callable —
    in practice a :class:`~lensing_fisher.local_lens.BinnedLensingTheory`.
    ``tau_prior`` is an optional ``{mean, sigma}`` Gaussian, needed only when the
    dataset carries no low-ell EE likelihood to constrain tau itself.
    """
    import candl.tools

    like_funcs = [candl.tools.get_params_to_logl_func(lk, theory) for lk in likelihoods]

    if tau_prior is None:
        return lambda pars: sum(f(pars) for f in like_funcs)

    mean, sigma = float(tau_prior["mean"]), float(tau_prior["sigma"])

    def with_prior(pars):
        total = sum(f(pars) for f in like_funcs)
        return total - ((mean - pars["tau"]) / sigma) ** 2.0

    return with_prior
