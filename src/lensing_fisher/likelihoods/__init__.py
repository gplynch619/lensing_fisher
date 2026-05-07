"""Likelihood loading from YAML configs.

The YAML ``likelihoods:`` block is a mapping of instance-name → spec, where
each spec has a ``backend:`` key (``clipy`` or ``candl``) plus backend-specific
kwargs. ``build_likelihoods`` returns a list of likelihood objects (in YAML
order) and a dict of merged fiducial nuisance-parameter values.
"""

from typing import Any

from .registry import LIKELIHOOD_FACTORIES, get_factory, register

# Import backends to register them.
from . import clipy_backend  # noqa: F401
from . import candl_backend  # noqa: F401


def _extract_clipy_nuisance_fiducials(like) -> dict[str, float]:
    """Pull nuisance fiducial values from a clipy likelihood's default_par dict."""
    out: dict[str, float] = {}
    for k, v in like.default_par.items():
        if k == "Dl":
            continue
        out[k] = float(v)
    return out


def _extract_candl_nuisance_fiducials(like) -> dict[str, float]:
    """Pull nuisance fiducial values from a candl likelihood's prior central values.

    Falls back to 1.0 for known calibration-style parameters (``Ecal``, ``P_act``)
    that are required but not in any prior — preserves the behavior of the old
    driver at lensing_sensitivity_fisher.py:373-388.
    """
    out: dict[str, float] = {}
    for par_name in like.required_nuisance_parameters:
        found = False
        for prior in like.priors:
            if par_name in prior.par_names:
                idx = prior.par_names.index(par_name)
                out[par_name] = float(prior.central_value[idx])
                found = True
                break
        if not found and par_name in ("Ecal", "P_act"):
            out[par_name] = 1.0
    return out


def build_likelihoods(cfg: dict) -> tuple[list[Any], dict[str, float]]:
    """Build all likelihoods listed in ``cfg["likelihoods"]``.

    Returns
    -------
    likelihoods : list
        Likelihood objects in YAML order.
    nuisance_fiducials : dict
        Merged fiducial values for nuisance parameters across all likelihoods.
        Earlier-listed likelihoods win on conflicts (matches old driver behavior).
    """
    spec_block = cfg.get("likelihoods")
    if not spec_block:
        raise ValueError("config has no 'likelihoods:' block")

    likelihoods: list[Any] = []
    nuisance: dict[str, float] = {}

    for name, spec in spec_block.items():
        spec = dict(spec)
        backend = spec.pop("backend", None)
        if backend is None:
            raise ValueError(f"likelihood {name!r}: missing required 'backend' key")
        factory = get_factory(backend)
        like = factory(**spec)
        likelihoods.append(like)

        if backend == "clipy":
            extracted = _extract_clipy_nuisance_fiducials(like)
        elif backend == "candl":
            extracted = _extract_candl_nuisance_fiducials(like)
        else:
            extracted = {}
        for k, v in extracted.items():
            nuisance.setdefault(k, v)

    return likelihoods, nuisance


__all__ = [
    "LIKELIHOOD_FACTORIES",
    "build_likelihoods",
    "get_factory",
    "register",
]
