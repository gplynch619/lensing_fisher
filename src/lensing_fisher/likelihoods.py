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


def drop_parameter_priors(like, names, label: str = "") -> None:
    """Remove a candl likelihood's internal Gaussian priors on ``names``, in place.

    candl datasets ship stand-in priors for parameters they cannot constrain
    themselves. ACT DR6 carries ``tau = 0.0566 +/- 0.0058`` and SPT-3G D1
    ``tau = 0.051 +/- 0.006``, both applied inside ``log_like``. They are meant
    as an *alternative* to a low-ell EE likelihood — the ACT dataset file says as
    much — so a combination that already includes sroll2 and keeps them counts
    tau two or three times over. For SPA that turns sigma(tau) ~ 0.007 into
    ~ 0.0035, and since tau is degenerate with A_s it propagates straight into
    the lensing amplitude this package exists to measure.

    Note this is *selective*, which is the reason it exists. candl's own
    ``clear_internal_priors`` is all-or-nothing, and clearing everything would
    also drop the calibration priors (``A_act``, ``Tcal``), which are genuine
    nuisance constraints we want to keep.

    **Call this before the likelihood's first evaluation.** candl caches the
    compiled log-likelihood on first ``log_like`` call, so editing the prior list
    after that is silently ignored — the list shrinks, the logl does not change,
    and nothing warns. :func:`make_candl` does it at construction, which is the
    only supported place; do not call it on a likelihood already in use.

    Raises rather than no-ops on an unmatched name: a silently ineffective
    ``drop_priors`` looks exactly like a correct one in the output.
    """
    wanted = set(names)
    kept, removed = [], set()
    for prior in like.priors:
        matched = wanted.intersection(prior.par_names)
        if not matched:
            kept.append(prior)
            continue
        others = set(prior.par_names) - wanted
        if others:
            raise ValueError(
                f"{label}: prior on {prior.par_names} is joint over {sorted(others)}, "
                f"which drop_priors {sorted(wanted)} would silently remove too"
            )
        removed |= matched

    if wanted - removed:
        raise ValueError(
            f"{label}: drop_priors names {sorted(wanted - removed)}, which carries "
            f"no internal prior here (has: "
            f"{sorted({p for pr in like.priors for p in pr.par_names})})"
        )

    like.priors = kept
    like.init_priors()   # belt and braces; the ordering above is what matters
    info(f"{label}: dropped internal prior(s) on {sorted(removed)}")


def make_candl(*, dataset: str, ell_cuts=None, drop_priors=(), **kwargs):
    """``candl.Like`` from a dotted dataset path, optionally ell-cut.

    Note that candl's ``ell_min``/``ell_max`` are *not* reduced by the resulting
    ``data_selection``; a cut likelihood still requests theory over its original
    range. :mod:`lensing_fisher.windows` deals with the consequences.

    ``drop_priors: [tau]`` removes internal priors that would double-count a
    likelihood the dataset already includes; see :func:`drop_parameter_priors`.
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

    like = candl.Like(data, **kwargs)
    if drop_priors:
        drop_parameter_priors(like, drop_priors, label=dataset)
    return like


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


def combined_loglike(likelihoods, theory, tau_prior=None, tied=None):
    """Sum of ``log_like`` over likelihoods, as a function of a parameter dict.

    ``theory`` is a ``pars_to_theory_specs(pars, ell_high, ell_low)`` callable —
    in practice a :class:`~lensing_fisher.local_lens.BinnedLensingTheory`.
    ``tau_prior`` is an optional ``{mean, sigma}`` Gaussian, needed only when the
    dataset carries no low-ell EE likelihood to constrain tau itself.

    ``tied`` maps a parameter onto another that supplies its value, e.g.
    ``{"A_planck": "A_act"}`` for a single shared calibration across Planck and
    ACT. Tied parameters are filled in here and are *not* members of the Fisher
    vector — see :func:`~lensing_fisher.driver.assemble_parameters`. This mirrors
    Cobaya's ``A_planck: {value: 'lambda A_act: A_act'}``, so the chain and the
    Fisher vary the same number of things.
    """
    import candl.tools

    like_funcs = [candl.tools.get_params_to_logl_func(lk, theory) for lk in likelihoods]
    tied = dict(tied or {})

    def evaluate(pars):
        if tied:
            pars = {**pars, **{target: pars[source] for target, source in tied.items()}}
        total = sum(f(pars) for f in like_funcs)
        if tau_prior is not None:
            mean, sigma = float(tau_prior["mean"]), float(tau_prior["sigma"])
            total -= ((mean - pars["tau"]) / sigma) ** 2.0
        return total

    return evaluate
