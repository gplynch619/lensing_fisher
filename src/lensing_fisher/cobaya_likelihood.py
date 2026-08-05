"""Cobaya likelihood that reuses the Fisher driver's own dataset definition.

The point of this class is consistency. The Fisher driver builds its likelihoods
from a YAML ``likelihoods:`` block via :func:`build_likelihoods`; so does this.
Point both at the same file and the chain and the Fisher see bit-identical data,
crops and nuisance parameters — which is what the pre-2026 setup did not do (the
template chain used Cobaya-native Planck likelihoods while the Fisher used clipy,
an inconsistency flagged in ``results/notes/update.tex``).

Usage in a Cobaya YAML::

    likelihood:
      lensing_fisher.cobaya_likelihood.CandlClipyCombined:
        dataset_file: /path/to/up_planck_act_spt.yaml  # the `likelihoods:` block
        python_path: /path/to/lensing_fisher/src

Nuisance parameters are declared automatically from the likelihoods' own
fiducial values; give them priors in the Cobaya ``params:`` block as usual.
"""

import numpy as np
from cobaya.likelihood import Likelihood

from .likelihoods import build_likelihoods
from .windows import effective_ell_max

# candl spectrum keys, and the Cobaya Cl keys they map from.
_SPEC_MAP = {"tt": "TT", "te": "TE", "ee": "EE", "bb": "BB"}


class CandlClipyCombined(Likelihood):
    """Sum of candl/clipy likelihoods built from a lensing_fisher YAML config."""

    dataset_file: str = ""
    #: Optional inline ``{name: spec}`` mapping, used instead of ``dataset_file``.
    likelihoods: dict = {}
    #: Requested lensing potential lmax; ``pp``/``kk`` are only needed by
    #: reconstruction likelihoods, so this is usually harmless.
    want_lensing: bool = False
    #: Drop candl's internal nuisance priors so Cobaya's ``params:`` block owns
    #: them. Default True, matching ``candl.interface.CandlCobayaLikelihood``.
    #: Set False to keep them, which is what the Fisher driver does — then the
    #: chain and the Fisher have identical priors, and the Cobaya ``params:``
    #: block should declare these parameters with wide *flat* priors only.
    clear_internal_priors: bool = True
    #: Explicit theory lmax. Leave 0 to derive it from where the retained
    #: bandpower windows actually have support.
    theory_lmax: int = 0
    #: Window weight allowed to fall above the derived theory lmax.
    window_support_tol: float = 1.0e-4

    def initialize(self):
        from . import config as _config

        if self.likelihoods:
            cfg = {"likelihoods": dict(self.likelihoods)}
        elif self.dataset_file:
            full = _config.load_raw(self.dataset_file)
            if "likelihoods" not in full:
                raise ValueError(
                    f"{self.dataset_file} has no 'likelihoods:' block"
                )
            cfg = {"likelihoods": full["likelihoods"]}
        else:
            raise ValueError(
                "CandlClipyCombined needs either 'dataset_file' or an inline "
                "'likelihoods' mapping"
            )

        self._names, self._likes, self._nuisance_fid = build_likelihoods(cfg["likelihoods"])
        self._ell_min = min(int(lk.ell_min) for lk in self._likes)
        # Arrays handed to each likelihood must span its advertised ell_max ...
        self._ell_max = max(int(lk.ell_max) for lk in self._likes)
        # ... but CAMB only has to reach where the retained windows actually are.
        eff = {n: effective_ell_max(lk, self.window_support_tol)
               for n, lk in zip(self._names, self._likes)}
        self._driver = max(eff, key=eff.get)
        self._theory_lmax = int(self.theory_lmax) if self.theory_lmax else eff[self._driver]
        self._handle_internal_priors()
        self.log.info(
            "built %d likelihood(s) %s; ell range %d..%d; nuisance %s",
            len(self._likes), self._names, self._ell_min, self._ell_max,
            sorted(self._nuisance_fid),
        )
        self.log.info(
            "theory lmax %d, set by %s (per-likelihood: %s)",
            self._theory_lmax, self._driver,
            ", ".join(f"{n}={v}" for n, v in eff.items()),
        )
        if self._theory_lmax < self._ell_max:
            self.log.info(
                "this is below the advertised max ell_max=%d; the omitted "
                "bandpower windows carry < %g of the total weight",
                self._ell_max, self.window_support_tol,
            )

    def _handle_internal_priors(self):
        """Clear candl's internal priors, and report clipy's, to avoid double-counting.

        candl priors are a plain list and emptying it works — but only here, in
        ``initialize()``. candl compiles the log-likelihood on its first call and
        ignores later edits to the list, so this must stay ahead of any ``logp``.
        clipy bakes its priors in at construction (``all_priors=True`` in the
        dataset spec) and offers no equivalent, so they are reported instead, and
        those parameters must not also carry a prior in the Cobaya ``params:``
        block.

        This is all-or-nothing. To keep the calibration priors while dropping a
        specific one, use ``drop_priors:`` in the dataset file instead.
        """
        for name, like in zip(self._names, self._likes):
            if hasattr(like, "priors") and isinstance(getattr(like, "priors", None), list):
                if self.clear_internal_priors and like.priors:
                    par_names = sorted({p for pr in like.priors for p in pr.par_names})
                    self.log.info("%s: clearing internal priors on %s", name, par_names)
                    like.priors = []
                    like.init_priors()
                continue

            internal = getattr(like, "_prior", None)
            if internal:
                self.log.warning(
                    "%s: carries internal (clipy) priors on %s which cannot be "
                    "cleared; do not also give these a prior in the Cobaya "
                    "params block, or set all_priors: false in the dataset file",
                    name, sorted(internal),
                )

    def get_requirements(self):
        spectra = {s: self._theory_lmax for s in ("tt", "te", "ee", "bb")}
        if self.want_lensing:
            spectra["pp"] = self._theory_lmax
        return {"Cl": spectra}

    def get_can_support_params(self):
        return list(self._nuisance_fid)

    def _slice_for(self, cl, like):
        """Build a candl-shaped Dl dict spanning one likelihood's ell range.

        Zero-pads above the theory lmax, exactly as the Fisher driver does. That
        padding is only safe because ``effective_ell_max`` established the
        retained windows have no support up there.

        A fresh dict per likelihood: candl's own Cobaya wrapper mutates the
        provider's array dict in place, which is not safe when several
        likelihoods share one provider.
        """
        lo, hi = int(like.ell_min), int(like.ell_max) + 1
        n = hi - lo
        avail = len(cl["ell"])
        stop = min(hi, avail)
        n_copy = max(0, stop - lo)

        out = {"ell": np.arange(lo, hi)}
        for lower, upper in _SPEC_MAP.items():
            if lower in cl:
                col = np.zeros(n)
                col[:n_copy] = np.asarray(cl[lower][lo:stop])
                out[upper] = col
        if "pp" in cl:
            pp = np.zeros(n)
            pp[:n_copy] = np.asarray(cl["pp"][lo:stop])
            out["pp"] = pp
            out["kk"] = pp * np.pi / 2.0
        return out

    def logp(self, **params_values):
        """Sum of the component log-likelihoods, or ``-inf`` if any is not finite.

        The guard is load-bearing, not defensive tidiness. Cobaya evaluates the
        posterior with ``make_finite=True``, which is ``np.nan_to_num`` — and that
        maps NaN to **0.0**, not to a bad value. A NaN from any one likelihood
        therefore becomes a log-likelihood of zero, i.e. a perfect fit, better
        than any real point in the space. A minimizer walks straight into it and
        a chain that proposes one can never leave.

        This is not hypothetical: it sank the first UP-PAS minimization
        (2026-08-05). Two of sixteen starts ran to the corner of the prior box
        (logA at its 3.5 maximum, tau 0.140) and reported -logpost = -25.3613,
        which is exactly minus the log-prior volume — the signature of a
        likelihood contributing precisely zero. Returning -inf instead lets
        ``make_finite`` do the right thing, since it maps -inf to -1.8e308.
        """
        cl = self.provider.get_Cl(ell_factor=True, units="muK2")
        total = 0.0
        for name, like in zip(self._names, self._likes):
            pars = dict(params_values)
            pars["Dl"] = self._slice_for(cl, like)
            value = float(like.log_like(pars))
            if not np.isfinite(value):
                self._n_nonfinite = getattr(self, "_n_nonfinite", 0) + 1
                # Loud once, then quiet: a chain skirting the prior boundary can
                # hit this often, and a flood of warnings would bury it.
                report = self.log.warning if self._n_nonfinite == 1 else self.log.debug
                report(
                    "%s returned a non-finite log-likelihood (%r); rejecting this "
                    "point (occurrence %d). Parameters: %r",
                    name, value, self._n_nonfinite,
                    {k: v for k, v in sorted(params_values.items())},
                )
                return -np.inf
            total += value
        return total
