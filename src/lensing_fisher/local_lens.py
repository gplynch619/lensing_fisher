"""The binned Cl_pp parametrization: parameters -> lensed CMB spectra.

The lensing potential used to lens the primary CMB is

    C_L^phiphi(theta, q) = C_fid(L) * (1 + sum_i q_i B_i(L))

with ``B_i`` a smooth top-hat over bin ``i`` and ``q_i`` a *fractional*
perturbation, so the fiducial is ``q_i = 0``.

``C_fid`` is frozen at setup and does not track the cosmology being varied.
Cosmology enters only through the unlensed spectra inside the CAMB results
object. This is what makes the Fisher matrix the Fisher matrix *of the
A_template model*: :class:`~lensing_fisher.template_lensing.TemplateLensingCAMB`
lenses with ``A_template * <frozen template>`` in exactly the same way, so a
uniform ``q_i = a`` is precisely ``A_template = 1 + a``. It also keeps the
Jacobian ``r_i = C_tem(L_i)/C_fid(L_i)`` well defined, since there is a single
C_fid to take the ratio against.
"""

from collections import OrderedDict

import numpy as np
from scipy.special import expit

#: Parameters that change the CAMB computation. Everything else — nuisance
#: parameters, and every clpp_* bin amplitude — leaves a results object valid,
#: which is what makes the cache in BinnedLensingTheory worthwhile.
COSMO_KEYS = ("H0", "ombh2", "omch2", "tau", "logA", "ns", "mnu", "omk")

#: Column order of camb's get_lensed_cls_with_spectrum output.
CAMB_IX = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}


def smooth_indicator(L_center, width, steepness):
    """Smooth top-hat: sigmoid rise at ``L_center - width/2``, fall at ``+width/2``.

    ``expit`` rather than ``1/(1+exp(-x))`` written out: the naive form overflows
    for L far below the bin, which happens routinely now that bins start at L=2
    and the basis is evaluated over the full CAMB ell range.
    """
    def H(x):
        return expit(2.0 * steepness * np.asarray(x, dtype=float))

    return lambda L: H(L - L_center + width / 2.0) - H(L - L_center - width / 2.0)


def basis_matrix(bin_edges, ells, steepness) -> np.ndarray:
    """``(n_ell, n_bins)`` matrix of bin basis functions, ``B[L, i] = B_i(L)``.

    ``bin_edges`` has length ``n_bins + 1`` and describes contiguous bins of
    possibly differing width, so bin ``i`` spans ``[edges[i], edges[i+1])``.

    The outermost basis functions are one-sided: the first is 1 everywhere below
    its right edge, the last is 1 everywhere above its left edge. A plain sigmoid
    top-hat evaluates to 0.5 exactly *at* the outer edges and decays past them,
    which would leave the endpoints only half-scaled by a uniform q and break the
    correspondence with A_template. One-sided ends make the basis sum to 1
    everywhere.
    """
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 3:
        raise ValueError(f"need at least 2 bins (3 edges); got shape {edges.shape}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")

    ells = np.asarray(ells, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    B = np.empty((ells.size, centers.size), dtype=float)
    for i, (center, width) in enumerate(zip(centers, widths)):
        B[:, i] = smooth_indicator(center, width, steepness)(ells)

    B[:, 0] = 1.0 - expit(2.0 * steepness * (ells - edges[1]))
    B[:, -1] = expit(2.0 * steepness * (ells - edges[-2]))
    return B


def pack_dls(powers, clpp, ell_low: int, ell_high: int) -> dict:
    """Assemble the Dl dict that a candl/clipy likelihood expects.

    Likelihoods request theory over their own ``[ell_min, ell_max]``, which can
    reach past the lensed-CMB lmax — candl's ``ell_max`` is not reduced by
    ``data_selection``, so a likelihood cropped to L<=1000 still asks for 8501.
    Multipoles above what CAMB returned are zero-padded. That is only safe
    because :func:`~lensing_fisher.likelihoods.windows.check_theory_lmax` has
    already verified no retained bandpower window has support up there.
    """
    n_ell = ell_high - ell_low + 1
    camb_lmax = powers.shape[0] - 1

    copy_stop = min(camb_lmax, ell_high) + 1     # exclusive, in ell units
    n_copy = max(0, copy_stop - ell_low)

    def padded(source_2d=None, source_1d=None, column=None):
        out = np.zeros(n_ell)
        src = source_1d if source_1d is not None else source_2d[:, column]
        out[:n_copy] = src[ell_low:copy_stop]
        return out

    dls = {"ell": np.arange(ell_low, ell_high + 1)}
    for name, column in CAMB_IX.items():
        dls[name] = padded(source_2d=powers, column=column)
    pp = padded(source_1d=clpp)
    dls["pp"] = pp
    dls["kk"] = pp * np.pi / 2.0
    return dls


class BinnedLensingTheory:
    """Callable mapping a parameter dict to CMB Dls, for one fixed bin grid.

    Call signature is ``theory(pars, ell_high_cut, ell_low_cut=2)``, matching what
    ``candl.tools.get_params_to_logl_func`` expects.

    Parameters
    ----------
    camb_pars
        Pre-configured ``camb.model.CAMBparams``; accuracy, lmax and matter power
        are taken as given, cosmology is overwritten per call.
    bin_edges
        Contiguous bin edges, length ``n_bins + 1``.
    clpp_fid
        Frozen ``[L(L+1)]^2 C_L^phiphi/2pi``, length ``camb_pars.max_l + 1``.
    steepness
        Sigmoid steepness of the bin edges.
    cache_size
        CAMB results objects to retain. Elements of the Fisher matrix are ordered
        so that runs of them share a cosmology (see ``FisherMatrix._tasks``), so
        a handful of entries gives a near-perfect hit rate.
    cosmology_kwargs
        Everything ``camb.set_cosmology`` needs *besides* the parameters being
        varied — the fixed cosmology (``mnu``, ``omk``) and the config's
        ``camb.set_cosmology`` block (``bbn_predictor``, ``nnu``,
        ``num_massive_neutrinos``).

        These must be re-supplied on every call. ``set_cosmology`` recomputes
        ``YHe`` from the BBN predictor each time it runs, so omitting
        ``bbn_predictor`` here does not leave the value set by
        :func:`~lensing_fisher.driver.build_camb_params` in place — it silently
        reverts to CAMB's default (PRIMAT in 1.6.4, against the PArthENoPE the
        Cobaya chains ask for) on the first cache miss and stays there. That is a
        0.19% shift in ``YHe``, landing in the damping tail where ACT and SPT
        carry their weight, and it would make this Fisher describe a different
        model from the chains.
    """

    def __init__(self, camb_pars, bin_edges, clpp_fid, steepness=2.0, cache_size=4,
                 cosmology_kwargs=None):
        import camb

        self._camb = camb
        self.camb_pars = camb_pars
        self.cosmology_kwargs = dict(cosmology_kwargs or {})
        self.bin_edges = np.asarray(bin_edges, dtype=float)

        n_ell = camb_pars.max_l + 1
        self.clpp_fid = np.asarray(clpp_fid, dtype=float)
        if self.clpp_fid.shape != (n_ell,):
            raise ValueError(
                f"clpp_fid must have length camb_pars.max_l + 1 = {n_ell}; "
                f"got {self.clpp_fid.shape}"
            )

        self.basis = basis_matrix(self.bin_edges, np.arange(n_ell), steepness)
        self.n_bins = self.basis.shape[1]

        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self.cache_stats = {"hits": 0, "misses": 0}

    # ------------------------------------------------------------------

    def _camb_results(self, pars: dict):
        """CAMB results at this cosmology, reusing a cached object when possible."""
        key = tuple(round(float(pars[k]), 14) for k in COSMO_KEYS if k in pars)
        if key in self._cache:
            self.cache_stats["hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.cache_stats["misses"] += 1
        # Fixed values first, then anything the Fisher is actually varying, so a
        # parameter promoted to the free vector (mnu, say) overrides its fixed
        # counterpart rather than being silently pinned.
        kwargs = dict(self.cosmology_kwargs)
        kwargs.update({k: pars[k] for k in ("mnu", "omk") if k in pars})
        self.camb_pars.set_cosmology(
            H0=pars["H0"],
            ombh2=pars["ombh2"],
            omch2=pars["omch2"],
            tau=pars["tau"],
            **kwargs,
        )
        self.camb_pars.InitPower.set_params(As=np.exp(pars["logA"]) * 1e-10, ns=pars["ns"], r=0)

        results = self._camb.get_results(self.camb_pars)
        self._cache[key] = results
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return results

    def bin_amplitudes(self, pars: dict) -> np.ndarray:
        """Pull ``clpp_1 .. clpp_n`` out of the parameter dict, in bin order."""
        return np.array([pars[f"clpp_{i + 1}"] for i in range(self.n_bins)], dtype=float)

    def clpp(self, q) -> np.ndarray:
        """``C_fid * (1 + B q)`` — the lensing spectrum for bin amplitudes ``q``."""
        return self.clpp_fid * (1.0 + self.basis @ np.asarray(q, dtype=float))

    def __call__(self, pars: dict, ell_high_cut: int, ell_low_cut: int = 2) -> dict:
        results = self._camb_results(pars)
        clpp = self.clpp(self.bin_amplitudes(pars))
        powers = results.get_lensed_cls_with_spectrum(clpp, CMB_unit="muK")
        return pack_dls(powers, clpp, ell_low_cut, ell_high_cut)
