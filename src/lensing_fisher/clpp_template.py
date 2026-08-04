"""Loading and interpolation of a frozen Cl_pp template.

Everything in this package uses the CAMB convention for the lensing potential
power spectrum, ``[L(L+1)]^2 C_L^phiphi / 2pi``. That is what
``CAMBdata.get_lensed_cls_with_spectrum`` expects, what
``CAMBdata.get_lens_potential_cls(...)[:, 0]`` returns, and what the template
pickles written by ``template_test_plots.ipynb`` contain.

The pickle format is a dict with keys ``L`` and ``CL_pp_fid``, matching what
``lensing_fisher.template_lensing.TemplateLensingCAMB`` already consumes,
so the Fisher driver and the Cobaya template chain can be pointed at the same
file.
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, splrep


def load_template(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``{L, CL_pp_fid}`` template pickle.

    Returns ``(L, CL_pp)`` with non-positive entries dropped, so the result is
    safe for log-log interpolation.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "L" not in data or "CL_pp_fid" not in data:
        raise ValueError(
            f"template {Path(path)!s} must be a dict with keys 'L' and 'CL_pp_fid'"
        )

    L = np.asarray(data["L"], dtype=float)
    CL = np.asarray(data["CL_pp_fid"], dtype=float)
    if L.shape != CL.shape:
        raise ValueError(
            f"template {Path(path)!s}: L and CL_pp_fid have shapes "
            f"{L.shape} and {CL.shape}"
        )

    good = (L > 0) & (CL > 0)
    if good.sum() < 4:
        raise ValueError(
            f"template {Path(path)!s} has fewer than 4 positive samples; "
            f"cannot interpolate"
        )
    return L[good], CL[good]


def interpolate_to_ells(L, CL, ells) -> np.ndarray:
    """Log-log cubic B-spline interpolation of ``CL`` onto integer ``ells``.

    Mirrors ``TemplateLensingCAMB._interpolate_template_to_camb_ells`` so the
    Fisher driver and the template chain lens with the same numbers, with one
    difference: outside ``[L.min(), L.max()]`` this continues as a power law
    fixed by the end-point log-log slope rather than letting the cubic run free.

    CAMB needs C_fid out to ``max_l = lmax + lens_margin``, which is typically
    ~1000 beyond where a template pickle stops, and an unconstrained cubic
    extrapolated that far can swing by orders of magnitude or go negative.
    C^phiphi is close to a power law in the tail, so this is both safer and more
    accurate. ``ell = 0`` is set to zero (CAMB convention).
    """
    ells = np.asarray(ells)
    logL, logCL = np.log10(L), np.log10(CL)
    spline = BSpline(*splrep(logL, logCL, k=min(3, len(L) - 1), s=0))

    x = np.log10(np.maximum(ells, 1.0))
    x_clipped = np.clip(x, logL[0], logL[-1])
    out_log = spline(x_clipped)

    # Power-law continuation beyond either end, using the local end slope.
    lo_slope = (logCL[1] - logCL[0]) / (logL[1] - logL[0])
    hi_slope = (logCL[-1] - logCL[-2]) / (logL[-1] - logL[-2])
    below, above = x < logL[0], x > logL[-1]
    out_log[below] += lo_slope * (x[below] - logL[0])
    out_log[above] += hi_slope * (x[above] - logL[-1])

    out = np.asarray(10.0**out_log, dtype=float)
    out[ells == 0] = 0.0
    return out


def clpp_fid_from_camb(camb_pars, lmax: int) -> np.ndarray:
    """Self-consistent ``[L(L+1)]^2 C_L^pp/2pi`` at ``camb_pars``, length ``lmax+1``.

    Convenience for generating a frozen template from a fiducial cosmology when
    no best-fit template pickle exists yet (the early binning iterations). The
    array is frozen by the caller at setup; it is *not* recomputed as the Fisher
    varies cosmological parameters.
    """
    import camb

    results = camb.get_results(camb_pars)
    cl = results.get_lens_potential_cls(lmax=lmax)[:, 0]
    if len(cl) < lmax + 1:
        cl = np.concatenate([cl, np.zeros(lmax + 1 - len(cl))])
    return np.asarray(cl[: lmax + 1], dtype=float)
