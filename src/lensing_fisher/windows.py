"""Bandpower window diagnostics: how high in ell does a likelihood really reach?

Theory spectra are zero-padded above the lensed-CMB lmax. If a retained
bandpower window has support up there, those bandpowers get compared against
zero theory and the result is quietly wrong — no error, no NaN, just a bad
matrix. This module exists to make that impossible to do by accident.

It matters because candl's ``ell_max`` does not respond to ``data_selection``: a
likelihood cropped to ell<=1000 still advertises 8501 (ACT DR6) or 4095
(SPT-3G D1). Believing the advertised value wastes CAMB time; ignoring it
silently truncates real data. The window functions settle it either way.
"""

from typing import Optional

import numpy as np

from .mpi_log import info


def window_amplitude(like):
    """``(ells, amplitude)`` summed over the likelihood's bandpower windows.

    ``amplitude[k]`` is the total absolute window weight at ``ells[k]``, over all
    spectra and bandpowers retained after any data selection.

    Returns ``None`` when windows are not available. Every clipy likelihood is in
    that category: it raises ``NotImplementedError`` for ``window_functions``,
    and its ``lmax`` array keeps pre-crop values, so nothing here can help it.
    """
    try:
        blocks = like.window_functions
        ell_min, ell_max = int(like.ell_min), int(like.ell_max)
    except (AttributeError, NotImplementedError):
        return None

    if not isinstance(blocks, (list, tuple)):
        blocks = [blocks]

    ells = np.arange(ell_min, ell_max + 1)
    amplitude = np.zeros(ells.size)
    for block in blocks:
        W = np.abs(np.asarray(block, dtype=float))
        if W.ndim != 2:
            continue
        if W.shape[0] != ells.size and W.shape[1] == ells.size:
            W = W.T                       # orient so axis 0 runs over ell
        if W.shape[0] != ells.size:
            return None
        amplitude += W.sum(axis=1)

    return (ells, amplitude) if amplitude.sum() > 0 else None


def support_above(like, lmax: int) -> Optional[float]:
    """Fraction of window weight sitting above ``lmax``; None if unknowable."""
    got = window_amplitude(like)
    if got is None:
        return None
    ells, amplitude = got
    return float(amplitude[ells > lmax].sum() / amplitude.sum())


def effective_ell_max(like, tol: float = 1e-4) -> int:
    """Smallest lmax retaining all but ``tol`` of the window weight.

    Falls back to the advertised ``ell_max`` when windows are unavailable. That
    fallback is deliberately conservative: guessing a smaller value from bin
    centres plus an assumed bandpower width would risk feeding zero theory to
    real data, which is exactly the failure this module prevents.
    """
    got = window_amplitude(like)
    if got is None:
        return int(like.ell_max)
    ells, amplitude = got
    cumulative = np.cumsum(amplitude) / amplitude.sum()
    idx = int(np.searchsorted(cumulative, 1.0 - tol))
    return int(ells[min(idx, ells.size - 1)])


def check_theory_lmax(names, likelihoods, theory_lmax: int, tol: float = 1e-3) -> None:
    """Raise if any likelihood needs theory above ``theory_lmax``.

    Raising rather than warning is the point: the failure leaves no trace in the
    output. ``data/full_fishers/lensing_sensitivity_reference.pkl`` was computed
    with theory to lmax=2500 against ACT DR6 bandpowers reaching ell~6126, and
    nothing about the resulting file says so.
    """
    offenders = []
    for name, like in zip(names, likelihoods):
        fraction = support_above(like, theory_lmax)
        if fraction is None:
            info(f"{name}: no windows exposed, lmax check skipped")
            continue
        verdict = "ok" if fraction <= tol else "TOO HIGH"
        info(f"{name}: {fraction:.2e} of window weight above "
             f"lmax={theory_lmax}  [{verdict}]")
        if fraction > tol:
            offenders.append((name, fraction))

    if offenders:
        worst = max(f for _, f in offenders)
        detail = ", ".join(f"{n} ({f:.2e})" for n, f in offenders)
        raise ValueError(
            f"theory lmax={theory_lmax} is too low: {detail} of bandpower window "
            f"weight lies above it and would be compared against zero theory. "
            f"Raise camb.set_for_lmax.lmax, tighten the likelihood ell_cuts, or "
            f"set camb.window_support_tol above {worst:.1e} if you have decided "
            f"this is acceptable."
        )
