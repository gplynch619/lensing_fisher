"""CAMB-based pars→theory-spectra builder for the local-bump Cl_pp parametrization.

Cl_pp is parametrized as a sum of smooth indicator (sigmoid) bumps centered at
``L_centers`` with width ``width`` and steepness ``steepness``. The bump
amplitudes are read from the input ``pars`` dict by name (``clpp_1``, ``clpp_2``,
...) and multiplied by ``amp_unit`` so step sizes can be O(1).
"""

from typing import Callable

import numpy as np


def _smooth_indicator(L_center: float, amp: float, width: float, k: float = 1.0):
    """Sigmoid-difference indicator: amp * (H(L - Lc + w/2) - H(L - Lc - w/2))."""
    H = lambda L: 1.0 / (1.0 + np.exp(-2.0 * k * L))
    return lambda L: amp * (H(L - L_center + width / 2.0) - H(L - L_center - width / 2.0))


def build_local_lens_theory(
    camb_pars,
    L_centers,
    width: float,
    steepness: float = 1.0,
    amp_unit: float = 1e-7,
) -> Callable[[dict, int, int], dict]:
    """Build a ``pars_to_theory_specs(pars, ell_high_cut, ell_low_cut=2)`` closure.

    Parameters
    ----------
    camb_pars
        Pre-configured ``camb.model.CAMBparams`` (accuracy / lmax / matter power
        already set). Cosmology values are overwritten on each call from ``pars``.
    L_centers
        Iterable of L values where bumps are centered.
    width
        Width of each bump (same units as L).
    steepness
        Sigmoid steepness ``k`` controlling how sharp the indicator edges are.
    amp_unit
        Scale applied to the bump amplitudes. Default ``1e-7`` matches the
        existing convention so YAML step sizes for ``clpp_*`` are O(1).
    """

    import camb
    import jax.numpy as jnp

    def _jax_set(arr, ix, el):
        return arr.at[ix].set(el)

    L_centers = np.asarray(L_centers)
    CAMB_ix = {"TT": 0, "EE": 1, "BB": 2, "TE": 3}

    def pars_to_theory_specs(pars: dict, ell_high_cut: int, ell_low_cut: int = 2) -> dict:
        camb_pars.set_cosmology(
            H0=pars["H0"],
            ombh2=pars["ombh2"],
            omch2=pars["omch2"],
            mnu=pars.get("mnu", 0.06),
            omk=pars.get("omk", 0.0),
            tau=pars["tau"],
        )
        camb_pars.InitPower.set_params(
            As=np.exp(pars["logA"]) * 1e-10, ns=pars["ns"], r=0
        )

        camb_ells = jnp.arange(camb_pars.max_l + 1)

        amps = np.array(
            [v for name, v in pars.items() if name.startswith("clpp")]
        ) * amp_unit

        perturbation = jnp.zeros(len(camb_ells))
        for i, amp in enumerate(amps):
            perturbation = perturbation + _smooth_indicator(
                L_centers[i], amp, width, k=steepness
            )(camb_ells)

        clpp_for_lensing = perturbation

        results = camb.get_results(camb_pars)
        powers = results.get_lensed_cls_with_spectrum(clpp_for_lensing, CMB_unit="muK")
        camb_ells = jnp.arange(powers.shape[0])

        N_ell = ell_high_cut - ell_low_cut + 1
        theory_start_ix = max(int(camb_ells[0]), ell_low_cut) - int(camb_ells[0])
        theory_stop_ix = min(int(camb_ells[-1]), ell_high_cut) + 1 - int(camb_ells[0])
        like_start_ix = max(int(camb_ells[0]), ell_low_cut) - ell_low_cut
        like_stop_ix = min(int(camb_ells[-1]), ell_high_cut) + 1 - ell_low_cut

        Dls = {
            "ell": np.arange(ell_low_cut, ell_high_cut + 1),
            "pp": clpp_for_lensing[theory_start_ix:theory_stop_ix],
            "kk": clpp_for_lensing[theory_start_ix:theory_stop_ix] * jnp.pi / 2.0,
        }
        for ky, ix in CAMB_ix.items():
            Dls[ky] = jnp.zeros(N_ell)
            Dls[ky] = _jax_set(
                Dls[ky],
                np.arange(like_start_ix, like_stop_ix),
                powers[theory_start_ix:theory_stop_ix, ix],
            )
        return Dls

    return pars_to_theory_specs
