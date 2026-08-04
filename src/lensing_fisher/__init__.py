"""Effective bandpower windows for the CMB 2pt lensing measurement.

A Fisher analysis over binned perturbations to Cl_pp says which angular scales
the primary CMB actually responds to, giving an effective L with horizontal
error bars that can be plotted against reconstruction bandpowers.
"""

from .fisher import FisherMatrix
from .local_lens import BinnedLensingTheory

__all__ = ["FisherMatrix", "BinnedLensingTheory"]
__version__ = "0.2.0"
