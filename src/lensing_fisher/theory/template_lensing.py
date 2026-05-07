"""
Template Lensing CAMB Theory Class

A Cobaya theory class that inherits from CAMB and modifies lensing to use
a template CL_pp scaled by A_template parameter, instead of self-consistent lensing.
"""

import os
import pickle
import numpy as np
from scipy.interpolate import splrep, BSpline
from typing import NamedTuple, Callable, Optional

from cobaya.theories.camb import CAMB
from cobaya.log import LoggedError


# Collector NamedTuple matching cobaya's definition
class Collector(NamedTuple):
    method: Callable
    args: list = []
    kwargs: dict = {}
    z_pool: Optional[object] = None
    post: Optional[Callable] = None


class TemplateLensingCAMB(CAMB):
    """
    CAMB theory class with template-based lensing.
    
    This class inherits from CAMB and replaces the self-consistent lensing computation
    with a template-based approach. A template CL_pp is loaded from a pickle file and
    scaled by the A_template parameter to lens the primary CMB spectra.
    
    Theory input options:
    - clpp_template_file: Path to pickle file containing template CL_pp (dict with 'L' and 'CL_pp_fid' arrays)
    
    Input parameters:
    - A_template: Scaling factor for the template CL_pp (default: 1.0)
    """
    
    clpp_template_file: str = ""
    
    def initialize(self):
        """Initialize the class and load template CL_pp file."""
        # Initialize parent CAMB class first
        super().initialize()
        
        # Initialize A_template storage
        self._current_A_template = 1.0
        
        # Load template CL_pp file
        if not self.clpp_template_file:
            raise LoggedError(
                self.log,
                "clpp_template_file must be provided for TemplateLensingCAMB"
            )
        
        if not os.path.exists(self.clpp_template_file):
            raise LoggedError(
                self.log,
                f"Template file not found: {self.clpp_template_file}"
            )
        
        try:
            with open(self.clpp_template_file, 'rb') as f:
                template_data = pickle.load(f)
            
            if isinstance(template_data, dict):
                self.L_template = template_data['L']
                self.CL_pp_fid = template_data['CL_pp_fid']
            else:
                raise ValueError("Template file must contain dict with 'L' and 'CL_pp_fid'")
            
            self.L_template = np.asarray(self.L_template)
            self.CL_pp_fid = np.asarray(self.CL_pp_fid)
            
            if len(self.L_template) != len(self.CL_pp_fid):
                raise LoggedError(
                    self.log,
                    f"Template arrays L and CL_pp_fid must have same length: "
                    f"{len(self.L_template)} vs {len(self.CL_pp_fid)}"
                )
            
            if len(self.L_template) == 0:
                raise LoggedError(
                    self.log,
                    "Template arrays cannot be empty"
                )
            
        except Exception as e:
            raise LoggedError(
                self.log,
                f"Error loading template file {self.clpp_template_file}: {e}"
            )
    
    def initialize_with_params(self):
        """Initialize with parameters - call parent and ensure A_template is recognized."""
        # Call parent initialization
        super().initialize_with_params()
        
        # A_template will be passed via params_values_dict in calculate()
        # No special registration needed - it's a custom parameter
    
    def get_can_support_params(self):
        """Return parameters that this theory can support, including A_template."""
        # Get parent's supported params (power_params, nonlin_params, sigma8)
        params = list(super().get_can_support_params())
        # Add our custom parameter
        params.append("A_template")
        return params
    
    def _interpolate_template_to_camb_ells(self, camb_ells):
        """
        Log-log template interpolation to CAMB ell range.
        
        Uses log-log B-spline interpolation similar to lensing_sensitivity_fisher.py
        
        Parameters
        ----------
        camb_ells : array-like
            Ell values from CAMB (typically np.arange(max_l+1))
        
        Returns
        -------
        clpp_interpolated : ndarray
            Interpolated CL_pp values at camb_ells
        """
        # Convert to numpy arrays
        camb_ells = np.asarray(camb_ells)
        L_template = self.L_template
        CL_pp_fid = self.CL_pp_fid
        
        # Filter out zero/negative values for log interpolation
        valid_mask = (L_template > 0) & (CL_pp_fid > 0)
        if not np.all(valid_mask):
            self.log.warning(
                "Some template values are non-positive. Using only valid values for interpolation."
            )
            L_template = L_template[valid_mask]
            CL_pp_fid = CL_pp_fid[valid_mask]
        
        if len(L_template) < 2:
            raise LoggedError(
                self.log,
                "Need at least 2 valid template points for interpolation"
            )
        
        # Log-log interpolation using B-spline
        log_L = np.log10(L_template)
        log_CL = np.log10(CL_pp_fid)
        
        # Create B-spline representation
        tck = splrep(log_L, log_CL, k=min(3, len(L_template)-1), s=0)
        
        # Extract knots and coefficients
        knots, coeffs, degree = tck
        
        # Create BSpline object
        bspline = BSpline(knots, coeffs, degree)
        
        # Interpolate to CAMB ell range
        log_ells = np.log10(np.maximum(camb_ells, 1.0))  # Avoid log(0)
        log_result = bspline(log_ells)
        clpp_interpolated = 10**log_result
        
        # Set values for ell=0 to zero (CAMB convention)
        clpp_interpolated[camb_ells == 0] = 0.0
        
        return clpp_interpolated
    
    def must_provide(self, **requirements):
        """
        Override must_provide to replace collectors for Cl and lensed_scal_Cl
        with our custom template-based lensing method.
        """
        # Call parent must_provide first to set up all collectors normally
        result = super().must_provide(**requirements)
        
        # Now replace collectors for lensed Cl products with our custom method
        for k in self._must_provide.keys():
            if k == "Cl" or k == "lensed_scal_Cl":
                # Get the existing collector's kwargs to preserve spectra list
                existing_kwargs = {}
                if k in self.collectors and self.collectors[k] is not None:
                    existing_kwargs = self.collectors[k].kwargs.copy()
                
                # Replace with our custom collector method
                self.collectors[k] = Collector(
                    method=self._get_template_lensed_cmb_spectra,
                    kwargs=existing_kwargs
                )
        
        return result
    
    def _get_template_lensed_cmb_spectra(self, results, spectra=None, raw_cl=False):
        """
        Custom collector method that replaces CAMBdata.get_cmb_power_spectra.
        
        Uses get_lensed_cls_with_spectrum with the template Cl_pp scaled by A_template
        to produce lensed CMB power spectra. Returns the self-consistent lensing
        potential from get_lens_potential_cls for the pp spectrum.
        
        Parameters
        ----------
        results : CAMBdata
            CAMB results object
        spectra : list, optional
            List of spectra to return (e.g., ["total", "lens_potential"])
        raw_cl : bool
            If False, return ell(ell+1)Cl/2pi; if True, return raw Cl
        
        Returns
        -------
        dict
            Dictionary with requested spectra, matching format of get_cmb_power_spectra
        """
        if spectra is None:
            spectra = ["total"]
        
        # Get lmax from CAMB params
        lmax = results.Params.max_l
        
        # Prepare template Cl_pp scaled by A_template
        camb_ells = np.arange(lmax + 1)
        clpp_template_interp = self._interpolate_template_to_camb_ells(camb_ells)
        clpp_for_lensing = self._current_A_template * clpp_template_interp
        
        # Get lensed CMB spectra using template Cl_pp
        # get_lensed_cls_with_spectrum returns array (lmax+1, 4) with columns [TT, EE, BB, TE]
        # When raw_cl=False, returns ell(ell+1)Cl/2pi (which is what we want)
        lensed_cls = results.get_lensed_cls_with_spectrum(
            clpp_for_lensing, 
            lmax=lmax,
            raw_cl=raw_cl
        )
        
        # Build result dict matching get_cmb_power_spectra format
        result = {}
        
        for spec in spectra:
            if spec == "total":
                # "total" key gets the lensed CMB Cls
                result["total"] = lensed_cls.copy()
            elif spec == "lensed_scalar":
                # "lensed_scalar" also gets the lensed CMB Cls
                result["lensed_scalar"] = lensed_cls.copy()
            elif spec == "lens_potential":
                # For lens_potential, return the SELF-CONSISTENT spectrum from CAMB
                # (not the template we used for lensing)
                # get_lens_potential_cls returns array (lmax+1, 3) with columns [pp, pT, pE]
                # When raw_cl=False, returns [L(L+1)]^2 Cl_pp/2pi for pp column
                lens_pot = results.get_lens_potential_cls(lmax=lmax, raw_cl=raw_cl)
                result["lens_potential"] = lens_pot.copy()
            elif spec == "unlensed_scalar":
                # For unlensed, use standard CAMB method
                unlensed = results.get_unlensed_scalar_cls(lmax=lmax, raw_cl=raw_cl)
                result["unlensed_scalar"] = unlensed.copy()
            elif spec == "unlensed_total":
                # For unlensed total, use standard CAMB method
                unlensed = results.get_unlensed_total_cls(lmax=lmax, raw_cl=raw_cl)
                result["unlensed_total"] = unlensed.copy()
            elif spec == "tensor":
                # For tensor, use standard CAMB method
                tensor = results.get_tensor_cls(lmax=lmax, raw_cl=raw_cl)
                result["tensor"] = tensor.copy()
        
        return result
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Calculate theory predictions using template-based lensing.
        
        Stores A_template for access by the custom collector method, then
        delegates to parent calculate().
        """
        # Store A_template for access by the custom collector method
        self._current_A_template = params_values_dict.get('A_template', 1.0)
        
        # Call parent calculate - this will invoke our custom collector
        super().calculate(state, want_derived=want_derived, **params_values_dict)
