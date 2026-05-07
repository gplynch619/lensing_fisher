import numpy as np
from typing import Callable, Optional, Union
import warnings
import pickle
import os


class FisherMatrix:
    """
    Parameters
    ----------
    pars_to_loglike : callable
        Function that takes a dictionary of cosmological parameters as input and returns
        log-likelihood value. Should have signature: func(params_dict) -> float
    fiducial_params : dict
        Dictionary with fiducial parameter values around which to compute the Fisher matrix.
        Keys are parameter names, values are fiducial parameter values.
    default_step_size : float, optional
        Default step size for numerical derivatives (default: 1e-2)
    min_step_size : float, optional
        Minimum allowed step size (default: 1e-2)
    max_step_size : float, optional
        Maximum allowed step size (default: 5e-2)
    use_central_differences : bool, list, or array, optional
        Whether to use central finite differences for each parameter. If bool, applies to all
        parameters. If list/array, must have length equal to number of parameters.
        Central differences require negative step sizes; use False for parameters with
        fiducial=0 that must be non-negative. (default: True, all central)
    """
    
    def __init__(self, 
                 pars_to_loglike: Callable[[dict], float],
                 fiducial_params: dict,
                 default_step_size: float = 1e-2,
                 min_step_size: float = 1e-2,
                 max_step_size: float = 5e-2,
                 use_central_differences: Optional[Union[bool, list, np.ndarray]] = True,
                 comm=None):
        
        self.pars_to_loglike = pars_to_loglike
        
        # Extract parameter names and values from dictionary
        self.param_names = list(fiducial_params.keys())
        self.fiducial_params = np.array([fiducial_params[name] for name in self.param_names], dtype=float)
        self.n_params = len(self.param_names)
        
        # Store the fiducial dictionary for easy access
        self.fiducial_dict = fiducial_params.copy()
        
        # Step size parameters
        self.default_step_size = default_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        
        # Convert use_central_differences to a numpy array
        if use_central_differences is None:
            use_central_differences = True
        
        if isinstance(use_central_differences, bool):
            # Single boolean: apply to all parameters
            self.use_central_diff = np.full(self.n_params, use_central_differences, dtype=bool)
        elif isinstance(use_central_differences, (list, np.ndarray)):
            # List/array: one flag per parameter
            use_central_differences = np.asarray(use_central_differences, dtype=bool)
            if len(use_central_differences) != self.n_params:
                raise ValueError(
                    f"use_central_differences must have length {self.n_params}, "
                    f"got {len(use_central_differences)}"
                )
            self.use_central_diff = use_central_differences
        else:
            raise TypeError(
                f"use_central_differences must be bool, list, or array, "
                f"got {type(use_central_differences)}"
            )
        
        # Cache for computed values
        self._fisher_matrix = None
        self._step_sizes_used = None
        self._stability_metrics = None

        self.comm = comm
        if comm is not None:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.rank = 0
            self.size = 1
        self.use_mpi = self.size > 1

        
    def compute_fisher_matrix(self, 
                            step_size: Optional[Union[float, np.ndarray]] = None,
                            relative: bool = True,
                            adaptive: bool = True,
                            stability_threshold: float = 0.01) -> np.ndarray:
        """
        Compute the Fisher matrix using 4th-order numerical differentiation.
        
        Parameters
        ----------
        step_size : float or array_like, optional
            Step size(s) for numerical differentiation. If None, uses adaptive
            step size selection. If float, uses same step for all parameters.
            If array, should have length equal to number of parameters.
        relative : bool, optional
            Whether to interpret step_size as relative to fiducial parameter values.
            If True, step_size is a fraction (e.g., 0.01 = 1% of fiducial value).
            If False, step_size is absolute (default: True)
        adaptive : bool, optional
            Whether to use adaptive step size selection (default: True)
        stability_threshold : float, optional
            Threshold for stability assessment (default: 0.01)
            
        Returns
        -------
        fisher_matrix : ndarray
            The computed Fisher matrix (negative Hessian)
            
        Raises
        ------
        ValueError
            If step_size has wrong dimensions
        RuntimeError
            If likelihood function evaluation fails
        """
        
        # Determine step sizes
        if step_size is None:
            if adaptive:
                print("Using adaptive step size selection...")
                # Get optimal step sizes (relative or absolute depending on 'relative' parameter)
                optimal_steps = self._adaptive_step_selection(stability_threshold, relative)
                # Convert to absolute step sizes for computation
                if relative:
                    step_sizes = self._convert_to_absolute_steps(optimal_steps)
                else:
                    step_sizes = optimal_steps
            else:
                # Use default step sizes
                if relative:
                    default_relative = np.full(self.n_params, self.default_step_size)
                    step_sizes = self._convert_to_absolute_steps(default_relative)
                else:
                    step_sizes = np.full(self.n_params, self.default_step_size)
        else:
            # User provided step sizes
            step_sizes = np.asarray(step_size, dtype=float)
            if step_sizes.ndim == 0:  # scalar
                step_sizes = np.full(self.n_params, step_sizes)
            elif len(step_sizes) != self.n_params:
                raise ValueError(f"Step size array must have length {self.n_params}")
            # Convert to absolute step sizes if relative
            if relative:
                step_sizes = self._convert_to_absolute_steps(step_sizes)
        
        # Validate step sizes (now all absolute)
        #step_sizes = np.clip(step_sizes, self.min_step_size, self.max_step_size)
        if self.rank == 0:
            print(f"Step sizes: {step_sizes}")
        # Compute Fisher matrix
        fisher_matrix = np.zeros((self.n_params, self.n_params))
        
        if self.use_mpi:
            self._fisher_matrix = self._compute_fisher_matrix_mpi(fisher_matrix, step_sizes)
        else:
            self._fisher_matrix = self._compute_fisher_matrix_serial(fisher_matrix, step_sizes)
            
        # Store results
        self._fisher_matrix = fisher_matrix
        self._step_sizes_used = step_sizes.copy()
        
        # Compute stability metrics
        #self._stability_metrics = self._assess_stability(step_sizes)
    
        return fisher_matrix
    
    def _compute_fisher_matrix_serial(self, fisher_matrix, step_sizes):
        try:
            # Compute diagonal elements (second derivatives)
            for i in range(self.n_params):
                if self.rank == 0:
                    print(f"Computing ∂²/∂θ{i}θ{i} with step size {step_sizes[i]}")
                
                fisher_matrix[i, i] = self._compute_second_derivative(
                    i, step_sizes[i]
                )
            
            # Compute off-diagonal elements (mixed second derivatives)
            for i in range(self.n_params):
                for j in range(i + 1, self.n_params):
                    print(f"Computing ∂²/∂θ{i}θ{j} with step size ({step_sizes[i]}, {step_sizes[j]})")
                    mixed_deriv = self._compute_mixed_second_derivative(
                        i, j, step_sizes[i], step_sizes[j]
                    )
                    fisher_matrix[i, j] = mixed_deriv
                    fisher_matrix[j, i] = mixed_deriv  # Symmetric matrix
                    
        except Exception as e:
            raise RuntimeError(f"Failed to compute Fisher matrix: {str(e)}") from e
    
    def _compute_fisher_matrix_mpi(self, fisher_matrix, step_sizes):
        from mpi4py import MPI

        diagonal_indices, offdiag_pairs = self._get_rank_tasks()

        try:
            local_diagonal = {}
            for i in diagonal_indices:
                print(f"[mpi: {self.rank}] Computing ∂²/∂θ{i}θ{i} with step size {step_sizes[i]}")
                local_diagonal[i] = self._compute_second_derivative(i, step_sizes[i])

            local_offdiagonal = {}
            for i, j in offdiag_pairs:
                print(f"[mpi: {self.rank}] Computing ∂²/∂θ{i}θ{j} with step size ({step_sizes[i]}, {step_sizes[j]})")
                local_offdiagonal[(i,j)] = self._compute_mixed_second_derivative(i,j,step_sizes[i], step_sizes[j])

            diagonal_gathered = self.comm.gather(local_diagonal, root=0)
            offdiag_gathered = self.comm.gather(local_offdiagonal, root=0)

            if self.rank == 0:
                fisher_matrix = np.zeros((self.n_params, self.n_params))
                for rank_diagonal in diagonal_gathered:
                    for param_idx, value in rank_diagonal.items():
                        fisher_matrix[param_idx, param_idx] = value

                for rank_offdiagonal in offdiag_gathered:
                    for (i,j), value in rank_offdiagonal.items():
                        fisher_matrix[i,j] = value
                        fisher_matrix[j,i] = value

            fisher_matrix = self.comm.bcast(fisher_matrix, root=0)
        except Exception as e:
            raise RuntimeError(f"Failed to compute Fisher matrix: {str(e)}") from e

        return fisher_matrix

    def _convert_to_absolute_steps(self, relative_steps: np.ndarray) -> np.ndarray:
        """
        Convert relative step sizes to absolute step sizes.
        
        Parameters
        ----------
        relative_steps : ndarray
            Relative step sizes as fractions of fiducial parameter values
            
        Returns
        -------
        absolute_steps : ndarray
            Absolute step sizes for numerical differentiation
        """
        abs_fiducial = np.abs(self.fiducial_params)
        
        # Handle zero or very small fiducial values
        min_scale = np.max(abs_fiducial) * 1e-10  # Minimum scale to avoid division issues
        
        # Use relative steps, but with a minimum scale for very small parameters
        effective_scale = np.maximum(abs_fiducial, min_scale)
        absolute_steps = relative_steps * effective_scale
        
        return absolute_steps
    
    def _compute_second_derivative(self, 
                                 param_idx: int, 
                                 step_size: float) -> float:
        """Compute second derivative with respect to parameter at param_idx, using either central or forward differences."""
        
        if self.use_central_diff[param_idx]:
            return self._compute_second_derivative_central(param_idx, step_size)
        else:
            return self._compute_second_derivative_forward(param_idx, step_size)
    
    def _compute_second_derivative_central(self, 
                                           param_idx: int, 
                                           step_size: float) -> float:
        """Compute second derivative using 4th-order central differences."""
        
        # 4th-order central difference formula for second derivative:
        # f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)
        
        param_name = self.param_names[param_idx]
        
        # Create parameter dictionaries for different step sizes
        params_plus2 = self.fiducial_dict.copy()
        params_plus = self.fiducial_dict.copy()
        params_minus = self.fiducial_dict.copy()
        params_minus2 = self.fiducial_dict.copy()
        
        params_plus2[param_name] += 2 * step_size
        params_plus[param_name] += step_size
        params_minus[param_name] -= step_size
        params_minus2[param_name] -= 2 * step_size
        
        f_plus2 = self.pars_to_loglike(params_plus2)
        f_plus = self.pars_to_loglike(params_plus)
        f_minus = self.pars_to_loglike(params_minus)
        f_minus2 = self.pars_to_loglike(params_minus2)
        f_fiducial = self.pars_to_loglike(self.fiducial_dict)
        
        second_deriv = (-f_plus2 + 16*f_plus - 30*f_fiducial + 16*f_minus - f_minus2) / (12 * step_size**2)
        
        return -second_deriv  # Negative of second derivative
    
    def _compute_second_derivative_forward(self, 
                                           param_idx: int, 
                                           step_size: float) -> float:
        """Compute second derivative using 2nd-order forward (one-sided) differences.
        
        This is used for parameters with fiducial=0 that must be non-negative.
        
        Formula: f''(x) ≈ [-f(x+3h) + 4 f(x+2h) - 5 f(x+h) +2 f(x)] / h² + O(h²)
        """
        
        # 2n-order forward difference formula for second derivative:
        # f''(x) ≈ [-f(x+3h) + 4 f(x+2h) - 5 f(x+h) +2 f(x)] / h² + O(h²)
        
        param_name = self.param_names[param_idx]
        
        # Create parameter dictionaries for different step sizes (all positive)
        params_fiducial = self.fiducial_dict.copy()
        params_plus = self.fiducial_dict.copy()
        params_plus2 = self.fiducial_dict.copy()
        params_plus3 = self.fiducial_dict.copy()
        
        params_plus[param_name] += step_size
        params_plus2[param_name] += 2 * step_size
        params_plus3[param_name] += 3 * step_size
        
        f_fiducial = self.pars_to_loglike(params_fiducial)
        f_plus = self.pars_to_loglike(params_plus)
        f_plus2 = self.pars_to_loglike(params_plus2)
        f_plus3 = self.pars_to_loglike(params_plus3)
        
        second_deriv = (-f_plus3 + 4*f_plus2 - 5*f_plus + 2*f_fiducial) / (step_size**2)

        return -second_deriv  # Negative of second derivative
    
    def _compute_mixed_second_derivative(self, 
                                       param_idx1: int, 
                                       param_idx2: int,
                                       step_size1: float, 
                                       step_size2: float) -> float:
        """Compute mixed second derivative with respect to two parameters.
        
        Handles all combinations: central/central, central/forward, forward/central, forward/forward.
        Uses the "derivative of derivative" approach for heterogeneous schemes.
        """
        
        param_name1 = self.param_names[param_idx1]
        param_name2 = self.param_names[param_idx2]
        
        # Determine which schemes to use
        use_central1 = self.use_central_diff[param_idx1]
        use_central2 = self.use_central_diff[param_idx2]
        
        # Case 1: Both central (simple 2nd-order formula)
        if use_central1 and use_central2:
            params_pp = self.fiducial_dict.copy()
            params_pm = self.fiducial_dict.copy()
            params_mp = self.fiducial_dict.copy()
            params_mm = self.fiducial_dict.copy()
            
            params_pp[param_name1] += step_size1
            params_pp[param_name2] += step_size2
            
            params_pm[param_name1] += step_size1
            params_pm[param_name2] -= step_size2
            
            params_mp[param_name1] -= step_size1
            params_mp[param_name2] += step_size2
            
            params_mm[param_name1] -= step_size1
            params_mm[param_name2] -= step_size2
            
            f_pp = self.pars_to_loglike(params_pp)
            f_pm = self.pars_to_loglike(params_pm)
            f_mp = self.pars_to_loglike(params_mp)
            f_mm = self.pars_to_loglike(params_mm)
            
            mixed_deriv = (f_pp - f_pm - f_mp + f_mm) / (4 * step_size1 * step_size2)
        
        else:
            # Cases 2-4: At least one parameter uses forward differences
            # Use derivative-of-derivative approach
            mixed_deriv = self._compute_mixed_deriv_heterogeneous(
                param_idx1, param_idx2, step_size1, step_size2
            )
        
        return -mixed_deriv  # Negative of mixed second derivative
    
    def _compute_mixed_deriv_heterogeneous(self,
                                          param_idx1: int,
                                          param_idx2: int,
                                          step_size1: float,
                                          step_size2: float) -> float:
        """Compute mixed derivative when schemes are heterogeneous.
        
        Uses ∂²f/∂x∂y = ∂/∂x (∂f/∂y) approach, applying appropriate
        difference scheme to each direction.
        """
        
        param_name1 = self.param_names[param_idx1]
        param_name2 = self.param_names[param_idx2]
        
        use_central1 = self.use_central_diff[param_idx1]
        use_central2 = self.use_central_diff[param_idx2]
        
        # Compute ∂f/∂y at several x-points
        # x points: fiducial, ±h₁ (if central), or +0, +h₁, +2h₁ (if forward)
        
        if use_central1:
            # Central differences for param1: need x-h₁, x, x+h₁
            x_points = [
                ('-', -step_size1),
                ('+', step_size1)
            ]
        else:
            # Forward differences for param1: need x, x+h₁, x+2h₁
            x_points = [
                ('0', 0),
                ('+', step_size1),
                ('++', 2*step_size1),
            ]
        
        # For each x point, compute ∂f/∂y using appropriate scheme for param2
        df_dy_values = {}
        for x_label, dx in x_points:
            params_x = self.fiducial_dict.copy()
            params_x[param_name1] += dx
            
            # Compute ∂f/∂y at this x using param2's scheme
            if use_central2:
                # Central:  [f(x, y+h₂) - f(x, y-h₂)] / (2h₂) + O(h₂²)
                params_y_plus = params_x.copy()
                params_y_plus[param_name2] += step_size2
                params_y_minus = params_x.copy()
                params_y_minus[param_name2] -= step_size2
                
                f_y_plus = self.pars_to_loglike(params_y_plus)
                f_y_minus = self.pars_to_loglike(params_y_minus)
                df_dy_values[x_label] = (f_y_plus - f_y_minus) / (2 * step_size2)
            else:
                # Forward: [-f(x, y+2h₂) + 4f(x, y+h₂) - 3 f(x,y)] / (2h₂) + O(h₂)
                params_y_0 = params_x.copy()
                params_y_plus = params_x.copy()
                params_y_plus2 = params_x.copy()
                
                params_y_plus[param_name2] += step_size2
                params_y_plus2[param_name2] += 2 * step_size2
                
                f_y_0 = self.pars_to_loglike(params_y_0)
                f_y_plus = self.pars_to_loglike(params_y_plus)
                f_y_plus2 = self.pars_to_loglike(params_y_plus2)
                df_dy_values[x_label] = (-f_y_plus2 + 4*f_y_plus - 3*f_y_0) / (2*step_size2)
        
        # Now compute ∂²f/∂x∂y = ∂/∂x (∂f/∂y)
        if use_central1:
            # Central difference on x-direction
            mixed_deriv = (df_dy_values['+'] - df_dy_values['-']) / (2 * step_size1)
        else:
            # Forward 2nd-order difference on x-direction
            # Formula: [-∂f/∂y(x+3h₁) + 4∂f/∂y(x+2h₁) - 5∂f/∂y(x+h₁) + 2∂f/∂y(x)] / h₁
            mixed_deriv = (-df_dy_values['++'] + 4*df_dy_values["+"] - 3*df_dy_values['0']) / (2*step_size1)
        
        return mixed_deriv #negative taken care of in dispatch function
    
    def _get_rank_tasks(self):

        if not self.use_mpi:
            diagonal_indices = list(range(self.n_params))
            offdiag_pairs = [(i,j) for i in range(self.n_params) for j in range(i+1, self.n_params)]
            return diagonal_indices, offdiag_pairs

        diagonal_indices = []
        for i in range(self.n_params):
            if  i % self.size == self.rank:
                diagonal_indices.append(i)

        offdiag_pairs = []
        for i in range(self.n_params):
            for j in range(i+1, self.n_params):
                global_idx = i * self.n_params + j
                if global_idx % self.size == self.rank:
                    offdiag_pairs.append((i,j))
    
        return diagonal_indices, offdiag_pairs


    def _adaptive_step_selection(self, stability_threshold: float, relative: bool = True) -> np.ndarray:
        """
        Select optimal step sizes using adaptive algorithm.
        
        The algorithm tests different step sizes and selects the one that
        gives the most stable numerical derivatives.
        
        Parameters
        ----------
        stability_threshold : float
            Threshold for stability assessment
        relative : bool
            Whether to work with relative step sizes
        """
        

        step_candidates = np.logspace(
            np.log10(self.min_step_size),
            np.log10(self.max_step_size),
            num=10
        )

        
        optimal_steps = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            best_step = self.default_step_size
            best_stability = float('inf')
            print(f"Testing step sizes for parameter {i}")
            for step in step_candidates:
                try:
                    # Convert to absolute step size for derivative computation
                    if relative:
                        abs_step = self._convert_to_absolute_steps(np.array([step]))[0]
                    else:
                        abs_step = step
                    print(f"Testing step size {abs_step} for parameter {i}")
                    # Compute derivative with this step size
                    deriv1 = self._compute_second_derivative(i, abs_step)
                    deriv2 = self._compute_second_derivative(i, abs_step/2)
                    
                    # Estimate stability (relative difference between step sizes)
                    if abs(deriv1) > 1e-15:
                        stability = abs(deriv1 - deriv2) / abs(deriv1)
                    else:
                        stability = abs(deriv1 - deriv2)
                    
                    if stability < best_stability:
                        best_stability = stability
                        best_step = step  # Store the original step (relative or absolute)
                        
                except Exception as e:
                    print(f"Error computing derivative for parameter {i}: {e}")
                    continue
            print(f"Best step size for parameter {i}: {best_step}")
            optimal_steps[i] = best_step
        
        return optimal_steps
    
    def _assess_stability(self, 
                         step_sizes: np.ndarray) -> dict:
        """Assess the stability of the computed derivatives."""
        
        stability_metrics = {}
        
        # Test stability by comparing with half step size
        diagonal_stabilities = []
        
        for i in range(self.n_params):
            try:
                deriv_full = self._compute_second_derivative(i, step_sizes[i])
                deriv_half = self._compute_second_derivative(i, step_sizes[i]/2)
                
                if abs(deriv_full) > 1e-15:
                    relative_error = abs(deriv_full - deriv_half) / abs(deriv_full)
                else:
                    relative_error = abs(deriv_full - deriv_half)
                
                diagonal_stabilities.append(relative_error)
                
            except Exception:
                diagonal_stabilities.append(float('inf'))
        
        stability_metrics['diagonal_relative_errors'] = np.array(diagonal_stabilities)
        stability_metrics['max_relative_error'] = np.max(diagonal_stabilities)
        stability_metrics['mean_relative_error'] = np.mean(diagonal_stabilities)
        stability_metrics['step_sizes'] = step_sizes.copy()
        
        return stability_metrics
    
    def get_parameter_errors(self) -> np.ndarray:
        """
        Compute 1-sigma parameter errors from the Fisher matrix.
        
        Returns
        -------
        errors : ndarray
            1-sigma parameter errors (diagonal elements of inverse Fisher matrix)
        """
        if self._fisher_matrix is None:
            raise RuntimeError("Fisher matrix not computed yet. Call compute_fisher_matrix() first.")
        
        try:
            fisher_inv = np.linalg.inv(self._fisher_matrix)
            errors = np.sqrt(np.diag(fisher_inv))
        except np.linalg.LinAlgError:
            warnings.warn("Fisher matrix is singular or near-singular. Parameter errors may be unreliable.")
            errors = np.full(self.n_params, np.nan)
        
        return errors
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Compute parameter correlation matrix from the Fisher matrix.
        
        Returns
        -------
        correlation_matrix : ndarray
            Parameter correlation matrix
        """
        if self._fisher_matrix is None:
            raise RuntimeError("Fisher matrix not computed yet. Call compute_fisher_matrix() first.")
        
        try:
            fisher_inv = np.linalg.inv(self._fisher_matrix)
            errors = np.sqrt(np.diag(fisher_inv))
            
            # Avoid division by zero
            correlation_matrix = np.zeros_like(fisher_inv)
            for i in range(self.n_params):
                for j in range(self.n_params):
                    if errors[i] > 0 and errors[j] > 0:
                        correlation_matrix[i, j] = fisher_inv[i, j] / (errors[i] * errors[j])
                    else:
                        correlation_matrix[i, j] = 0.0
            
        except np.linalg.LinAlgError:
            warnings.warn("Fisher matrix is singular or near-singular. Correlation matrix may be unreliable.")
            correlation_matrix = np.full((self.n_params, self.n_params), np.nan)
        
        return correlation_matrix
    
    def print_summary(self):
        """Print a summary of the Fisher matrix computation results."""
        if self._fisher_matrix is None:
            print("Fisher matrix not computed yet.")
            return
        
        print("=" * 60)
        print("FISHER MATRIX COMPUTATION SUMMARY")
        print("=" * 60)
        
        print(f"Number of parameters: {self.n_params}")
        print(f"Parameter names: {self.param_names}")
        print()
        
        print("Fiducial parameters:")
        for name, value in zip(self.param_names, self.fiducial_params):
            print(f"  {name}: {value:.6e}")
        print()
        
        if self._stability_metrics is not None:
            print("Stability Assessment:")
            print(f"  Maximum relative error: {self._stability_metrics['max_relative_error']:.2e}")
            print(f"  Mean relative error: {self._stability_metrics['mean_relative_error']:.2e}")
            print()
            
            print("Step sizes used:")
            for name, step in zip(self.param_names, self._step_sizes_used):
                print(f"  {name}: {step:.2e}")
            print()
        
        # Parameter errors
        try:
            errors = self.get_parameter_errors()
            print("1-sigma parameter errors:")
            for name, error in zip(self.param_names, errors):
                if np.isfinite(error):
                    print(f"  {name}: {error:.6e}")
                else:
                    print(f"  {name}: NaN (singular matrix)")
            print()
        except Exception:
            print("Could not compute parameter errors (singular matrix)")
            print()
        
        print("Fisher matrix eigenvalues:")
        eigenvals = np.linalg.eigvals(self._fisher_matrix)
        eigenvals_sorted = np.sort(eigenvals)[::-1]  # Largest first
        for i, eigenval in enumerate(eigenvals_sorted):
            print(f"  λ_{i+1}: {eigenval:.6e}")
        
        print("=" * 60)
    
    def save_to_pickle(self, directory: str, filename: Optional[str] = None) -> str:
        """
        Save the computed Fisher matrix to a pickle file in the specified directory.
        
        Parameters
        ----------
        directory : str
            Directory path where the pickle file will be saved
        filename : str, optional
            Name of the pickle file. If None, uses default name 'fisher_matrix.pkl'
            
        Returns
        -------
        filepath : str
            Full path to the saved pickle file
            
        Raises
        ------
        RuntimeError
            If Fisher matrix has not been computed yet
        OSError
            If directory cannot be created or file cannot be written
        """

        if self.rank !=0:
            print(f"[mpi: {self.rank}] Skipping save (non-zero rank)")
            return ""

        if self._fisher_matrix is None:
            raise RuntimeError("Fisher matrix not computed yet. Call compute_fisher_matrix() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Set default filename if not provided
        if filename is None:
            filename = "fisher_matrix.pkl"
        
        # Ensure filename has .pkl extension
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        # Create full filepath
        filepath = os.path.join(directory, filename)
        
        # Prepare data dictionary
        data = {
            'fisher_matrix': self._fisher_matrix,
            'param_names': self.param_names,
            'fiducial_params': self.fiducial_params,
            'step_sizes_used': self._step_sizes_used,
            'stability_metrics': self._stability_metrics
        }
        
        # Save to pickle file
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise OSError(f"Failed to save Fisher matrix to {filepath}: {str(e)}") from e
        
        return filepath
