"""Fisher matrix -> effective bandpower window for the 2pt lensing measurement.

The chain of reasoning, following ``results/notes/update.tex``:

1. Marginalize the full matrix over cosmology and nuisance parameters, leaving
   ``F_lens`` over the bin amplitudes ``q_j``.
2. The template amplitude enters through ``r_j = dq_j/dA_tem``, the ratio
   ``C_tem/C_fid`` averaged over bin ``j``. It is exactly 1 when the Fisher was
   run with ``C_fid = C_tem``.
3. The information decomposes bin by bin,
   ``I(A_tem) = sum_jk r_j F_jk r_k = sum_j w_j`` with ``w_j = r_j (F r)_j``,
   so ``w_j`` is bin ``j``'s share.
4. ``L_eff`` and its 68% range are moments of ``w`` viewed as a distribution
   over L.

Step 4 is where non-uniform bins need care. ``w_j`` is a *total* for the bin, so
spreading it over L means dividing by the bin width; otherwise a bin counts
twice as much simply for being twice as wide, dragging ``L_eff`` toward the wide
high-L bins. For a uniform grid the division is a constant and cancels, which is
why the original notebook could omit it.
"""

import pickle
from typing import Optional

import numpy as np

from . import clpp_template
from .mpi_log import info

# numpy renamed trapz -> trapezoid in 2.0 and deprecated the old spelling.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_fisher(path) -> dict:
    """Read a Fisher pickle written by ``driver.run``."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# Step 1: marginalize
# ----------------------------------------------------------------------

def clpp_indices(param_names) -> np.ndarray:
    """Indices of the ``clpp_*`` parameters, ordered by bin number.

    Sorting numerically matters: lexical order would place ``clpp_10`` before
    ``clpp_2`` and silently scramble the bin ordering.
    """
    idx = [i for i, name in enumerate(param_names) if name.startswith("clpp_")]
    if not idx:
        raise ValueError("no clpp_* parameters in param_names")
    order = np.argsort([int(param_names[i].split("_")[1]) for i in idx])
    return np.asarray(idx, dtype=int)[order]


def marginalize(fisher_matrix, param_names) -> np.ndarray:
    """Marginalize over everything that is not a bin amplitude.

    Invert to a covariance, take the sub-block, invert back — the Schur
    complement. Slicing the Fisher matrix directly would instead *condition* on
    the other parameters, i.e. assume cosmology is known exactly.
    """
    idx = clpp_indices(param_names)
    covariance = np.linalg.inv(np.asarray(fisher_matrix, dtype=float))
    return np.linalg.inv(covariance[np.ix_(idx, idx)])


# ----------------------------------------------------------------------
# Step 2: the Jacobian to A_template
# ----------------------------------------------------------------------

def bin_averaged_ratio(bin_edges, clpp_tem, clpp_fid, ells) -> np.ndarray:
    """``r_j``, the C_fid-weighted mean of ``C_tem/C_fid`` over each bin.

    Taking the ratio at bin centres instead is fine for narrow bins but not for
    the wide high-L ones, where the ratio varies appreciably across the bin.
    """
    edges = np.asarray(bin_edges, dtype=float)
    ells = np.asarray(ells, dtype=float)
    which_bin = np.digitize(ells, edges) - 1

    r = np.empty(edges.size - 1)
    for j in range(r.size):
        inside = which_bin == j
        if not inside.any():
            raise ValueError(f"bin {j} = [{edges[j]}, {edges[j+1]}) contains no ell samples")
        r[j] = _trapz(clpp_tem[inside], ells[inside]) / _trapz(clpp_fid[inside], ells[inside])
    return r


# ----------------------------------------------------------------------
# Step 3: information per bin, spread over L
# ----------------------------------------------------------------------

def information_per_bin(F_lens, r) -> np.ndarray:
    """``w_j = r_j (F_lens r)_j``, bin j's share of the information on A_tem."""
    r = np.asarray(r, dtype=float)
    return r * (np.asarray(F_lens, dtype=float) @ r)


def weight_density(w_bins, bin_edges, L_grid) -> np.ndarray:
    """Per-bin information as a density in L, ``w_j / width_j``. Zero outside."""
    edges = np.asarray(bin_edges, dtype=float)
    w = np.asarray(w_bins, dtype=float)
    L = np.asarray(L_grid, dtype=float)

    which_bin = np.digitize(L, edges) - 1
    inside = (which_bin >= 0) & (which_bin < w.size)

    density = np.zeros(L.size)
    density[inside] = (w / np.diff(edges))[which_bin[inside]]
    return density


# ----------------------------------------------------------------------
# Step 4: moments
# ----------------------------------------------------------------------

def effective_L(w, L_grid, lower=0.16, upper=0.84):
    """``(L_eff, L_minus, L_plus)`` — the mean and central 68% of the weight."""
    L = np.asarray(L_grid, dtype=float)
    w = np.asarray(w, dtype=float)

    if np.any(w < 0):
        share = float(np.sum(w[w < 0]) / np.sum(np.abs(w)))
        info(f"warning: weight has negative regions ({share:.2%} of |w|); "
             f"the marginalized Fisher may be ill-conditioned")

    total = _trapz(w, L)
    if total <= 0:
        raise ValueError("total weight is non-positive; cannot form L_eff")

    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(L))])
    cdf /= cdf[-1]

    return (
        float(_trapz(L * w, L) / total),
        float(np.interp(lower, cdf, L)),
        float(np.interp(upper, cdf, L)),
    )


# ----------------------------------------------------------------------
# Placing the next bin grid
# ----------------------------------------------------------------------

def next_bin_edges(w, L_grid, n_bins: int, min_width: float, L_range=None) -> np.ndarray:
    """Contiguous bins carrying equal information, subject to ``min_width``.

    Edges start as quantiles of the weight distribution, then a forward pass and
    a backward pass push them apart to respect ``min_width``. Where that
    constraint binds the bins are no longer equal-information; the caller should
    watch how many end up at the floor.
    """
    L = np.asarray(L_grid, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)

    lo, hi = (float(L[0]), float(L[-1])) if L_range is None else map(float, L_range)
    if n_bins * min_width > (hi - lo):
        raise ValueError(
            f"{n_bins} bins of minimum width {min_width} cannot fit in [{lo}, {hi}]"
        )

    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(L))])
    if cdf[-1] <= 0:
        raise ValueError("weight is everywhere zero; cannot place bins")
    cdf /= cdf[-1]

    # Invert the CDF at equally spaced information levels. np.interp needs a
    # strictly increasing x, so drop the flat stretches where no weight accrues.
    rising = np.concatenate([[True], np.diff(cdf) > 0])
    edges = np.interp(np.linspace(0.0, 1.0, n_bins + 1), cdf[rising], L[rising])
    edges[0], edges[-1] = lo, hi

    for k in range(1, n_bins + 1):                  # forward: open each bin up
        edges[k] = max(edges[k], edges[k - 1] + min_width)
    edges[-1] = hi
    for k in range(n_bins - 1, 0, -1):              # backward: keep the last bin valid
        edges[k] = min(edges[k], edges[k + 1] - min_width)

    if np.any(np.diff(edges) <= 0):
        raise ValueError("could not satisfy min_width; reduce n_bins or min_width")
    return edges


def edges_converged(old, new, tol: float) -> bool:
    """True when no bin edge moved by more than ``tol``."""
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    return old.shape == new.shape and bool(np.max(np.abs(old - new)) <= tol)


# ----------------------------------------------------------------------

def summarize(fisher, template_file: Optional[str] = None, L_grid=None) -> dict:
    """Run the whole chain on a Fisher pickle (or an already-loaded dict).

    ``template_file`` supplies C_tem. Omit it when the Fisher was run with
    ``C_fid = C_tem``, in which case ``r`` is exactly 1 and no correction applies.
    """
    if not isinstance(fisher, dict):
        fisher = load_fisher(fisher)

    edges = np.asarray(fisher["bin_edges"], dtype=float)
    F_lens = marginalize(fisher["fisher_matrix"], fisher["param_names"])

    if L_grid is None:
        L_grid = np.arange(np.floor(edges[0]), np.ceil(edges[-1]) + 1, dtype=float)
    L_grid = np.asarray(L_grid, dtype=float)

    if template_file is None:
        r = np.ones(edges.size - 1)
    else:
        clpp_fid = np.asarray(fisher["clpp_fid"], dtype=float)
        fid_on_grid = np.interp(L_grid, np.arange(clpp_fid.size), clpp_fid)
        L_tem, CL_tem = clpp_template.load_template(template_file)
        tem_on_grid = clpp_template.interpolate_to_ells(L_tem, CL_tem, L_grid)
        r = bin_averaged_ratio(edges, tem_on_grid, fid_on_grid, L_grid)

    w_bins = information_per_bin(F_lens, r)
    w = weight_density(w_bins, edges, L_grid)
    L_eff, L_minus, L_plus = effective_L(w, L_grid)

    return {
        "F_lens": F_lens,
        "r": r,
        "w_bins": w_bins,
        "w": w,
        "L_grid": L_grid,
        "bin_edges": edges,
        "L_eff": L_eff,
        "L_minus": L_minus,
        "L_plus": L_plus,
        "sigma_A_template": float(1.0 / np.sqrt(np.sum(w_bins))),
    }
