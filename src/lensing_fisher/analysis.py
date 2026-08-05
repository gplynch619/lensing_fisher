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
        # Fewer than two samples makes the trapezoid rule degenerate. One sample
        # is the dangerous case: it integrates to 0, so r_j comes out 0/0 = NaN
        # rather than raising. Callers should sample the bin, not widen it —
        # see _ratio_grid.
        if inside.sum() < 2:
            raise ValueError(
                f"bin {j} = [{edges[j]}, {edges[j+1]}) has {int(inside.sum())} "
                f"ell sample(s); the C_fid-weighted mean needs at least 2. Pass a "
                f"grid that resolves every bin."
            )
        r[j] = _trapz(clpp_tem[inside], ells[inside]) / _trapz(clpp_fid[inside], ells[inside])
    return r


def _ratio_grid(bin_edges, L_grid, min_per_bin: int = 16) -> np.ndarray:
    """L samples dense enough that every bin resolves the ``r_j`` quadrature.

    ``r_j`` is a ratio of integrals of two smooth functions, so it is a quadrature
    question and not a multipole question: nothing requires the samples to be
    integers, and on a fine grid the integers alone do not suffice. A log-spaced
    grid over L=2..2000 puts several bins between consecutive multipoles, which
    left them with one sample or none.

    The integer grid is kept as well, so wide bins are sampled at least as well as
    before and the result is unchanged where it was already well defined.
    """
    edges = np.asarray(bin_edges, dtype=float)
    per_bin = [
        np.linspace(edges[j], edges[j + 1], min_per_bin, endpoint=False)
        for j in range(edges.size - 1)
    ]
    return np.unique(np.concatenate([np.asarray(L_grid, dtype=float), *per_bin]))


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

def weight_quantile(w, L_grid, q):
    """``L`` at cumulative-weight fraction ``q``. Scalar or array ``q``."""
    L = np.asarray(L_grid, dtype=float)
    w = np.asarray(w, dtype=float)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(L))])
    if cdf[-1] <= 0:
        raise ValueError("total weight is non-positive; cannot form a quantile")
    return np.interp(q, cdf / cdf[-1], L)


def effective_L(w, L_grid, lower=0.16, upper=0.84):
    """``(L_eff, L_minus, L_plus)`` — the mean and central 68% of the weight.

    The mean is the fragile one. It carries the full lever arm of the tail, so a
    single very wide bin holding almost no information can dominate it — which is
    why :func:`summarize` keeps the catch-all bin out of these moments and reports
    :func:`weight_quantile` at 0.5 alongside. See ``summarize``.
    """
    L = np.asarray(L_grid, dtype=float)
    w = np.asarray(w, dtype=float)

    if np.any(w < 0):
        share = float(np.sum(w[w < 0]) / np.sum(np.abs(w)))
        info(f"warning: weight has negative regions ({share:.2%} of |w|); "
             f"the marginalized Fisher may be ill-conditioned")

    total = _trapz(w, L)
    if total <= 0:
        raise ValueError("total weight is non-positive; cannot form L_eff")

    return (
        float(_trapz(L * w, L) / total),
        float(weight_quantile(w, L, lower)),
        float(weight_quantile(w, L, upper)),
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

def summarize(fisher, template_file: Optional[str] = None, L_grid=None,
              exclude_catchall: bool = True) -> dict:
    """Run the whole chain on a Fisher pickle (or an already-loaded dict).

    ``template_file`` supplies C_tem. Omit it when the Fisher was run with
    ``C_fid = C_tem``, in which case ``r`` is exactly 1 and no correction applies.

    ``exclude_catchall`` keeps the final bin out of the ``L_eff`` / band / median
    moments; see the comment at the point of use for why that bin is pathological
    for this purpose. It still contributes to ``w_bins``, ``w`` and
    ``sigma_A_template``, which are unaffected by the issue. Set False to
    reproduce the older behaviour.
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
        # Quadratured on its own grid: L_grid is the integer multipoles, which is
        # right for w(L) but can leave a narrow bin with too few samples for the
        # ratio integral.
        r_grid = _ratio_grid(edges, L_grid)
        # Both spectra go through the *same* log-log spline. They used to differ —
        # C_fid linearly interpolated, C_tem splined — which agreed only because
        # the old grid was integers, where both reproduce their samples exactly.
        # Off-grid they diverge, and steeply so at low L, biasing r_j there.
        clpp_fid = np.asarray(fisher["clpp_fid"], dtype=float)
        L_fid = np.arange(clpp_fid.size)
        positive = clpp_fid > 0
        fid_on_grid = clpp_template.interpolate_to_ells(
            L_fid[positive], clpp_fid[positive], r_grid)
        L_tem, CL_tem = clpp_template.load_template(template_file)
        tem_on_grid = clpp_template.interpolate_to_ells(L_tem, CL_tem, r_grid)
        r = bin_averaged_ratio(edges, tem_on_grid, fid_on_grid, r_grid)

    w_bins = information_per_bin(F_lens, r)
    w = weight_density(w_bins, edges, L_grid)

    # The moments stop below the last bin. That bin runs to CAMB's max_l — the
    # outermost basis function is one-sided, so it scales lensing all the way up
    # whatever its nominal top edge says — and its width is therefore set by the
    # theory lmax, not by the data. It exists so that a uniform q is exactly
    # A_template, which is a requirement of the parametrization and not a claim
    # about where information lives.
    #
    # Leaving it in wrecks the mean. weight_density spreads each bin's weight
    # uniformly, which is harmless for a bin a few multipoles wide and badly
    # wrong for one 7551 wide, where the real information falls steeply toward
    # the low edge. Measured on the SPA runs: iteration 0's catch-all was
    # [2000, 8550] and held 0.02% of the information, so L_eff was 187.2 either
    # way; iteration 1's was [999, 8550] and held 2.1%, and L_eff went 164.2 ->
    # 261.2 on that alone, landing outside its own 68% upper bound of 242.8. The
    # band and the median moved by less than 1% across the same change.
    excluding = bool(exclude_catchall and edges.size >= 3)
    moment_max = float(edges[-2]) if excluding else float(edges[-1])
    # Strictly less: bins are half-open [lo, hi), so L == edges[-2] is the first
    # multipole *of* the catch-all and carries its density, not its neighbour's.
    inside = (L_grid < moment_max) if excluding else (L_grid <= moment_max)
    L_eff, L_minus, L_plus = effective_L(w[inside], L_grid[inside])
    L_median = float(weight_quantile(w[inside], L_grid[inside], 0.5))
    excluded_share = float(np.sum(w_bins[-1:]) / np.sum(w_bins)) if excluding else 0.0

    return {
        "F_lens": F_lens,
        "r": r,
        "w_bins": w_bins,
        "w": w,
        "L_grid": L_grid,
        "bin_edges": edges,
        "L_eff": L_eff,
        "L_median": L_median,
        "L_minus": L_minus,
        "L_plus": L_plus,
        "moment_L_max": moment_max,
        "excluded_catchall": excluding,
        "excluded_information": excluded_share,
        "sigma_A_template": float(1.0 / np.sqrt(np.sum(w_bins))),
    }
