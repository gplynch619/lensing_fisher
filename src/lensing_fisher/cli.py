"""Command-line entry points.

    lensing-fisher       -i config.yaml        compute a Fisher matrix
    lensing-fisher-rebin fisher.pkl            propose the next bin grid
"""

import argparse

import numpy as np


def main(argv=None) -> int:
    """``lensing-fisher``: run the Fisher computation described by a YAML config."""
    parser = argparse.ArgumentParser(
        prog="lensing-fisher",
        description="Compute a CMB 2pt lensing sensitivity Fisher matrix.",
    )
    parser.add_argument("-i", "--input", required=True, help="path to YAML config")
    args = parser.parse_args(argv)

    from . import config, driver

    cfg = config.load(args.input)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    driver.run(cfg, comm=comm)
    return 0


def rebin(argv=None) -> int:
    """``lensing-fisher-rebin``: one step of the binning iteration.

    Reports L_eff and its 68% band, proposes an equal-information grid, and says
    whether the edges have settled relative to the grid that produced the input.
    """
    parser = argparse.ArgumentParser(
        prog="lensing-fisher-rebin",
        description="Propose the next equal-information bin grid from a Fisher pickle.",
    )
    parser.add_argument("fisher", help="Fisher pickle written by lensing-fisher")
    parser.add_argument("-n", "--n-bins", type=int, default=None,
                        help="bins to place (default: keep the current count)")
    parser.add_argument("--min-width", type=float, default=3.0,
                        help="minimum bin width in L (default: 3)")
    parser.add_argument("--template", default=None,
                        help="{L, CL_pp_fid} pickle for C_tem; omit if C_fid = C_tem")
    parser.add_argument("--tol", type=float, default=None,
                        help="convergence tolerance on edge movement (default: min-width)")
    parser.add_argument("--l-max", type=float, default=None,
                        help="top of the range to bin over; the catch-all above it is "
                             "left alone and re-appended by the driver "
                             "(default: the current grid's catch-all lower edge)")
    parser.add_argument("-o", "--output", default=None, help="write the YAML fragment here")
    args = parser.parse_args(argv)

    from . import analysis

    result = analysis.summarize(args.fisher, template_file=args.template)
    edges_in = result["bin_edges"]
    # Bin the informative range only. The final bin runs to CAMB's max_l and
    # exists solely so a uniform q is exactly A_template — its width is set by the
    # theory lmax, not by the data. Letting the equal-information placement span
    # it is what broke iteration 1: with ~98% of the weight below L~1000, all 50
    # bins were spread over 2..8550, so 49 landed below 999 and the 50th became
    # [999, 8550]. The resolved range collapsed from 2000 to 999 and a bin was
    # spent on the tail. It also made the catch-all's lower edge a free parameter,
    # which is where the 375.93 "edge movement" and the L_eff swing came from.
    #
    # driver.add_catchall_bin re-appends the tail, so the emitted grid stops here.
    l_max = float(args.l_max) if args.l_max is not None else float(edges_in[-2])
    n_bins = args.n_bins or (len(edges_in) - 2)

    print(f"input      : {len(edges_in) - 1} bins over L = {edges_in[0]:.0f}..{edges_in[-1]:.0f}")
    print(f"L_eff      : {result['L_eff']:.1f} (mean)   {result['L_median']:.1f} (median)"
          f"   68% band {result['L_minus']:.1f} .. {result['L_plus']:.1f}")
    if result["excluded_catchall"]:
        print(f"             moments over L <= {result['moment_L_max']:.0f}; the last bin "
              f"[{edges_in[-2]:.0f}, {edges_in[-1]:.0f}] runs to CAMB's lmax and is "
              f"excluded from them")
    # The mean carries the tail's lever arm and the median does not, so a gap
    # between them is the honest signal that the weight is skewed.
    if abs(result["L_eff"] - result["L_median"]) > 0.25 * (
            result["L_plus"] - result["L_minus"]):
        print(f"  NOTE: mean and median differ by "
              f"{abs(result['L_eff'] - result['L_median']):.0f}, more than a quarter of "
              f"the 68% width — the weight is strongly skewed, so prefer the median.")
    print(f"sigma(A_tem) implied by this Fisher: {result['sigma_A_template']:.4f}")

    # Restrict the weight itself, not just the endpoints: next_bin_edges inverts
    # the CDF, so leaving the tail's weight in would push the top quantiles past
    # l_max and the min-width passes would then have to claw them back.
    keep = result["L_grid"] <= l_max
    edges_out = analysis.next_bin_edges(
        result["w"][keep], result["L_grid"][keep], n_bins=n_bins,
        min_width=args.min_width, L_range=(edges_in[0], l_max),
    )

    widths = np.diff(edges_out)
    at_floor = int(np.sum(widths < args.min_width * 1.001))
    print(f"proposed   : {n_bins} bins over L = {edges_in[0]:.0f}..{l_max:.0f}, "
          f"widths {widths.min():.1f}..{widths.max():.0f}")
    print(f"             (+ a catch-all to CAMB's max_l, re-appended at run time)")
    if at_floor:
        print(f"  NOTE: {at_floor}/{n_bins} bins sit at the min-width floor, so the "
              f"equal-information target is not being met. Lower --min-width or "
              f"reduce --n-bins.")

    # Compare like with like: edges_in carries the catch-all's top edge, edges_out
    # stops at l_max.
    tol = args.tol if args.tol is not None else args.min_width
    previous = edges_in[:-1] if np.isclose(edges_in[-2], l_max) else None
    if previous is not None and len(previous) == len(edges_out):
        moved = float(np.max(np.abs(previous - edges_out)))
        verdict = "CONVERGED" if moved <= tol else "not converged, iterate again"
        print(f"max edge movement: {moved:.2f} (tol {tol:g}) -> {verdict}")
    else:
        print(f"max edge movement: not comparable — the previous grid was binned to "
              f"L={edges_in[-2]:.0f}, this one to L={l_max:.0f}")

    fragment = (
        f"# lensing-fisher-rebin: {n_bins} equal-information bins over "
        f"L = {edges_in[0]:.0f}..{l_max:.0f}, min width {args.min_width:g}.\n"
        f"# The catch-all bin to CAMB's max_l is appended automatically at run\n"
        f"# time (bins.extend_to_lmax), so it is deliberately absent here.\n"
        "bins:\n  edges:\n" + "\n".join(f"    - {e:.4g}" for e in edges_out) + "\n"
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(fragment)
        print(f"wrote {args.output}")
    else:
        print("\n" + fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
