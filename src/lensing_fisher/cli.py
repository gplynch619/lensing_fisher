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
    parser.add_argument("-o", "--output", default=None, help="write the YAML fragment here")
    args = parser.parse_args(argv)

    from . import analysis

    result = analysis.summarize(args.fisher, template_file=args.template)
    edges_in = result["bin_edges"]
    n_bins = args.n_bins or (len(edges_in) - 1)

    print(f"input      : {len(edges_in) - 1} bins over L = {edges_in[0]:.0f}..{edges_in[-1]:.0f}")
    print(f"L_eff      : {result['L_eff']:.1f}   68% band "
          f"{result['L_minus']:.1f} .. {result['L_plus']:.1f}")
    print(f"sigma(A_tem) implied by this Fisher: {result['sigma_A_template']:.4f}")

    edges_out = analysis.next_bin_edges(
        result["w"], result["L_grid"], n_bins=n_bins, min_width=args.min_width,
        L_range=(edges_in[0], edges_in[-1]),
    )

    widths = np.diff(edges_out)
    at_floor = int(np.sum(widths < args.min_width * 1.001))
    print(f"proposed   : {n_bins} bins, widths {widths.min():.1f}..{widths.max():.0f}")
    if at_floor:
        print(f"  NOTE: {at_floor}/{n_bins} bins sit at the min-width floor, so the "
              f"equal-information target is not being met. Lower --min-width or "
              f"reduce --n-bins.")

    tol = args.tol if args.tol is not None else args.min_width
    if len(edges_in) == len(edges_out):
        moved = float(np.max(np.abs(edges_in - edges_out)))
        verdict = "CONVERGED" if moved <= tol else "not converged, iterate again"
        print(f"max edge movement: {moved:.2f} (tol {tol:g}) -> {verdict}")

    fragment = (
        f"# lensing-fisher-rebin: {n_bins} equal-information bins, "
        f"min width {args.min_width:g}\n"
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
