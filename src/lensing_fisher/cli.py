"""Console entry point: ``lensing-fisher -i config.yaml``."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lensing-fisher",
        description="Run a Fisher-matrix computation for CMB 2pt lensing sensitivity.",
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

    driver.compute_fisher_matrix(cfg, comm=comm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
