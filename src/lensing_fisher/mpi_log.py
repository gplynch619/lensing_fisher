"""Rank-aware printing, so setup messages appear once rather than once per rank.

The alternative is threading a ``verbose`` flag through every function that has
something to say, which makes call sites noisier than the messages are worth.
``driver.run`` calls :func:`set_rank` once; everything else just calls
:func:`info`.

Per-element progress from the Fisher loop deliberately does *not* go through
here — you want to see every rank making progress.
"""

_rank = 0


def set_rank(rank: int) -> None:
    """Record this process's MPI rank. Only rank 0 prints."""
    global _rank
    _rank = int(rank)


def info(message: str) -> None:
    """Print ``message`` on rank 0 only."""
    if _rank == 0:
        print(f"[lensing_fisher] {message}")
