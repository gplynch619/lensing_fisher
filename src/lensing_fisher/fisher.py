"""Fisher matrix from finite differences of a log-likelihood.

    F_ij = - d^2 logL / dtheta_i dtheta_j   evaluated at the fiducial point

Diagonals use the 4th-order central formula, off-diagonals the standard 4-point
mixed formula. Both are written out below in the same form you would find them
in a textbook.

Work is distributed over MPI ranks by matrix element. The distribution is not a
plain round robin: elements are bucketed by which *expensive* parameters they
perturb, so a rank walks a run of elements that share an upstream computation
(for us, a CAMB results object) and a small cache in the likelihood actually
hits. See :meth:`FisherMatrix._tasks`.
"""

import os
import pickle
from typing import Callable, Iterable, Optional

import numpy as np


class FisherMatrix:
    """Finite-difference Fisher matrix over a dict-valued log-likelihood.

    Parameters
    ----------
    loglike
        ``f(params: dict) -> float``. Only relative values matter.
    fiducial
        Parameter names and the point to expand about. Iteration order sets the
        row/column order of the matrix.
    step_sizes
        Absolute finite-difference step per parameter, same order as ``fiducial``.
    comm
        ``mpi4py`` communicator, or None for serial.
    cached_params
        Names of parameters that invalidate an expensive upstream computation
        inside ``loglike``. Used only to order the work; results are unaffected.
    """

    def __init__(
        self,
        loglike: Callable[[dict], float],
        fiducial: dict,
        step_sizes: Iterable[float],
        comm=None,
        cached_params: Iterable[str] = (),
    ):
        self.loglike = loglike
        self.fiducial = dict(fiducial)
        self.param_names = list(self.fiducial)
        self.n_params = len(self.param_names)

        self.step_sizes = np.asarray(step_sizes, dtype=float)
        if self.step_sizes.shape != (self.n_params,):
            raise ValueError(
                f"step_sizes must have one entry per parameter ({self.n_params}); "
                f"got shape {self.step_sizes.shape}"
            )

        self._cached_indices = frozenset(
            self.param_names.index(p) for p in cached_params if p in self.param_names
        )

        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1
        self.use_mpi = self.size > 1

        self.matrix: Optional[np.ndarray] = None
        self._logl_fiducial: Optional[float] = None

    # ------------------------------------------------------------------
    # Likelihood evaluation
    # ------------------------------------------------------------------

    def _logl(self, offsets: dict) -> float:
        """log-likelihood with ``{param_index: shift}`` added to the fiducial."""
        params = dict(self.fiducial)
        for idx, shift in offsets.items():
            params[self.param_names[idx]] += shift
        return self.loglike(params)

    @property
    def logl_fiducial(self) -> float:
        """The fiducial log-likelihood, evaluated once and reused.

        Every diagonal element needs it; without caching that is one redundant
        likelihood call per parameter.
        """
        if self._logl_fiducial is None:
            self._logl_fiducial = self.loglike(dict(self.fiducial))
        return self._logl_fiducial

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    def _second_derivative(self, i: int) -> float:
        """F_ii, from the 4th-order central second difference.

        f''(x) = [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / 12h^2
        """
        h = self.step_sizes[i]
        f = self._logl
        d2 = (
            -f({i: 2 * h})
            + 16 * f({i: h})
            - 30 * self.logl_fiducial
            + 16 * f({i: -h})
            - f({i: -2 * h})
        ) / (12 * h**2)
        return -d2

    def _mixed_derivative(self, i: int, j: int) -> float:
        """F_ij, from the 4-point central mixed difference.

        d2f/dxdy = [f(+,+) - f(+,-) - f(-,+) + f(-,-)] / 4 hx hy
        """
        hi, hj = self.step_sizes[i], self.step_sizes[j]
        f = self._logl
        d2 = (
            f({i: hi, j: hj})
            - f({i: hi, j: -hj})
            - f({i: -hi, j: hj})
            + f({i: -hi, j: -hj})
        ) / (4 * hi * hj)
        return -d2

    def _element(self, task: tuple) -> float:
        return self._second_derivative(*task) if len(task) == 1 else self._mixed_derivative(*task)

    # ------------------------------------------------------------------
    # Work distribution
    # ------------------------------------------------------------------

    def _tasks(self) -> list:
        """This rank's matrix elements, in execution order.

        Every element is a 1-tuple (diagonal) or 2-tuple (off-diagonal). They are
        bucketed by which ``cached_params`` they perturb — usually none — and
        dealt round robin *within* each bucket. Each bucket is spread evenly over
        the ranks, so load balance matches a plain round robin, but consecutive
        elements on a rank now share an upstream computation.

        For a typical run the "perturbs no cached parameter" bucket holds ~85% of
        all elements (every bin-bin, bin-nuisance and nuisance-nuisance pair) and
        is evaluated entirely at the fiducial cosmology.

        Callers must preserve the returned order; sorting it defeats the point.
        """
        elements = [(i,) for i in range(self.n_params)]
        elements += [(i, j) for i in range(self.n_params) for j in range(i + 1, self.n_params)]

        buckets: dict = {}
        for task in elements:
            key = frozenset(i for i in task if i in self._cached_indices)
            buckets.setdefault(key, []).append(task)

        # Cheapest bucket (perturbs nothing expensive) first, then by size.
        ordered_keys = sorted(buckets, key=lambda k: (len(k), sorted(k)))

        mine = []
        for key in ordered_keys:
            for n, task in enumerate(buckets[key]):
                if n % self.size == self.rank:
                    mine.append(task)
        return mine

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def compute(self) -> np.ndarray:
        """Evaluate every matrix element and assemble the full matrix."""
        if self.rank == 0:
            print(f"[fisher] {self.n_params} parameters, "
                  f"{self.n_params * (self.n_params + 1) // 2} elements, "
                  f"{self.size} rank(s)")

        local = {}
        for task in self._tasks():
            local[task] = self._element(task)
            label = f"{task[0]},{task[0]}" if len(task) == 1 else f"{task[0]},{task[1]}"
            print(f"[fisher rank {self.rank}] F[{label}] = {local[task]:.6e}")

        # allgather rather than gather+bcast: every rank ends up with the whole
        # matrix, and there is no rank-0-only branch to get wrong.
        merged = {}
        if self.use_mpi:
            for chunk in self.comm.allgather(local):
                merged.update(chunk)
        else:
            merged = local

        matrix = np.zeros((self.n_params, self.n_params))
        for task, value in merged.items():
            if len(task) == 1:
                matrix[task[0], task[0]] = value
            else:
                i, j = task
                matrix[i, j] = matrix[j, i] = value

        expected = self.n_params * (self.n_params + 1) // 2
        if len(merged) != expected:
            raise RuntimeError(
                f"assembled {len(merged)} matrix elements but expected {expected}; "
                f"work distribution dropped elements"
            )

        self.matrix = matrix
        return matrix

    # ------------------------------------------------------------------
    # Inspection and output
    # ------------------------------------------------------------------

    def _require_matrix(self) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("Fisher matrix not computed yet; call compute() first")
        return self.matrix

    def parameter_errors(self) -> np.ndarray:
        """1-sigma marginalized errors, sqrt(diag(F^-1))."""
        return np.sqrt(np.diag(np.linalg.inv(self._require_matrix())))

    def summary(self) -> None:
        """Print parameter errors and the eigenvalue spectrum.

        Non-positive eigenvalues mean the matrix is not a valid information
        matrix — usually a step size too small for a parameter, so its second
        difference is dominated by numerical noise.
        """
        F = self._require_matrix()
        print("=" * 60)
        print(f"Fisher matrix: {self.n_params} parameters")
        print("=" * 60)

        errors = self.parameter_errors()
        print("\n1-sigma errors (marginalized):")
        for name, fid, step, err in zip(
            self.param_names, self.fiducial.values(), self.step_sizes, errors
        ):
            print(f"  {name:>12s}  fid={fid:12.6g}  step={step:9.3g}  sigma={err:12.6g}")

        eig = np.sort(np.linalg.eigvalsh(F))[::-1]
        print(f"\nEigenvalues: max={eig[0]:.4e}  min={eig[-1]:.4e}  "
              f"condition={eig[0] / eig[-1]:.3e}")
        n_bad = int(np.sum(eig <= 0))
        if n_bad:
            print(f"  WARNING: {n_bad} non-positive eigenvalue(s); the matrix is "
                  f"not positive definite")
        print("=" * 60)

    def save(self, directory: str, filename: str, metadata: Optional[dict] = None) -> str:
        """Pickle the matrix plus provenance. No-op on non-zero ranks.

        ``metadata`` is merged in at the top level so a saved file can be poked
        at interactively without unwrapping; keys may not shadow the core ones.
        """
        if self.rank != 0:
            return ""

        F = self._require_matrix()
        if np.all(F == 0):
            raise RuntimeError(
                "refusing to save an all-zero Fisher matrix; the likelihood is "
                "probably returning a constant"
            )

        payload = {
            "fisher_matrix": F,
            "param_names": self.param_names,
            "fiducial_params": np.array(list(self.fiducial.values()), dtype=float),
            "step_sizes": self.step_sizes,
        }
        clashes = set(payload) & set(metadata or {})
        if clashes:
            raise ValueError(f"metadata keys shadow core keys: {sorted(clashes)}")
        payload.update(metadata or {})

        os.makedirs(directory, exist_ok=True)
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        path = os.path.join(directory, filename)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path
