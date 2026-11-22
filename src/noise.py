"""Fixed-state mini-batch covariance diagnostics, independent of optimizers."""

from dataclasses import dataclass

import numpy as np

from .optimizers import GradientOracle


@dataclass(frozen=True)
class NoiseEstimate:
    mean: np.ndarray
    eigenvalues: np.ndarray
    trace: float
    rank_entropy: float | None
    rank_participation: float | None
    covariance: np.ndarray | None
    unstable_power: float | None
    unstable_fraction: float | None
    sample_count: int
    dimension: int


def estimate_noise(gradients, *, unstable_basis=None, materialize: bool = False) -> NoiseEstimate:
    """Rows are independent means of mini-batches evaluated at the same state.

    Uses divisor m-1. The returned spectrum omits structural trailing zeros
    when m < d. Both proposed effective ranks are reported under distinct names;
    both are undefined (None) at exactly zero trace.
    """
    g = np.asarray(gradients, dtype=np.float64)
    if (g.ndim != 2 or g.shape[0] < 2 or g.shape[1] == 0
            or not np.isfinite(g).all()):
        raise ValueError("Expected m >= 2 finite gradient rows with equal positive dimension")
    m, d = g.shape
    # Shift before centering: exactly repeated rows produce exactly zero noise.
    try:
        with np.errstate(over="raise", invalid="raise"):
            shifted = g - g[0]
            mean_shift = shifted.mean(axis=0)
            mean = g[0] + mean_shift
            scaled = (shifted - mean_shift) / np.sqrt(m - 1)
            eigenvalues = np.linalg.svd(scaled, compute_uv=False) ** 2
            trace = float(eigenvalues.sum())
            covariance = scaled.T @ scaled if materialize else None
    except FloatingPointError as exc:
        raise ValueError("Covariance calculation exceeds numerical range") from exc
    if not np.isfinite(trace) or (covariance is not None and not np.isfinite(covariance).all()):
        raise ValueError("Nonfinite covariance estimate")

    entropy = participation = None
    if trace > 0:
        p = eigenvalues[eigenvalues > 0] / trace
        p = p[p > 0]
        entropy = float(np.exp(-np.dot(p, np.log(p))))
        participation = float(1 / np.dot(p, p))

    power = fraction = None
    if unstable_basis is not None:
        basis = np.asarray(unstable_basis, dtype=np.float64)
        if (basis.ndim != 2 or basis.shape[0] != d or basis.shape[1] > d
                or not np.isfinite(basis).all()
                or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8, rtol=0)):
            raise ValueError("unstable_basis must have orthonormal columns in parameter space")
        power = float(np.linalg.norm(scaled @ basis, ord="fro") ** 2)
        if trace > 0:
            fraction = power / trace
    return NoiseEstimate(mean, eigenvalues, trace, entropy, participation, covariance,
                         power, fraction, m, d)


def sample_noise(oracle: GradientOracle, x, *, batch_size: int, sample_count: int,
                 rng: np.random.Generator, unstable_basis=None, materialize: bool = False
                 ) -> tuple[NoiseEstimate, int]:
    """Use an independent diagnostic RNG; return cost for the caller to classify."""
    if (type(sample_count) is not int or sample_count < 2
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Expected sample_count >= 2 and batch_size >= 1")
    point = np.array(x, dtype=float, copy=True)
    if point.ndim != 1 or point.size == 0 or not np.isfinite(point).all():
        raise ValueError("Expected a finite, nonempty state vector")
    point.setflags(write=False)
    rows = []
    component_evals = 0
    for _ in range(sample_count):
        sample = oracle.evaluate(point, batch_size, rng)
        row = np.asarray(sample.gradient, dtype=float)
        if row.shape != point.shape or not np.isfinite(row).all():
            raise ValueError("Expected a finite gradient vector at the fixed state")
        if type(sample.component_evals) is not int or sample.component_evals < 0:
            raise ValueError("Expected a nonnegative integer cost")
        rows.append(row.copy())
        component_evals += sample.component_evals
    return estimate_noise(rows, unstable_basis=unstable_basis, materialize=materialize), component_evals
