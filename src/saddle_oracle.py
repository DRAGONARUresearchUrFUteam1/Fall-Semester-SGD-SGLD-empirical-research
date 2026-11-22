"""Paired finite-sum oracle with exact mini-batch covariance benchmarks."""

import numpy as np

from .optimizers import FiniteSumOracle, GradientEvaluation
from .saddle_benchmark import SaddleBenchmark


class PairedSaddleOracle:
    """Components are F_c(x) + a_i^T x, in order +s_1, -s_1, +s_2, -s_2, ..."""

    sampling = "uniform_without_replacement_per_step"
    covariance_source = "known_by_design"

    def __init__(self, benchmark: SaddleBenchmark, directions):
        try:
            rows = np.asarray(directions, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Expected finite two-dimensional directions") from exc
        if rows.ndim != 2 or rows.shape[1] != 2 or not len(rows) or not np.isfinite(rows).all():
            raise ValueError("Expected finite two-dimensional directions")

        self._benchmark = benchmark
        self._directions = tuple(tuple(map(float, row)) for row in rows)
        # C_N for signed pairs equals the average outer product of the p directions.
        scaled = rows / np.sqrt(len(rows))
        with np.errstate(over="ignore", invalid="ignore"):
            self._population_covariance = scaled.T @ scaled
            population_trace = np.trace(self._population_covariance)
        if not np.isfinite(population_trace):
            raise ValueError("Direction covariance exceeds numerical range")
        self._finite_sum = FiniteSumOracle(2 * len(rows), self.component_gradient)

    @property
    def directions(self) -> tuple[tuple[float, float], ...]:
        return self._directions

    @property
    def population_size(self) -> int:
        return self._finite_sum.population_size

    def component_gradient(self, x: np.ndarray, index: int) -> np.ndarray:
        if not 0 <= index < self.population_size:
            raise IndexError("Component index outside population")
        sign = 1 if index % 2 == 0 else -1
        with np.errstate(over="ignore", invalid="ignore"):
            return (np.asarray(self._benchmark.subgradient(x))
                    + sign * np.asarray(self.directions[index // 2]))

    def evaluate(
        self, x: np.ndarray, batch_size: int, rng: np.random.Generator
    ) -> GradientEvaluation:
        return self._finite_sum.evaluate(x, batch_size, rng)

    def gradient_covariance(self, batch_size: int) -> np.ndarray:
        """Covariance of the batch-mean gradient, before learning-rate scaling."""
        n = self.population_size
        if type(batch_size) is not int or not 1 <= batch_size <= n:
            raise ValueError("batch_size must be within the finite population")
        return ((n - batch_size) / (batch_size * (n - 1))) * self._population_covariance

    def gradient_covariance_trace(self, batch_size: int) -> float:
        return float(np.trace(self.gradient_covariance(batch_size)))
