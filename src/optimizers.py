"""Basic optimizer updates; noise-matching and schedules are not fixed here."""

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class GradientEvaluation:
    gradient: np.ndarray
    component_evals: int


class GradientOracle(Protocol):
    def evaluate(
        self, x: np.ndarray, batch_size: int, rng: np.random.Generator
    ) -> GradientEvaluation: ...


@dataclass(frozen=True)
class FiniteSumOracle:
    """Mean of a uniform sample of components, without replacement per step."""

    population_size: int
    component_gradient: Callable[[np.ndarray, int], np.ndarray]

    def evaluate(self, x, batch_size, rng) -> GradientEvaluation:
        if (type(self.population_size) is not int or type(batch_size) is not int
                or not 1 <= batch_size <= self.population_size):
            raise ValueError("batch_size must be within the finite population")
        ids = (range(self.population_size) if batch_size == self.population_size else
               rng.choice(self.population_size, size=batch_size, replace=False))
        gradients = [np.asarray(self.component_gradient(x, int(i)), dtype=float) for i in ids]
        if any(g.shape != x.shape for g in gradients):
            raise ValueError("Each component gradient must match the state shape")
        return GradientEvaluation(np.mean(gradients, axis=0), batch_size)


@dataclass(frozen=True)
class StepResult:
    x: np.ndarray
    gradient: np.ndarray  # evaluated at the input state, not at x
    added_noise: np.ndarray
    expected_added_energy: float
    component_evals: int


class Optimizer(Protocol):
    @property
    def hyperparameters(self) -> dict: ...

    def step(self, x: np.ndarray, oracle: GradientOracle) -> StepResult: ...


class SGD:
    """Fixed-step SGD; the oracle owns sampling semantics and cost reporting."""

    def __init__(self, *, learning_rate: float, batch_size: int, seed: int):
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if type(batch_size) is not int or batch_size < 1 or type(seed) is not int or seed < 0:
            raise ValueError("Expected positive integer batch_size and nonnegative integer seed")
        self.learning_rate, self.batch_size, self.seed = learning_rate, batch_size, seed
        batch_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
        self._batch_rng = np.random.default_rng(batch_seed)
        self._noise_rng = np.random.default_rng(noise_seed)

    @property
    def hyperparameters(self) -> dict:
        return {"learning_rate": self.learning_rate, "batch_size": self.batch_size,
                "seed": self.seed}

    def _noise(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        return np.zeros_like(x), 0.0

    def step(self, x: np.ndarray, oracle: GradientOracle) -> StepResult:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("Expected a finite, nonempty vector")
        result = oracle.evaluate(x, self.batch_size, self._batch_rng)
        g = np.asarray(result.gradient, dtype=float)
        if g.shape != x.shape:
            raise ValueError("Oracle gradient must match state shape")
        if type(result.component_evals) is not int or result.component_evals < 0:
            raise ValueError("Oracle must return a nonnegative integer cost")
        noise, energy = self._noise(x)
        # Report the attempted cost even when the arithmetic produces a
        # nonfinite state; the caller decides how to classify/stop that run.
        with np.errstate(over="ignore", invalid="ignore"):
            updated = x - self.learning_rate * g + noise
        return StepResult(updated, g, noise, energy, result.component_evals)


class SGLD(SGD):
    """Fixed-temperature SGLD from the implementation card."""

    def __init__(self, *, temperature: float, **kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        self.temperature = temperature

    @property
    def hyperparameters(self) -> dict:
        return {**super().hyperparameters, "temperature": self.temperature}

    def _noise(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        variance = 2 * (self.learning_rate * self.temperature)
        if variance == 0:
            return super()._noise(x)
        return self._noise_rng.normal(size=x.shape) * math.sqrt(variance), variance * x.size
