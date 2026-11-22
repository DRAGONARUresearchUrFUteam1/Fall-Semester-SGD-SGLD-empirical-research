"""The two active-manifold profiles described in issue #6."""

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SaddleBenchmark:
    """F_c, with kappa=lambda=mu=1 and the accepted diagnostic radii."""

    c: float

    def __post_init__(self) -> None:
        if type(self.c) not in (int, float) or self.c not in (0, 0.25):
            raise ValueError("The registered profiles use c=0 or c=0.25")

    @property
    def profile_id(self) -> str:
        return f"active_manifold_c{self.c:g}_v1"

    @staticmethod
    def point(x: Sequence[float]) -> tuple[float, float]:
        u, v = x
        u, v = float(u), float(v)
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError("Nonfinite state")
        return u, v

    def residual(self, u: float, v: float) -> float:
        return v - (self.c * u) * u

    def objective(self, x: Sequence[float]) -> float:
        u, v = self.point(x)
        return u ** 4 / 4 - u ** 2 / 2 + v ** 2 / 2 + abs(self.residual(u, v))

    def subgradient(self, x: Sequence[float]) -> tuple[float, float]:
        """Select sign(0)=0 at the kink; a prototype oracle convention."""
        u, v = self.point(x)
        residual = self.residual(u, v)
        sign = (residual > 0) - (residual < 0)
        return u ** 3 - u - 2 * self.c * u * sign, v + sign

    def minima_on_manifold(self) -> tuple[tuple[float, float], tuple[float, float]]:
        u = 1 / math.sqrt(1 + 2 * self.c ** 2)
        return (-u, self.c * u ** 2), (u, self.c * u ** 2)
