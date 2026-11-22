"""Independent first detections for the accepted 2-D saddle profiles."""

import math
from dataclasses import asdict, dataclass, replace
from typing import Sequence

from .saddle_benchmark import SaddleBenchmark


@dataclass(frozen=True)
class CostSnapshot:
    """Counts by purpose, never a choice of scalar budget."""

    train: int = 0
    calibration: int = 0
    diagnostics: int = 0
    function_evals_train: int = 0
    function_evals_calibration: int = 0
    function_evals_diagnostics: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("Costs must be nonnegative integers")


@dataclass(frozen=True)
class FirstDetection:
    iteration: int
    x: tuple[float, float]
    objective: float | None
    cost: CostSnapshot
    boundary_reason: str | None = None


class EscapeTracker:
    """Sample every iterate; an invalid measurement never becomes a crossing."""

    def __init__(self, benchmark: SaddleBenchmark, *, delta: float | None = None):
        if delta is not None and (not math.isfinite(delta) or delta <= 0):
            raise ValueError("delta must be finite and positive")
        self.benchmark = benchmark
        self.delta = delta
        self._first: dict[str, FirstDetection | None] = dict.fromkeys(("geo", "tan", "good"))
        self._last_checked: dict[str, int | None] = dict.fromkeys(self._first)
        self._error: dict[str, str | None] = dict.fromkeys(self._first)
        self._last_iteration = -1
        self._last_cost = CostSnapshot()
        self._valid_start = False

    def observe(
        self, iteration: int, x: Sequence[float], cost: CostSnapshot
    ) -> dict:
        """Evaluate and charge one fresh objective call on a finite point.

        ``cost`` includes operations through the update, before this observation.
        The returned snapshot can be carried into the next update.
        """
        if type(iteration) is not int or iteration != self._last_iteration + 1:
            raise ValueError("Observe consecutive integer indices starting at zero")
        if any(
            value < previous
            for value, previous in zip(asdict(cost).values(), asdict(self._last_cost).values())
        ):
            raise ValueError("Cumulative cost cannot decrease")
        self._last_iteration = iteration
        enabled = ("geo", "tan") if self.delta is None else ("geo", "tan", "good")
        try:
            point = self.benchmark.point(x)
        except (TypeError, ValueError, OverflowError):
            self._last_cost = cost
            for key in enabled:
                self._error[key] = "invalid_coordinates"
            return self._record(iteration, None, cost, region_status="diagnostic_error",
                                region_error="invalid_coordinates")

        u, v = point
        tangent = abs(u) >= 0.25
        transverse = abs(self.benchmark.residual(u, v))
        normal = transverse >= 0.10
        reason = None
        if tangent:
            reason = "both" if normal else "tangent"
        elif normal:
            reason = "transverse"
        inside = reason is None

        cost = replace(cost, function_evals_diagnostics=cost.function_evals_diagnostics + 1)
        self._last_cost = cost
        try:
            objective = self.benchmark.objective(point)
        except OverflowError:
            objective = None
        if objective is not None and not math.isfinite(objective):
            objective = None

        if iteration == 0:
            self._valid_start = inside and objective is not None
        elif self._valid_start:
            conditions = {"geo": reason is not None, "tan": tangent}
            if self.delta is not None and objective is not None:
                # F_c(0, 0) = 0 for both profiles.
                conditions["good"] = tangent and objective <= -self.delta
            elif self.delta is not None:
                self._error["good"] = "nonfinite_objective"
            for key, observed in conditions.items():
                self._last_checked[key] = iteration
                if observed and self._first[key] is None:
                    self._first[key] = FirstDetection(
                        iteration, point, objective, cost, reason if key == "geo" else None
                    )

        return self._record(
            iteration, point, cost, objective=objective,
            objective_status="measured" if objective is not None else "nonfinite_objective",
            region_status="inside" if inside else "outside", boundary_reason=reason,
            tangent_coordinate_abs=abs(u),
            transverse_deviation=transverse if math.isfinite(transverse) else None,
            transverse_boundary_reached=normal, tangent_boundary_reached=tangent,
        )

    def _record(self, iteration, point, cost, *, objective=None,
                objective_status="not_evaluated", **fields) -> dict:
        return {
            "iteration": iteration, "x": point, "objective": objective,
            "objective_status": objective_status, "profile_id": self.benchmark.profile_id,
            "cost": asdict(cost), **fields,
        }

    def events(self) -> dict:
        events = {}
        for key, first in self._first.items():
            configured = key != "good" or self.delta is not None
            checked = self._last_checked[key]
            events[key] = {
                "evaluation_status": "enabled" if configured else "not_configured",
                "event_observed": True if first else False if checked is not None else None,
                "first_detection": asdict(first) if first is not None else None,
                "last_evaluated_iteration": checked,
                "evaluation_error": self._error[key],
                "reason": "delta_unset" if not configured else None,
            }
        return events
