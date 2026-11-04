"""Append-only, versioned JSONL logs; no pandas/torch dependency required."""

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 1
ESCAPE_STATUSES = frozenset({"not_evaluated", "not_escaped", "escaped"})
METRIC_COLUMNS = [
    "schema_version", "experiment_id", "method", "seed", "learning_rate",
    "batch_size", "temperature", "iteration", "loss", "gradient_norm",
    "parameters", "escape_status", "logged_at",
]
_UNSET = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    # Tensor.detach() prevents graph retention; cpu() also supports CUDA tensors.
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _json_copy(value: Any) -> Any:
    """Snapshot arrays/tensors and reject NaN/Infinity (not valid JSON)."""
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _integer(name: str, value: Any, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: Any, *, nullable: bool = False,
            nonnegative: bool = False) -> None:
    if value is None and nullable:
        return
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError(f"{name} must be finite" + (" and >= 0" if nonnegative else ""))


def validate_config(config: dict) -> None:
    """Validate the common fields; arbitrary JSON-compatible extras are allowed."""
    required = {"method", "seed", "learning_rate", "batch_size", "temperature"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(sorted(missing))}")
    if not isinstance(config["method"], str) or not config["method"].strip():
        raise ValueError("method must be a nonempty string")
    _integer("seed", config["seed"])
    _integer("batch_size", config["batch_size"], 1)
    _number("learning_rate", config["learning_rate"], nonnegative=True)
    _number("temperature", config["temperature"], nullable=True, nonnegative=True)


def validate_record(record: dict) -> None:
    missing = set(METRIC_COLUMNS) - record.keys()
    if missing:
        raise ValueError(f"Missing metric fields: {', '.join(sorted(missing))}")
    if type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {record['schema_version']!r}")
    validate_config(record)
    _integer("iteration", record["iteration"])
    _number("loss", record["loss"], nullable=True)
    _number("gradient_norm", record["gradient_norm"], nullable=True, nonnegative=True)
    if (not isinstance(record["escape_status"], str)
            or record["escape_status"] not in ESCAPE_STATUSES):
        raise ValueError(f"escape_status must be one of {sorted(ESCAPE_STATUSES)}")


class ExperimentLogger:
    """One new UUID directory per invocation; write and flush each iteration.

    Required config: method, seed, learning_rate, batch_size. Temperature defaults
    to None (not applicable). Keep additional task/optimizer/escape definitions in
    config. Use the context manager, or call close() explicitly. A logger has one
    writer; parallel processes must each create their own logger.
    """

    def __init__(self, config: Mapping[str, Any], root: str | Path = "results/runs"):
        self._config = _json_copy(dict(config))
        self._config.setdefault("temperature", None)
        validate_config(self._config)
        self.experiment_id = str(uuid4())
        self.run_dir = Path(root) / self.experiment_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "created_at": _now(),
            "config": self._config,
        }
        with (self.run_dir / "config.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        self._stream = (self.run_dir / "metrics.jsonl").open("x", encoding="utf-8")
        self._last_iteration = -1

    def log(self, *, iteration: int, loss: float | None,
            gradient_norm: float | None, parameters: Any,
            escape_status: str = "not_evaluated",
            learning_rate: Any = _UNSET, batch_size: Any = _UNSET,
            temperature: Any = _UNSET) -> None:
        """Record actual per-step hyperparameters, overriding config defaults.

        parameters contains current optimization variables/weights, not optimizer
        hyperparameters. Pass None explicitly to omit a large weight snapshot.
        None for loss/norm denotes an unavailable measurement, never zero.
        """
        if self._stream.closed:
            raise ValueError("Logger is closed")
        record = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "method": self._config["method"],
            "seed": self._config["seed"],
            "learning_rate": self._config["learning_rate"] if learning_rate is _UNSET else learning_rate,
            "batch_size": self._config["batch_size"] if batch_size is _UNSET else batch_size,
            "temperature": self._config["temperature"] if temperature is _UNSET else temperature,
            "iteration": iteration,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "parameters": parameters,
            "escape_status": escape_status,
            "logged_at": _now(),
        }
        record = _json_copy(record)
        validate_record(record)
        if record["iteration"] <= self._last_iteration:
            raise ValueError("iteration must be strictly increasing within a run")
        self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._stream.flush()
        self._last_iteration = record["iteration"]

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
