"""Load all run directories into one pandas table, optionally exporting CSV."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .experiment_logging import METRIC_COLUMNS, SCHEMA_VERSION, validate_config, validate_record

RESULT_COLUMNS = METRIC_COLUMNS + ["created_at", "config", "run_dir"]


def _reject_constant(value: str):
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def _read_json(text: str):
    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite JSON number: {value}")
        return number

    return json.loads(text, parse_constant=_reject_constant, parse_float=finite_float)


def load_results(root: str | Path = "results/runs") -> pd.DataFrame:
    """Recursively aggregate JSONL logs, retaining nested config/parameter values.

    root may be a single run directory or a collection of runs. Corrupt or
    incompatible records raise an error with a path/line instead of silently
    dropping evidence. Empty runs (config but no metrics yet) contribute no rows.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    rows = []
    seen_ids = set()
    # Include config-only runs to diagnose missing metric files as well.
    run_dirs = sorted({p.parent for name in ("config.json", "metrics.jsonl")
                       for p in root.rglob(name)})
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        try:
            metadata = _read_json(config_path.read_text(encoding="utf-8"))
            if (type(metadata["schema_version"]) is not int
                    or metadata["schema_version"] != SCHEMA_VERSION):
                raise ValueError("Unsupported schema_version")
            experiment_id = metadata["experiment_id"]
            if not isinstance(experiment_id, str) or not experiment_id:
                raise ValueError("experiment_id must be a nonempty string")
            if experiment_id in seen_ids:
                raise ValueError(f"Duplicate experiment_id: {experiment_id}")
            validate_config(metadata["config"])
            created_at = metadata["created_at"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{config_path}: {exc}") from exc
        seen_ids.add(experiment_id)
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            raise ValueError(f"{metrics_path}: missing metric file")
        last_iteration = -1
        with metrics_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = _read_json(line)
                    validate_record(record)
                    for key, expected in (("experiment_id", experiment_id),
                                          ("method", metadata["config"]["method"]),
                                          ("seed", metadata["config"]["seed"])):
                        if record[key] != expected:
                            raise ValueError(f"{key} does not match config.json")
                    if record["iteration"] <= last_iteration:
                        raise ValueError("iteration must be strictly increasing")
                    last_iteration = record["iteration"]
                    rows.append({**record, "created_at": created_at,
                                 "config": metadata["config"], "run_dir": str(run_dir)})
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise ValueError(f"{metrics_path}:{line_number}: {exc}") from exc
    # Prevent pandas from turning scalar parameter snapshots + None into floats
    # + NaN: parameters must retain their original JSON types for CSV roundtrips.
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS, dtype=object)
    # Keep integer identifiers exact (including full-width torch seeds). Use
    # object only for integers outside pandas' signed/unsigned 64-bit range.
    for column in ("schema_version", "seed", "batch_size", "iteration"):
        values = [row[column] for row in rows]
        if all(value <= 2**63 - 1 for value in values):
            dtype = "Int64"
        elif all(value <= 2**64 - 1 for value in values):
            dtype = "UInt64"
        else:
            dtype = object
        frame[column] = pd.Series(values, dtype=dtype)
    for column in ("learning_rate", "temperature", "loss", "gradient_norm"):
        frame[column] = frame[column].astype("Float64")
    return frame


def save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Encode nested columns as JSON strings, not Python repr, inside the CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in ("parameters", "config"):
        output[column] = output[column].map(
            lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="results/runs")
    parser.add_argument("--output", type=Path, help="Aggregated CSV output path")
    args = parser.parse_args()
    frame = load_results(args.root)
    print(f"Loaded {len(frame)} records from {frame['experiment_id'].nunique()} runs")
    if args.output:
        save_csv(frame, args.output)
        print(f"Saved {args.output}")
    else:
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
