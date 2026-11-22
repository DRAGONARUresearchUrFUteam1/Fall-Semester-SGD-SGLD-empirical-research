"""Local wiring for explicitly synthetic 2-D pilot configurations."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from .escape import CostSnapshot, EscapeTracker
from .experiment_logging import ExperimentLogger
from .optimizers import SGD, SGLD
from .saddle_benchmark import SaddleBenchmark
from .saddle_oracle import PairedSaddleOracle


METHODS = {"SGD": SGD, "SGLD": SGLD}
LEGACY_ORACLE = "synthetic_finite_sum_pilot"
PAIRED_ORACLE = "synthetic_paired_finite_sum_v1"


def expand_grid(spec: dict) -> list[dict]:
    """Pair the same seeds across the explicitly listed configurations."""
    if not isinstance(spec, dict) or set(spec) != {"common", "methods", "seeds"}:
        raise ValueError("Expected a configuration with common, methods and seeds")
    common, methods, seeds = spec["common"], spec["methods"], spec["seeds"]
    if not isinstance(common, dict) or {"method", "seed", "temperature"} & common.keys():
        raise ValueError("method, seed and temperature are not common settings")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Expected a nonempty list of nonnegative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds")
    if (not isinstance(methods, list) or not methods or
            any(not isinstance(m, dict) or
                set(m) - {"method", "temperature", "learning_rate", "batch_size"}
                or not isinstance(m.get("method"), str) or m["method"] not in METHODS
                for m in methods)):
        raise ValueError("Expected explicit SGD/SGLD method configurations")
    return [{**common, **settings, "seed": seed} for settings in methods for seed in seeds]


def build_pilot_oracle(config: dict, benchmark: SaddleBenchmark) -> PairedSaddleOracle:
    if config["data_kind"] == LEGACY_ORACLE:
        if "slope" not in config or "directions" in config:
            raise ValueError("The legacy pilot requires slope, not directions")
        return PairedSaddleOracle(benchmark, [config["slope"]])
    if config["data_kind"] == PAIRED_ORACLE:
        if "directions" not in config or "slope" in config:
            raise ValueError("The paired pilot requires directions, not slope")
        oracle = PairedSaddleOracle(benchmark, config["directions"])
        if len(oracle.directions) < 2:
            raise ValueError("The paired pilot requires at least two directions")
        return oracle
    raise ValueError("This runner only supports the synthetic pilots")


def pilot_components(config: dict):
    """Construct the chosen profile, synthetic finite sum and baseline method."""
    required = {"data_kind", "c", "x0", "step_limit", "method",
                "seed", "learning_rate", "batch_size"}
    optional = {"slope", "directions", "temperature", "delta"}
    if not required <= config.keys() or config.keys() - required - optional:
        raise ValueError("Missing or unrecognized pilot setting")
    if type(config["step_limit"]) is not int or config["step_limit"] < 1:
        raise ValueError("step_limit must be a positive integer")
    benchmark = SaddleBenchmark(config["c"])
    x = np.asarray(benchmark.point(config["x0"]))
    oracle = build_pilot_oracle(config, benchmark)
    b = config["batch_size"]
    if type(b) is not int or not 1 <= b <= oracle.population_size:
        raise ValueError("batch_size must be within the finite population")

    params = {key: config[key] for key in ("learning_rate", "batch_size", "seed")}
    method = config["method"]
    temperature = config.get("temperature")
    if method == "SGLD" and temperature is not None:
        optimizer = SGLD(temperature=temperature, **params)
    elif method == "SGD" and temperature is None:
        optimizer = SGD(**params)
    else:
        raise ValueError("A temperature is required for SGLD only")
    return x, EscapeTracker(benchmark, delta=config.get("delta")), oracle, optimizer


def _finite_or_none(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def run_one(config: dict, root: Path) -> Path:
    """Write v1 logs and a separate measurement journal for one pilot."""
    x, tracker, oracle, optimizer = pilot_components(config)
    # Serialize the proposed journal independently of the existing v1 logger.
    metadata = {
        **config, "profile_id": tracker.benchmark.profile_id,
        "cost_unit": "synthetic_component_subgradient", "measurement_schema_version": 1,
        "oracle": {
            "kind": config["data_kind"],
            "population_size": oracle.population_size,
            "directions": oracle.directions,
            "sampling": oracle.sampling,
            "batch_gradient_covariance_source": oracle.covariance_source,
            "batch_gradient_covariance": oracle.gradient_covariance(config["batch_size"]),
            "batch_gradient_covariance_trace": oracle.gradient_covariance_trace(config["batch_size"]),
        },
    }
    with ExperimentLogger(metadata, root) as logger, (
        logger.run_dir / "saddle_measurements.jsonl"
    ).open("x", encoding="utf-8") as journal:
        def emit(kind: str, **payload) -> None:
            row = {"measurement_schema_version": 1, "experiment_id": logger.experiment_id,
                   "kind": kind, **payload}
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()

        def observe(k, point, cost) -> dict:
            record = tracker.observe(k, point, cost)
            emit("state", **record)
            for kind, item in tracker.events().items():
                detection = item["first_detection"]
                if detection is not None and detection["iteration"] == k:
                    emit("first_detection", event_type=kind, **detection)
            # v1 escape_status is deliberately not repurposed as any of the three endpoints.
            logger.log(iteration=k, loss=record["objective"], gradient_norm=None,
                       parameters=record["x"])
            return record

        last = observe(0, x, CostSnapshot())
        cost = CostSnapshot(**last["cost"])
        last_valid = 0 if last["x"] is not None and last["objective"] is not None else None
        attempted = 0
        reason = "invalid_initial_state" if last["region_status"] != "inside" else "step_limit"
        if reason == "step_limit":
            for k in range(1, config["step_limit"] + 1):
                attempted = k
                step = optimizer.step(x, oracle)
                cost = replace(cost, train=cost.train + step.component_evals)
                x = step.x
                gradient = _finite_or_none(step.gradient)
                noise = _finite_or_none(step.added_noise)
                energy = step.expected_added_energy
                gnorm = math.hypot(*gradient) if None not in gradient else math.inf
                nnorm = math.hypot(*noise) if None not in noise else math.inf
                emit("transition", from_iteration=k - 1, to_iteration=k,
                     gradient_at_input=gradient, added_noise=noise,
                     gradient_norm_at_input=gnorm if math.isfinite(gnorm) else None,
                     added_noise_norm=nnorm if math.isfinite(nnorm) else None,
                     expected_added_energy=energy if math.isfinite(energy) else None,
                     component_evals=step.component_evals)
                last = observe(k, x, cost)
                cost = CostSnapshot(**last["cost"])
                if last["region_status"] == "diagnostic_error":
                    reason = "nonfinite_state"
                    break
                if last["objective"] is None:
                    reason = "nonfinite_objective"
                    break
                last_valid = k
        emit("termination", termination_reason=reason, last_valid_iteration=last_valid,
             attempted_iteration=attempted, cost=last["cost"], events=tracker.events())
    return logger.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/runs"))
    parser.add_argument("--plan", action="store_true", help="Validate and print, do not run")
    args = parser.parse_args()
    try:
        grid = expand_grid(yaml.safe_load(args.config.read_text(encoding="utf-8")))
        for config in grid:
            pilot_components(config)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.plan:
        print(json.dumps(grid, indent=2, allow_nan=False))
        return
    for config in grid:
        print(run_one(config, args.output))


if __name__ == "__main__":
    main()
