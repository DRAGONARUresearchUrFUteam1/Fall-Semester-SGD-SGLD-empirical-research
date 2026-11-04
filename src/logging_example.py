"""Write synthetic demo records to exercise the logging API (not ML evidence)."""

import argparse
from pathlib import Path

from .experiment_logging import ExperimentLogger


def write_example(root: str | Path = "results/runs") -> Path:
    config = {
        "method": "SGLD", "seed": 42, "learning_rate": 0.01,
        "batch_size": 32, "temperature": 0.001,
        "data_kind": "synthetic_logging_demo",
        "description": "Hand-written values for format demonstration; not research results.",
        "escape_rule": "Synthetic status transition for API demonstration only",
    }
    with ExperimentLogger(config, root) as logger:
        for iteration, loss, norm, point, status in (
            (0, 1.0, 0.2, [0.0, 0.0], "not_escaped"),
            (1, 0.9, 0.3, [0.1, 0.0], "not_escaped"),
            (2, 0.7, 0.4, [0.3, 0.1], "escaped"),
        ):
            logger.log(iteration=iteration, loss=loss, gradient_norm=norm,
                       parameters=point, escape_status=status)
        return logger.run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    args = parser.parse_args()
    print(write_example(args.output_dir))


if __name__ == "__main__":
    main()
