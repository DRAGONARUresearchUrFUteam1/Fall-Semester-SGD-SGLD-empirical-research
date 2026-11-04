import json
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from src.experiment_logging import ExperimentLogger, METRIC_COLUMNS
from src.logging_example import write_example
from src.results import RESULT_COLUMNS, load_results, save_csv


@pytest.fixture
def config():
    return {"method": "SGD", "seed": 42, "learning_rate": 0.01,
            "batch_size": 32, "task": {"name": "toy", "dimension": 2}}


def log_step(logger, **overrides):
    values = dict(iteration=0, loss=1.0, gradient_norm=0.5,
                  parameters={"x": [0.1, 0.2]}, escape_status="not_escaped")
    logger.log(**(values | overrides))


def test_config_snapshot_and_unique_ids(tmp_path, config):
    with ExperimentLogger(config, tmp_path) as first:
        config["task"]["name"] = "changed"
        log_step(first)
        UUID(first.experiment_id)
        metadata = json.loads((first.run_dir / "config.json").read_text())
        assert metadata["config"]["task"]["name"] == "toy"
        assert metadata["config"]["temperature"] is None
        assert metadata["experiment_id"] == first.experiment_id
        assert metadata["created_at"]
        # Records are visible before close(), not buffered until the run ends.
        record = json.loads((first.run_dir / "metrics.jsonl").read_text())
        assert set(record) == set(METRIC_COLUMNS)
    with ExperimentLogger(config, tmp_path) as second:
        assert second.experiment_id != first.experiment_id
    with pytest.raises(ValueError, match="closed"):
        log_step(first, iteration=1)


def test_numpy_serialization_and_actual_schedule(tmp_path, config):
    config["seed"] = np.int64(42)
    config["temperature"] = 0.1
    with ExperimentLogger(config, tmp_path) as logger:
        log_step(logger, iteration=np.int64(0), loss=np.float32(0.5),
                 parameters={"x": np.array([1.0, 2.0])})
        log_step(logger, iteration=10, learning_rate=0.001, batch_size=64,
                 temperature=0.0, parameters=None, escape_status="escaped")
        log_step(logger, iteration=20, temperature=None)
    frame = load_results(tmp_path)
    assert frame["parameters"][0] == {"x": [1.0, 2.0]}
    assert frame["learning_rate"][1] == 0.001
    assert frame["batch_size"][1] == 64
    assert frame["temperature"][1] == 0.0
    assert pd.isna(frame["temperature"][2])
    assert frame["config"][1]["learning_rate"] == 0.01


@pytest.mark.parametrize("field,value", [
    ("method", ""), ("seed", -1), ("seed", True), ("batch_size", 0),
    ("batch_size", 2.5), ("learning_rate", -0.1), ("learning_rate", float("nan")),
    ("temperature", -1), ("temperature", float("inf")),
])
def test_invalid_config_leaves_no_run(tmp_path, config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        ExperimentLogger(config, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_config_field(tmp_path, config):
    del config["seed"]
    with pytest.raises(ValueError, match="seed"):
        ExperimentLogger(config, tmp_path)


@pytest.mark.parametrize("overrides", [
    {"iteration": -1}, {"iteration": True}, {"loss": float("nan")},
    {"gradient_norm": float("inf")}, {"gradient_norm": -1},
    {"escape_status": "unknown"}, {"parameters": [float("nan")]},
    {"learning_rate": -1}, {"batch_size": 0}, {"temperature": -0.1},
])
def test_invalid_record_is_not_appended(tmp_path, config, overrides):
    with ExperimentLogger(config, tmp_path) as logger:
        with pytest.raises(ValueError):
            log_step(logger, **overrides)
        assert (logger.run_dir / "metrics.jsonl").read_text() == ""
        log_step(logger)  # Invalid attempts must not consume the iteration.
    assert len(load_results(tmp_path)) == 1


def test_iteration_order_and_null_measurements(tmp_path, config):
    with ExperimentLogger(config, tmp_path) as logger:
        log_step(logger, iteration=2, loss=None, gradient_norm=None)
        with pytest.raises(ValueError, match="strictly increasing"):
            log_step(logger, iteration=2)
        with pytest.raises(ValueError, match="strictly increasing"):
            log_step(logger, iteration=1)
    frame = load_results(tmp_path)
    assert pd.isna(frame["loss"][0])
    assert pd.isna(frame["gradient_norm"][0])


def test_multiple_runs_recursive_aggregation_and_csv(tmp_path, config):
    with ExperimentLogger(config, tmp_path / "round1") as first:
        log_step(first)
        log_step(first, iteration=1, escape_status="escaped")
    config.update(method="SGLD", seed=43, temperature=0.002)
    with ExperimentLogger(config, tmp_path / "round2") as second:
        log_step(second)
    frame = load_results(tmp_path)
    assert len(frame) == 3
    assert frame["experiment_id"].nunique() == 2
    assert set(frame["method"]) == {"SGD", "SGLD"}
    assert set(frame.columns) == set(RESULT_COLUMNS)
    assert frame.groupby("experiment_id").size().sort_values().tolist() == [1, 2]
    assert len(load_results(first.run_dir)) == 2
    path = tmp_path / "export" / "all.csv"
    save_csv(frame, path)
    reloaded = pd.read_csv(path)
    assert len(reloaded) == 3
    assert json.loads(reloaded["parameters"][0]) == {"x": [0.1, 0.2]}
    assert json.loads(reloaded["config"][0])["task"]["name"] == "toy"


def test_empty_collection_and_empty_run(tmp_path, config):
    empty = load_results(tmp_path)
    assert empty.empty
    assert list(empty.columns) == RESULT_COLUMNS
    with ExperimentLogger(config, tmp_path):
        pass
    assert load_results(tmp_path).empty


def test_scalar_and_missing_parameter_snapshots_csv(tmp_path, config):
    with ExperimentLogger(config, tmp_path) as logger:
        log_step(logger, parameters=0.5)
        log_step(logger, iteration=1, parameters=None)
    frame = load_results(tmp_path)
    assert frame["parameters"].tolist() == [0.5, None]
    save_csv(frame, tmp_path / "scalar.csv")
    reloaded = pd.read_csv(tmp_path / "scalar.csv", keep_default_na=False)
    assert [json.loads(value) for value in reloaded["parameters"]] == [0.5, None]


@pytest.mark.parametrize("seed", [2**63 - 1, 2**64 - 1, 2**80])
def test_large_seeds_remain_exact(tmp_path, config, seed):
    config["seed"] = seed
    with ExperimentLogger(config, tmp_path) as logger:
        log_step(logger)
    assert load_results(tmp_path)["seed"][0] == seed


def test_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing")


def test_context_manager_preserves_logs_on_exception(tmp_path, config):
    with pytest.raises(RuntimeError):
        with ExperimentLogger(config, tmp_path) as logger:
            log_step(logger)
            raise RuntimeError("training failed")
    assert len(load_results(tmp_path)) == 1
    with pytest.raises(ValueError, match="closed"):
        log_step(logger, iteration=1)


@pytest.mark.parametrize("corruption", [
    "truncated", "missing_field", "version", "id", "method", "seed", "nan", "iteration",
])
def test_bad_metrics_are_reported_with_path_and_line(tmp_path, config, corruption):
    with ExperimentLogger(config, tmp_path) as logger:
        log_step(logger)
    path = logger.run_dir / "metrics.jsonl"
    record = json.loads(path.read_text())
    if corruption == "truncated":
        text = '{"iteration":'
    else:
        if corruption == "missing_field":
            del record["loss"]
        elif corruption == "version":
            record["schema_version"] = 99
        elif corruption == "id":
            record["experiment_id"] = "wrong"
        elif corruption == "method":
            record["method"] = "other"
        elif corruption == "seed":
            record["seed"] = 999
        elif corruption == "nan":
            record["loss"] = float("nan")
        text = json.dumps(record) + "\n"
        if corruption == "iteration":
            text += text
    path.write_text(text)
    with pytest.raises(ValueError, match=r"metrics\.jsonl:[12]:"):
        load_results(tmp_path)


def test_missing_config_and_missing_metrics(tmp_path, config):
    with ExperimentLogger(config, tmp_path) as logger:
        log_step(logger)
    metadata_path = logger.run_dir / "config.json"
    metadata = metadata_path.read_text()
    metadata_path.unlink()
    with pytest.raises(ValueError, match=r"config\.json"):
        load_results(tmp_path)
    metadata_path.write_text(metadata)
    (logger.run_dir / "metrics.jsonl").unlink()
    with pytest.raises(ValueError, match="missing metric file"):
        load_results(tmp_path)


def test_duplicate_ids_are_not_silently_double_counted(tmp_path, config):
    with ExperimentLogger(config, tmp_path) as first:
        log_step(first)
    with ExperimentLogger(config, tmp_path) as second:
        log_step(second)
    (second.run_dir / "config.json").write_text(
        (first.run_dir / "config.json").read_text()
    )
    with pytest.raises(ValueError, match="Duplicate experiment_id|does not match"):
        load_results(tmp_path)


def test_generated_and_saved_synthetic_examples(tmp_path):
    write_example(tmp_path)
    generated = load_results(tmp_path)
    assert generated["iteration"].tolist() == [0, 1, 2]
    assert generated["escape_status"].tolist() == ["not_escaped", "not_escaped", "escaped"]
    saved = load_results("docs/examples/logging")
    assert len(saved) == 3
    assert saved["config"][0]["data_kind"] == "synthetic_logging_demo"


def test_torch_parameters_if_installed(tmp_path, config):
    torch = pytest.importorskip("torch")
    with ExperimentLogger(config, tmp_path) as logger:
        log_step(logger, parameters={"weight": torch.tensor([1.0, 2.0], requires_grad=True)})
    assert load_results(tmp_path)["parameters"][0] == {"weight": [1.0, 2.0]}
