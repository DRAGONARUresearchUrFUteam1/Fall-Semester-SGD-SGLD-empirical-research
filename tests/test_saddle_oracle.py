import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.results import load_results
from src.saddle_benchmark import SaddleBenchmark
from src.saddle_oracle import PairedSaddleOracle
from src.train import expand_grid, pilot_components, run_one


class FixedBatch:
    def __init__(self, indices):
        self.indices = indices

    def choice(self, population_size, *, size, replace):
        assert not replace and size == len(self.indices)
        assert all(0 <= i < population_size for i in self.indices)
        return self.indices


@pytest.mark.parametrize("c", [0, 0.25])
@pytest.mark.parametrize("batch_size, variances", [
    (1, [2, 1/2]), (2, [2/3, 1/6]), (3, [2/9, 1/18]), (4, [0, 0]),
])
def test_enumerated_batches_match_analytic_reference(c, batch_size, variances):
    benchmark = SaddleBenchmark(c)
    oracle = PairedSaddleOracle(benchmark, [(2, 0), (0, 1)])
    assert oracle.population_size == 4
    for x in (np.array([0.125, c * 0.125**2]), np.array([0.125, 0.25])):
        draws = []
        for indices in combinations(range(4), batch_size):
            result = oracle.evaluate(x, batch_size, FixedBatch(indices))
            assert result.component_evals == batch_size
            draws.append(result.gradient)
        mean = np.mean(draws, axis=0)
        np.testing.assert_allclose(mean, benchmark.subgradient(x), atol=1e-15)
        residuals = draws - mean
        population_covariance = residuals.T @ residuals / len(draws)
        np.testing.assert_allclose(population_covariance, np.diag(variances), atol=1e-15)
        np.testing.assert_allclose(oracle.gradient_covariance(batch_size),
                                   population_covariance, atol=1e-15)
        assert oracle.gradient_covariance_trace(batch_size) == pytest.approx(sum(variances))


def test_full_batch_evaluates_every_component(monkeypatch):
    visited = []
    original = PairedSaddleOracle.component_gradient

    def record(self, x, index):
        visited.append(index)
        return original(self, x, index)

    monkeypatch.setattr(PairedSaddleOracle, "component_gradient", record)
    oracle = PairedSaddleOracle(SaddleBenchmark(0), [(2, 0), (0, 1)])
    result = oracle.evaluate(np.zeros(2), 4, FixedBatch(()))
    np.testing.assert_array_equal(result.gradient, [0, 0])
    assert result.component_evals == len(visited) == 4
    assert visited == [0, 1, 2, 3]


def test_rotation_keeps_trace_and_swaps_variances():
    benchmark = SaddleBenchmark(0)
    original = PairedSaddleOracle(benchmark, [(2, 0), (0, 1)])
    rotated = PairedSaddleOracle(benchmark, [(0, 2), (-1, 0)])
    np.testing.assert_allclose(rotated.gradient_covariance(2), np.diag([1/6, 2/3]))
    assert rotated.gradient_covariance_trace(2) == pytest.approx(
        original.gradient_covariance_trace(2)
    )


def fixture_config(filename):
    config = Path(__file__).resolve().parents[1] / "configs" / filename
    return expand_grid(yaml.safe_load(config.read_text(encoding="utf-8")))[0]


@pytest.mark.parametrize("override", [
    {"directions": []}, {"directions": [[1, 2]]}, {"directions": [[1, 2, 3], [4, 5, 6]]},
    {"directions": [[1, 0], [float("nan"), 1]]},
    {"directions": [[1, 0], [0, float("inf")]]},
    {"directions": [[1e300, 0], [0, 1e300]]},
    {"batch_size": 0}, {"batch_size": 5}, {"batch_size": 1.5}, {"batch_size": True},
    {"slope": [0, 1]},
])
def test_invalid_paired_configs(override):
    config = fixture_config("saddle_pilot_paired.yaml")
    with pytest.raises(ValueError):
        pilot_components({**config, **override})


def test_legacy_config_preserves_component_order_and_covariance():
    config = fixture_config("saddle_pilot.yaml")
    x, tracker, oracle, _ = pilot_components(config)
    slope = np.asarray(config["slope"])
    baseline = np.asarray(tracker.benchmark.subgradient(x))
    np.testing.assert_array_equal(oracle.component_gradient(x, 0), baseline + slope)
    np.testing.assert_array_equal(oracle.component_gradient(x, 1), baseline - slope)
    np.testing.assert_allclose(oracle.gradient_covariance(1), np.outer(slope, slope))
    np.testing.assert_array_equal(oracle.gradient_covariance(2), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="requires slope"):
        pilot_components({**config, "directions": [[1, 0], [0, 1]]})


@pytest.mark.parametrize("c", [0, 0.25])
def test_paired_config_is_repeatable_and_cost_round_trips(tmp_path, c):
    config = {**fixture_config("saddle_pilot_paired.yaml"), "c": c, "step_limit": 3}

    def read_fixture():
        run = run_one(config, tmp_path)
        metadata = json.loads((run / "config.json").read_text())["config"]
        records = [json.loads(line) for line in
                   (run / "saddle_measurements.jsonl").read_text().splitlines()]
        for record in records:
            del record["experiment_id"]
        return metadata, records, load_results(run)

    metadata, records, frame = read_fixture()
    assert records == read_fixture()[1]
    saved = metadata["oracle"]
    assert saved["kind"] == config["data_kind"]
    assert saved["directions"] == config["directions"]
    assert saved["population_size"] == 4
    assert saved["sampling"] == "uniform_without_replacement_per_step"
    assert saved["batch_gradient_covariance_source"] == "known_by_design"
    np.testing.assert_allclose(saved["batch_gradient_covariance"],
                               np.diag([0.12**2 / 6, 0.05**2 / 6]))
    assert metadata["cost_unit"] == "synthetic_component_subgradient"
    states = [record for record in records if record["kind"] == "state"]
    assert [record["cost"]["train"] for record in states] == [0, 2, 4, 6]
    assert all(record["cost"]["calibration"] == 0 for record in states)
    assert all(record["cost"]["diagnostics"] == 0 for record in states)
    assert all(record["profile_id"] == SaddleBenchmark(c).profile_id for record in states)
    assert records[-1]["events"]["good"]["evaluation_status"] == "not_configured"
    assert frame["iteration"].tolist() == [0, 1, 2, 3]
