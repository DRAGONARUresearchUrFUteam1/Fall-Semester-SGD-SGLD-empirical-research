import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.escape import CostSnapshot, EscapeTracker
from src.noise import estimate_noise, sample_noise
from src.optimizers import FiniteSumOracle, SGD, SGLD
from src.results import load_results
from src.saddle_benchmark import SaddleBenchmark
from src.train import expand_grid, run_one


@pytest.mark.parametrize("c", [0.0, 0.25])
@pytest.mark.parametrize("point", [(0.11, 0.18), (-0.35, -0.15)])
def test_subgradient_off_kink_matches_centered_differences(c, point):
    task = SaddleBenchmark(c)
    x = np.asarray(point)
    h = 1e-6
    numerical = [(task.objective(x+h*e) - task.objective(x-h*e))/(2*h) for e in np.eye(2)]
    np.testing.assert_allclose(task.subgradient(x), numerical, atol=1e-9, rtol=0)


@pytest.mark.parametrize("c", [0.0, 0.25])
def test_saddle_and_kink(c):
    task = SaddleBenchmark(c)
    assert task.objective((0, 0)) == 0
    assert task.subgradient((0, 0)) == (0, 0)
    assert task.objective((0, 0.01)) > 0 > task.objective((0.1, c * 0.1 ** 2))
    for point in task.minima_on_manifold():
        assert task.objective(point) == pytest.approx(-1 / (4 * (1 + 2*c*c)))
    u = 0.25
    v = c * u ** 2
    # At the kink we select s=0, the midpoint of the one-sided subgradients.
    np.testing.assert_allclose(task.subgradient((u, v)),
                               (np.asarray(task.subgradient((u, v+1e-6))) +
                                task.subgradient((u, v-1e-6))) / 2)


def trajectory(tracker, *points):
    cost = CostSnapshot()
    records = []
    for k, point in enumerate(points):
        record = tracker.observe(k, point, replace(cost, train=k))
        records.append(record)
        cost = CostSnapshot(**record["cost"])
    return records


def test_escapes_are_first_hits_even_after_return():
    tracker = EscapeTracker(SaddleBenchmark(0.0), delta=0.02)
    rows = trajectory(tracker, (0, 0), (0, 0.11), (0, 0), (0.25, 0.10), (0.3, 0))
    assert rows[2]["region_status"] == "inside"
    events = tracker.events()
    assert [events[name]["first_detection"]["iteration"] for name in ("geo", "tan", "good")] == [1, 3, 4]
    geo = events["geo"]["first_detection"]
    assert geo["boundary_reason"] == "transverse"
    assert geo["cost"]["train"] == 1
    assert geo["cost"]["function_evals_diagnostics"] == 2


def test_equal_boundaries_and_unconfigured_good():
    tracker = EscapeTracker(SaddleBenchmark(0.0))
    rows = trajectory(tracker, (0, 0), (0.25, 0.10))
    assert rows[1]["region_status"] == "outside"
    events = tracker.events()
    assert events["geo"]["first_detection"]["boundary_reason"] == "both"
    assert events["tan"]["first_detection"]["iteration"] == 1
    assert events["good"]["evaluation_status"] == "not_configured"
    assert events["good"]["event_observed"] is None


def test_curvature_of_kink_moves_membership():
    a = EscapeTracker(SaddleBenchmark(0.0))
    b = EscapeTracker(SaddleBenchmark(0.25))
    point = (0.24, 0.11)
    assert trajectory(a, (0, 0), point)[-1]["region_status"] == "outside"
    assert trajectory(b, (0, 0), point)[-1]["region_status"] == "inside"


def test_no_event_before_error_is_only_observed_through_last_valid_step():
    tracker = EscapeTracker(SaddleBenchmark(0.0), delta=0.02)
    rows = trajectory(tracker, (0, 0), (0.10, 0.05), (float("nan"), 0))
    assert rows[-1]["region_status"] == "diagnostic_error"
    for event in tracker.events().values():
        assert event["event_observed"] is False
        assert event["first_detection"] is None
        assert event["last_evaluated_iteration"] == 1
        assert event["evaluation_error"] == "invalid_coordinates"
    assert rows[-1]["cost"]["train"] == 2
    assert rows[-1]["cost"]["function_evals_diagnostics"] == 2


def test_invalid_start_and_omitted_check_are_not_silent():
    tracker = EscapeTracker(SaddleBenchmark(0))
    trajectory(tracker, (0.25, 0), (0.4, 0))
    assert tracker.events()["geo"]["event_observed"] is None
    with pytest.raises(ValueError, match="consecutive"):
        tracker.observe(3, (0, 0), CostSnapshot())


def test_covariance_scale_spectrum_and_projection():
    gradients = np.array([[1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, -2, 0]])
    estimate = estimate_noise(gradients, unstable_basis=[[1], [0], [0]], materialize=True)
    np.testing.assert_allclose(estimate.covariance, np.diag([2/3, 8/3, 0]))
    np.testing.assert_allclose(estimate.eigenvalues, [8/3, 2/3, 0])
    assert estimate.trace == pytest.approx(10/3)
    assert estimate.unstable_fraction == pytest.approx(0.2)
    assert estimate.rank_participation == pytest.approx(1 / (0.8**2 + 0.2**2))
    assert estimate.rank_entropy == pytest.approx(np.exp(-0.8*np.log(0.8)-0.2*np.log(0.2)))


def test_zero_noise_and_bad_basis():
    gradients = np.repeat([[0.1, 0.3, 0.7]], 7, axis=0)
    estimate = estimate_noise(gradients, unstable_basis=[[1], [0], [0]])
    assert estimate.trace == estimate.unstable_power == 0
    assert estimate.covariance is None
    assert estimate.rank_entropy is estimate.rank_participation is estimate.unstable_fraction is None
    with pytest.raises(ValueError, match="orthonormal"):
        estimate_noise(gradients, unstable_basis=[[2], [0], [0]])


def test_oracle_sampling_measurements_share_point_and_report_cost():
    calls = []

    def gradient(x, i):
        calls.append((x.copy(), i))
        return np.array([i, -i])

    estimate, cost = sample_noise(FiniteSumOracle(4, gradient), [1, 2],
                                  batch_size=2, sample_count=8, rng=np.random.default_rng(7))
    assert (estimate.sample_count, estimate.dimension, cost) == (8, 2, 16)
    assert all(np.array_equal(x, [1, 2]) for x, _ in calls)
    assert all(calls[i][1] != calls[i+1][1] for i in range(0, len(calls), 2))
    assert estimate.covariance is None


def test_sgd_converges_on_convex_quadratic():
    oracle = FiniteSumOracle(2, lambda x, i: (i + 1) * x)
    optimizer = SGD(learning_rate=0.1, batch_size=1, seed=11)
    x = np.array([2.0, -1.0])
    for _ in range(100):
        step = optimizer.step(x, oracle)
        assert step.component_evals == 1
        assert step.x @ step.x < x @ x
        x = step.x
    assert np.linalg.norm(x) < 1e-4


def test_sgld_zero_temperature_matches_sgd_and_seeds_reproduce():
    oracle = FiniteSumOracle(2, lambda x, i: x + (1 if i == 0 else -1))

    def simulate(method):
        points = [np.array([0.0, 0.0])]
        for _ in range(8):
            points.append(method.step(points[-1], oracle).x)
        return points

    params = {"learning_rate": 0.01, "batch_size": 1, "seed": 3}
    sgd = simulate(SGD(**params))
    np.testing.assert_array_equal(sgd, simulate(SGLD(temperature=0, **params)))
    noisy = simulate(SGLD(temperature=0.1, **params))
    np.testing.assert_array_equal(noisy, simulate(SGLD(temperature=0.1, **params)))
    assert not np.array_equal(sgd, noisy)
    assert not np.array_equal(noisy, simulate(SGLD(temperature=0.1, **{**params, "seed": 4})))
    step = SGLD(temperature=0.1, **params).step(np.zeros(2), oracle)
    assert step.expected_added_energy == pytest.approx(0.004)


def test_grid_and_measurement_journal(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "saddle_pilot.yaml"
    configs = expand_grid(yaml.safe_load(config_path.read_text()))
    assert [(config["method"], config["seed"]) for config in configs] == [
        ("SGD", 11), ("SGD", 23), ("SGLD", 11), ("SGLD", 23)
    ]
    config = {**configs[0], "step_limit": 2}
    run_dir = run_one(config, tmp_path)
    rows = [json.loads(line) for line in (run_dir / "saddle_measurements.jsonl").read_text().splitlines()]
    states = [row for row in rows if row["kind"] == "state"]
    assert [row["iteration"] for row in states] == [0, 1, 2]
    assert [row["cost"]["train"] for row in states] == [0, 1, 2]
    assert [row["cost"]["function_evals_diagnostics"] for row in states] == [1, 2, 3]
    assert rows[-1]["kind"] == "termination"
    assert rows[-1]["termination_reason"] == "step_limit"
    assert rows[-1]["events"]["good"]["event_observed"] is None
    frame = load_results(run_dir)
    assert frame["iteration"].tolist() == [0, 1, 2]
    assert frame["escape_status"].tolist() == ["not_evaluated"] * 3


def test_boundary_detection_round_trips_through_journal(tmp_path):
    grid = expand_grid(yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "saddle_pilot.yaml").read_text()
    ))
    # One deterministic full-population update from an interior point.
    config = {**grid[0], "x0": [0.24, 0], "batch_size": 2,
              "learning_rate": 0.1, "step_limit": 1}
    journal = run_one(config, tmp_path) / "saddle_measurements.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    detections = {row["event_type"]: row for row in rows if row["kind"] == "first_detection"}
    assert set(detections) == {"geo", "tan"}
    assert detections["geo"]["boundary_reason"] == "tangent"
    assert detections["tan"]["cost"]["train"] == 2
    assert detections["tan"]["cost"]["function_evals_diagnostics"] == 2
    assert rows[-1]["events"]["geo"]["first_detection"]["iteration"] == 1
