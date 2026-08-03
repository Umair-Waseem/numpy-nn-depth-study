import argparse

import numpy as np
import pytest

from experiments.run_depth_study import (
    ARCHITECTURES,
    SEEDS,
    compare_to_baseline,
    positive_float,
    positive_int,
)
from experiments.tune_learning_rate import (
    LEARNING_RATES,
    SWEEP_ARCHITECTURES,
    format_summary,
    has_collapsed,
    summarise,
)
from src.activations import relu, relu_backward, sigmoid, sigmoid_backward
from src.backward import L_model_backward, linear_backward
from src.cost import compute_cost
from src.forward import L_model_forward, linear_activation_forward, linear_forward
from src.initialization import initialize_parameters_deep
from src.model import L_layer_model, accuracy, predict
from src.update import update_parameters

LAYER_DIMS = [8, 5, 3, 1]
EXAMPLES = 6


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((LAYER_DIMS[0], EXAMPLES))
    Y = (rng.standard_normal((1, EXAMPLES)) > 0).astype(float)
    return X, Y


@pytest.fixture
def parameters():
    return initialize_parameters_deep(LAYER_DIMS, seed=1)


def build_records(baseline_scores, candidate_scores):
    # Every architecture needs records, since the comparison looks all of them up.
    records = []
    for index, name in enumerate(ARCHITECTURES):
        scores = candidate_scores if index == 1 else baseline_scores
        for seed, score in zip(SEEDS, scores):
            records.append({"architecture": name, "seed": seed, "test_accuracy": score})
    return records


def test_sigmoid_maps_to_unit_interval():
    Z = np.array([[-50.0, 0.0, 50.0]])
    A, cache = sigmoid(Z)
    assert np.all((A >= 0) & (A <= 1))
    assert A[0, 1] == pytest.approx(0.5)
    assert np.array_equal(cache, Z)


def test_relu_clips_negative_values():
    A, _ = relu(np.array([[-2.0, 0.0, 3.0]]))
    assert np.array_equal(A, np.array([[0.0, 0.0, 3.0]]))


def test_activation_backward_preserves_shape():
    Z = np.array([[-1.0, 2.0], [3.0, -4.0]])
    dA = np.ones_like(Z)
    assert sigmoid_backward(dA, Z).shape == Z.shape
    assert relu_backward(dA, Z).shape == Z.shape


def test_relu_backward_blocks_inactive_units():
    dZ = relu_backward(np.array([[5.0, 5.0]]), np.array([[-1.0, 2.0]]))
    assert np.array_equal(dZ, np.array([[0.0, 5.0]]))


def test_initialisation_shapes_and_zero_biases(parameters):
    for l in range(1, len(LAYER_DIMS)):
        assert parameters[f"W{l}"].shape == (LAYER_DIMS[l], LAYER_DIMS[l - 1])
        assert parameters[f"b{l}"].shape == (LAYER_DIMS[l], 1)
        assert np.all(parameters[f"b{l}"] == 0)


def test_initialisation_is_reproducible():
    first = initialize_parameters_deep(LAYER_DIMS, seed=7)
    second = initialize_parameters_deep(LAYER_DIMS, seed=7)
    assert np.array_equal(first["W1"], second["W1"])


def test_linear_forward_shape(dataset, parameters):
    X, _ = dataset
    Z, cache = linear_forward(X, parameters["W1"], parameters["b1"])
    assert Z.shape == (LAYER_DIMS[1], EXAMPLES)
    assert len(cache) == 3


def test_forward_output_is_a_probability_row(dataset, parameters):
    X, _ = dataset
    AL, caches = L_model_forward(X, parameters)
    assert AL.shape == (1, EXAMPLES)
    assert np.all((AL > 0) & (AL < 1))
    assert len(caches) == len(LAYER_DIMS) - 1


def test_unknown_activation_is_rejected(dataset, parameters):
    X, _ = dataset
    with pytest.raises(ValueError):
        linear_activation_forward(X, parameters["W1"], parameters["b1"], "tanh")


def test_cost_is_a_positive_scalar(dataset, parameters):
    X, Y = dataset
    cost = compute_cost(L_model_forward(X, parameters)[0], Y)
    assert isinstance(cost, float)
    assert cost > 0


def test_cost_is_near_zero_for_perfect_predictions():
    Y = np.array([[1.0, 0.0, 1.0]])
    assert compute_cost(Y, Y) == pytest.approx(0.0, abs=1e-9)


def test_linear_backward_matches_parameter_shapes(dataset, parameters):
    X, _ = dataset
    _, cache = linear_forward(X, parameters["W1"], parameters["b1"])
    dA_prev, dW, db = linear_backward(np.ones((LAYER_DIMS[1], EXAMPLES)), cache)
    assert dA_prev.shape == X.shape
    assert dW.shape == parameters["W1"].shape
    assert db.shape == parameters["b1"].shape


def test_every_gradient_matches_its_parameter(dataset, parameters):
    X, Y = dataset
    AL, caches = L_model_forward(X, parameters)
    grads = L_model_backward(AL, Y, caches)
    for l in range(1, len(LAYER_DIMS)):
        assert grads[f"dW{l}"].shape == parameters[f"W{l}"].shape
        assert grads[f"db{l}"].shape == parameters[f"b{l}"].shape


def test_update_leaves_the_original_parameters_untouched(dataset, parameters):
    X, Y = dataset
    AL, caches = L_model_forward(X, parameters)
    grads = L_model_backward(AL, Y, caches)
    before = parameters["W1"].copy()
    updated = update_parameters(parameters, grads, learning_rate=0.1)
    assert np.array_equal(parameters["W1"], before)
    assert not np.array_equal(updated["W1"], before)


def test_predictions_are_binary_and_scored(dataset, parameters):
    X, Y = dataset
    predictions = predict(X, parameters)
    assert predictions.shape == Y.shape
    assert np.all(np.isin(predictions, [0.0, 1.0]))
    assert 0.0 <= accuracy(predictions, Y) <= 100.0


def test_training_reduces_the_cost(dataset):
    X, Y = dataset
    _, history = L_layer_model(X, Y, LAYER_DIMS, num_iterations=500, record_every=100)
    assert history[-1][1] < history[0][1]


def test_history_ends_on_the_returned_parameters(dataset):
    X, Y = dataset
    iterations = 300
    parameters, history = L_layer_model(
        X, Y, LAYER_DIMS, num_iterations=iterations, record_every=100
    )
    assert history[0][0] == 0
    assert history[-1][0] == iterations
    scored = compute_cost(L_model_forward(X, parameters)[0], Y)
    assert history[-1][1] == pytest.approx(scored)


def test_training_is_reproducible(dataset):
    X, Y = dataset
    first, _ = L_layer_model(X, Y, LAYER_DIMS, num_iterations=50, seed=3)
    second, _ = L_layer_model(X, Y, LAYER_DIMS, num_iterations=50, seed=3)
    assert np.array_equal(first["W1"], second["W1"])


def test_comparison_reports_the_mean_difference():
    records = build_records([80.0] * len(SEEDS), [81.0] * len(SEEDS))
    assert compare_to_baseline(records)[0]["mean_difference"] == pytest.approx(1.0)


def test_a_consistent_gain_is_significant():
    baseline = [80.0, 80.5, 79.5, 80.2, 79.8][: len(SEEDS)]
    candidate = [score + 1.0 for score in baseline]
    row = compare_to_baseline(build_records(baseline, candidate))[0]
    assert row["ci_low"] > 0
    assert row["significant"] == "yes"


def test_a_noisy_gain_is_not_significant():
    baseline = [80.0] * len(SEEDS)
    candidate = [82.0, 78.0, 81.0, 79.0, 80.5][: len(SEEDS)]
    row = compare_to_baseline(build_records(baseline, candidate))[0]
    assert row["ci_low"] < 0 < row["ci_high"]
    assert row["significant"] == "no"


def test_positive_int_accepts_valid_counts():
    assert positive_int("1") == 1
    assert positive_int("2500") == 2500


def test_positive_int_rejects_zero_and_negatives():
    for value in ("0", "-5"):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(value)


def test_positive_float_rejects_zero_and_negatives():
    assert positive_float("0.05") == pytest.approx(0.05)
    for value in ("0", "-0.1"):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_float(value)


def test_collapse_detects_a_single_predicted_class():
    assert has_collapsed(np.zeros((1, 8)))
    assert has_collapsed(np.ones((1, 8)))


def test_collapse_ignores_a_merely_inaccurate_model():
    # A weak but living model still predicts both classes.
    assert not has_collapsed(np.array([[0.0, 1.0, 0.0, 0.0]]))


def test_sweep_uses_architectures_from_the_main_study():
    for name, tail in SWEEP_ARCHITECTURES.items():
        assert ARCHITECTURES[name] == tail


def test_summary_counts_collapses_and_reports_sample_spread():
    records = [
        {
            "learning_rate": LEARNING_RATES[0],
            "architecture": name,
            "seed": seed,
            "test_accuracy": 80.0 + seed,
            "collapsed": "yes" if seed == 1 else "no",
        }
        for name in SWEEP_ARCHITECTURES
        for seed in (1, 2, 3)
    ]
    row = summarise(records)[0]
    assert row["collapsed"] == len(SWEEP_ARCHITECTURES)
    assert row["runs"] == 3 * len(SWEEP_ARCHITECTURES)
    name = next(iter(SWEEP_ARCHITECTURES))
    assert row[name + " mean"] == pytest.approx(82.0)
    assert row[name + " sd"] == pytest.approx(1.0)


def test_summary_table_columns_line_up():
    records = [
        {
            "learning_rate": rate,
            "architecture": name,
            "seed": seed,
            "test_accuracy": 80.0,
            "collapsed": "no",
        }
        for rate in LEARNING_RATES
        for name in SWEEP_ARCHITECTURES
        for seed in (1, 2, 3)
    ]
    lines = format_summary(summarise(records)).split("\n")
    header = lines[0]
    for name in SWEEP_ARCHITECTURES:
        column = header.index(name)
        for line in lines[1:]:
            assert line[column:].startswith("80.00")
