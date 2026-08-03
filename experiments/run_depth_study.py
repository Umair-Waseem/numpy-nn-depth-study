import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import CLASS_NAMES, load_binary_pair
from src.model import L_layer_model, accuracy, predict

# Render straight to file rather than requiring a display.
plt.switch_backend("Agg")

DESCRIPTION = (
    "Train networks of increasing depth on identical data and hyperparameters, "
    "so that depth is the only variable that changes."
)
ARCHITECTURES = {
    "logistic regression": [1],
    "1 hidden layer, 7 units": [7, 1],
    "1 hidden layer, 20 units": [20, 1],
    "2 hidden layers": [20, 7, 1],
    "3 hidden layers": [20, 7, 5, 1],
    "4 hidden layers": [20, 20, 7, 5, 1],
}
SEEDS = [1, 2, 3, 4, 5]
LEARNING_RATE = 0.05
NUM_ITERATIONS = 2500
RECORD_EVERY = 100
# Two-sided 95 percent t multipliers, keyed by degrees of freedom.
T_MULTIPLIER = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
NEGATIVE_CLASS = 0
POSITIVE_CLASS = 6
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=NUM_ITERATIONS,
        help="gradient descent steps",
    )
    parser.add_argument(
        "--learning-rate", type=positive_float, default=LEARNING_RATE, help="step size"
    )
    parser.add_argument(
        "--subset", type=positive_int, default=None, help="cap on training examples"
    )
    return parser.parse_args()


def run_grid(X_train, Y_train, X_test, Y_test, args):
    n_x = X_train.shape[0]
    records = []
    curves = {}

    for name, tail in ARCHITECTURES.items():
        layer_dims = [n_x] + tail
        for seed in SEEDS:
            start = time.time()
            parameters, history = L_layer_model(
                X_train,
                Y_train,
                layer_dims,
                learning_rate=args.learning_rate,
                num_iterations=args.iterations,
                seed=seed,
                record_every=RECORD_EVERY,
            )
            # Only one seed is plotted, so the curves stay comparable.
            if seed == SEEDS[0]:
                curves[name] = history
            record = {
                "architecture": name,
                "layer_dims": "-".join(str(d) for d in layer_dims),
                "hidden_layers": len(tail) - 1,
                "seed": seed,
                "train_accuracy": round(
                    accuracy(predict(X_train, parameters), Y_train), 2
                ),
                "test_accuracy": round(
                    accuracy(predict(X_test, parameters), Y_test), 2
                ),
                "final_cost": round(history[-1][1], 4),
                "seconds": round(time.time() - start, 1),
            }
            records.append(record)
            print(
                f"{name:<26} seed {seed}  "
                f"train {record['train_accuracy']:6.2f}%  "
                f"test {record['test_accuracy']:6.2f}%  "
                f"cost {record['final_cost']:.4f}  "
                f"({record['seconds']:.0f}s)",
                flush=True,
            )

    return records, curves


def compare_to_baseline(records):
    names = list(ARCHITECTURES)
    baseline = names[0]
    rows = []

    def test_accuracy(name, seed):
        return next(
            record["test_accuracy"]
            for record in records
            if record["architecture"] == name and record["seed"] == seed
        )

    for name in names[1:]:
        # Pairing on seed removes initialisation variance from the difference.
        differences = [
            test_accuracy(name, seed) - test_accuracy(baseline, seed) for seed in SEEDS
        ]
        mean = float(np.mean(differences))
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(SEEDS)))
        margin = T_MULTIPLIER[len(SEEDS) - 1] * standard_error
        rows.append(
            {
                "architecture": name,
                "mean_difference": round(mean, 3),
                "standard_error": round(standard_error, 3),
                "ci_low": round(mean - margin, 3),
                "ci_high": round(mean + margin, 3),
                "significant": "yes"
                if mean - margin > 0 or mean + margin < 0
                else "no",
            }
        )

    return rows


def write_csv(rows, path):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_costs(curves, path):
    plt.figure(figsize=(8, 5))
    for name, history in curves.items():
        iterations, costs = zip(*history)
        plt.plot(iterations, costs, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Training cost")
    plt.title("Training cost by network depth")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_accuracy(records, path):
    names = list(ARCHITECTURES)
    positions = np.arange(len(names))
    plt.figure(figsize=(9, 5))

    for split, marker in (("train", "o"), ("test", "s")):
        values = [
            [
                record[f"{split}_accuracy"]
                for record in records
                if record["architecture"] == name
            ]
            for name in names
        ]
        means = [np.mean(v) for v in values]
        # ddof=1 matches the sample standard deviation used in the comparison.
        spreads = [np.std(v, ddof=1) for v in values]
        plt.errorbar(
            positions,
            means,
            yerr=spreads,
            marker=marker,
            capsize=4,
            label=f"{split} accuracy",
        )

    plt.xlabel("Architecture")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy by architecture, mean of {len(SEEDS)} seeds")
    plt.xticks(positions, names, rotation=20, ha="right", fontsize=8)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, Y_train, X_test, Y_test = load_binary_pair(NEGATIVE_CLASS, POSITIVE_CLASS)
    if args.subset is not None:
        X_train, Y_train = X_train[:, : args.subset], Y_train[:, : args.subset]

    print(
        f"{CLASS_NAMES[NEGATIVE_CLASS]} vs {CLASS_NAMES[POSITIVE_CLASS]}  |  "
        f"train {X_train.shape[1]}  test {X_test.shape[1]}  "
        f"features {X_train.shape[0]}\n"
    )

    records, curves = run_grid(X_train, Y_train, X_test, Y_test, args)
    comparison = compare_to_baseline(records)

    print(f"\nTest accuracy against {next(iter(ARCHITECTURES))}, 95 percent interval:")
    for row in comparison:
        print(
            f"  {row['architecture']:<26} "
            f"{row['mean_difference']:+6.3f}  "
            f"[{row['ci_low']:+6.3f}, {row['ci_high']:+6.3f}]  "
            f"significant: {row['significant']}"
        )

    write_csv(records, RESULTS_DIR / "results.csv")
    write_csv(comparison, RESULTS_DIR / "baseline_comparison.csv")
    plot_costs(curves, RESULTS_DIR / "cost_curves.png")
    plot_accuracy(records, RESULTS_DIR / "accuracy_by_architecture.png")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
