import argparse
import time

import numpy as np

from experiments.run_depth_study import (
    ARCHITECTURES,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    RESULTS_DIR,
    positive_int,
    write_csv,
)
from src.data import CLASS_NAMES, load_binary_pair
from src.model import L_layer_model, accuracy, predict

DESCRIPTION = (
    "Sweep the learning rate across networks of different depth to find a step "
    "size that trains every architecture without collapsing any of them."
)
LEARNING_RATES = [0.01, 0.05, 0.10, 0.25, 0.50]
# Only networks with hidden layers can suffer dead ReLU, so the baseline is left out.
SWEEP_ARCHITECTURES = {
    name: ARCHITECTURES[name]
    for name in ("1 hidden layer, 7 units", "3 hidden layers", "4 hidden layers")
}
SEEDS = [1, 2, 3]
SUBSET = 3000
ITERATIONS = 800


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--iterations", type=positive_int, default=ITERATIONS, help="steps per run"
    )
    parser.add_argument(
        "--subset", type=positive_int, default=SUBSET, help="training examples used"
    )
    return parser.parse_args()


def has_collapsed(predictions):
    # A dead network emits one class for every input, whatever its accuracy.
    return len(np.unique(predictions)) == 1


def run_sweep(X_train, Y_train, X_test, Y_test, args):
    n_x = X_train.shape[0]
    records = []

    for learning_rate in LEARNING_RATES:
        for name, tail in SWEEP_ARCHITECTURES.items():
            for seed in SEEDS:
                start = time.time()
                parameters, _ = L_layer_model(
                    X_train,
                    Y_train,
                    [n_x] + tail,
                    learning_rate=learning_rate,
                    num_iterations=args.iterations,
                    seed=seed,
                )
                predictions = predict(X_test, parameters)
                collapsed = has_collapsed(predictions)
                records.append(
                    {
                        "learning_rate": learning_rate,
                        "architecture": name,
                        "seed": seed,
                        "test_accuracy": round(accuracy(predictions, Y_test), 2),
                        "collapsed": "yes" if collapsed else "no",
                        "seconds": round(time.time() - start, 1),
                    }
                )
                print(
                    f"lr {learning_rate:<5} {name:<26} seed {seed}  "
                    f"test {records[-1]['test_accuracy']:6.2f}%"
                    f"{'  COLLAPSED' if collapsed else ''}",
                    flush=True,
                )

    return records


def summarise(records):
    rows = []
    for learning_rate in LEARNING_RATES:
        group = [r for r in records if r["learning_rate"] == learning_rate]
        # Skip rates with no runs so an absent group never averages to NaN.
        if not group:
            continue
        row = {
            "learning_rate": learning_rate,
            "collapsed": sum(1 for r in group if r["collapsed"] == "yes"),
            "runs": len(group),
        }
        for name in SWEEP_ARCHITECTURES:
            scores = [r["test_accuracy"] for r in group if r["architecture"] == name]
            row[name + " mean"] = round(float(np.mean(scores)), 2)
            # ddof=1 matches the sample standard deviation used everywhere else.
            row[name + " sd"] = round(float(np.std(scores, ddof=1)), 2)
        rows.append(row)
    return rows


def format_summary(summary):
    width = max(len(name) for name in SWEEP_ARCHITECTURES) + 3
    header = "".join(f"{name:<{width}}" for name in SWEEP_ARCHITECTURES)
    lines = [f"{'learning rate':<16}{'collapsed':<12}{header}".rstrip()]

    for row in summary:
        cells = ""
        for name in SWEEP_ARCHITECTURES:
            score = f"{row[name + ' mean']:.2f} ± {row[name + ' sd']:.2f}"
            cells += f"{score:<{width}}"
        collapsed = f"{row['collapsed']} / {row['runs']}"
        lines.append(f"{row['learning_rate']:<16}{collapsed:<12}{cells}".rstrip())

    return "\n".join(lines)


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, Y_train, X_test, Y_test = load_binary_pair(NEGATIVE_CLASS, POSITIVE_CLASS)
    X_train, Y_train = X_train[:, : args.subset], Y_train[:, : args.subset]
    print(
        f"{CLASS_NAMES[NEGATIVE_CLASS]} vs {CLASS_NAMES[POSITIVE_CLASS]}  |  "
        f"train {X_train.shape[1]}  test {X_test.shape[1]}  "
        f"features {X_train.shape[0]}\n"
    )

    records = run_sweep(X_train, Y_train, X_test, Y_test, args)
    summary = summarise(records)

    print("\n" + format_summary(summary))
    write_csv(records, RESULTS_DIR / "learning_rate_sweep.csv")
    write_csv(summary, RESULTS_DIR / "learning_rate_summary.csv")
    print(f"\nResults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
