import numpy as np

from src.backward import L_model_backward
from src.cost import compute_cost
from src.forward import L_model_forward
from src.initialization import initialize_parameters_deep
from src.update import update_parameters

# A predicted probability above this value is read as the positive class.
DECISION_THRESHOLD = 0.5


def L_layer_model(
    X,
    Y,
    layer_dims,
    learning_rate=0.05,
    num_iterations=2500,
    seed=1,
    record_every=100,
    print_cost=False,
):
    parameters = initialize_parameters_deep(layer_dims, seed=seed)
    history = []

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % record_every == 0:
            history.append((i, cost))
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.6f}")

    # Costs are logged before each update, so the returned parameters need scoring.
    AL, _ = L_model_forward(X, parameters)
    history.append((num_iterations, compute_cost(AL, Y)))
    if print_cost:
        print(f"Cost after iteration {num_iterations}: {history[-1][1]:.6f}")

    return parameters, history


def predict(X, parameters):
    AL, _ = L_model_forward(X, parameters)
    return (AL > DECISION_THRESHOLD).astype(float)


def accuracy(predictions, Y):
    return float(np.mean(predictions == Y) * 100)
