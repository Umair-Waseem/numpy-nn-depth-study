import numpy as np

from src.activations import relu, sigmoid


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z, (A, W, b)


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Separate caches let an activation change without touching the linear code.
    return A, (linear_cache, activation_cache)


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    # Hidden layers use ReLU; only the output layer uses sigmoid.
    for l in range(1, L):
        A, cache = linear_activation_forward(
            A, parameters["W" + str(l)], parameters["b" + str(l)], "relu"
        )
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid"
    )
    caches.append(cache)

    return AL, caches
