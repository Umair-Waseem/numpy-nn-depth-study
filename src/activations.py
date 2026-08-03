import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    # Z is cached because the backward pass needs it to compute the local derivative.
    return A, Z


def relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    return dA * s * (1 - s)


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    # ReLU is flat for non-positive inputs, so no gradient passes there.
    dZ[Z <= 0] = 0
    return dZ
