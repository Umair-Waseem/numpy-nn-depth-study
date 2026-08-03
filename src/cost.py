import numpy as np

# Bounds the sigmoid output away from 0 and 1 so the logarithm stays finite.
EPSILON = 1e-12


def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, EPSILON, 1 - EPSILON)
    cost = -(1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return float(cost)
