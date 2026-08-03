import numpy as np


def initialize_parameters_deep(layer_dims, seed=1):
    np.random.seed(seed)
    parameters = {}

    for l in range(1, len(layer_dims)):
        # Dividing by the square root of the fan-in keeps activation variance stable.
        parameters["W" + str(l)] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) / np.sqrt(layer_dims[l - 1])
        # Biases can start at zero because random weights already break the symmetry.
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters
