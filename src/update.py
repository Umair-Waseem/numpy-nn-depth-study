import copy


def update_parameters(params, grads, learning_rate):
    # Copy first so the caller's parameters are never modified in place.
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters
