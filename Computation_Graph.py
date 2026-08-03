import math

def sigmoid(value):
    """
    Converts the weighted sum into a probability between 0 and 1.
    """
    return 1 / (1 + math.exp(-value))


def binary_cross_entropy(y, prediction):
    """
    Compares the actual label y with the predicted probability.
    """

    # Prevent math.log(0)
    epsilon = 1e-15
    prediction = max(epsilon, min(1 - epsilon, prediction))

    loss = -(
        y * math.log(prediction)
        + (1 - y) * math.log(1 - prediction)
    )

    return loss


def forward_propagation(w1, w2, x1, x2, bias, y):
    """
    Performs one forward pass through a single logistic-regression neuron.
    """

    # Small computation-graph operations
    u = w1 * x1
    v = w2 * x2
    z = u + v + bias
    prediction = sigmoid(z)

    loss = binary_cross_entropy(y, prediction)

    print("\n--- Forward propagation ---")
    print("x1:", x1)
    print("x2:", x2)
    print("u = w1 * x1:", u)
    print("v = w2 * x2:", v)
    print("z = u + v + bias:", z)
    print("prediction a:", prediction)
    print("actual label y:", y)
    print("loss:", loss)

    return prediction, loss


def back_propagation(prediction, y, x1, x2):
    """
    Calculates how much each parameter contributed to the loss.
    """

    dz = prediction - y

    dw1 = dz * x1
    dw2 = dz * x2
    db = dz

    print("\n--- Backpropagation ---")
    print("dz:", dz)
    print("dw1:", dw1)
    print("dw2:", dw2)
    print("db:", db)

    return dw1, dw2, db


