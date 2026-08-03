import math


def sigmoid(z):
    """
    Converts any raw score into a value between 0 and 1.
    This becomes the neuron's predicted probability.
    """
    return 1 / (1 + math.exp(-z))

def binary_cross_entropy(y, prediction):
    """
    Measures the difference between:
    y          -> actual answer
    prediction -> neuron's answer
    """

    # Prevent math.log(0)
    epsilon = 1e-15
    prediction = max(epsilon, min(1 - epsilon, prediction))

    loss = -(
        y * math.log(prediction)
        + (1 - y) * math.log(1 - prediction)
    )

    return loss


def forward_pass(x1, x2, w1, w2, bias, y):

    # Contribution of the first input
    u = w1 * x1

    # Contribution of the second input
    v = w2 * x2

    # Raw score of the neuron
    z = u + v + bias

    # Predicted probability
    prediction = sigmoid(z)

    # Difference between prediction and correct answer
    loss = binary_cross_entropy(y, prediction)

    return u, v, z, prediction, loss

def backward_pass(x1, x2, y, prediction):
    dz = prediction - y

    dw1 = dz * x1
    dw2 = dz * x2
    db = dz

    return dz, dw1, dw2, db


def main():
    # One training example
    x1 = 2.0
    x2 = 3.0

    # Initial model parameters
    w1 = 0.1
    w2 = -0.2
    bias = 0.5

    # Actual answer supplied by the dataset
    y = 1

    u, v, z, prediction, loss = forward_pass(
        x1,
        x2,
        w1,
        w2,
        bias,
        y
    )

# gradients:
    dz, dw1, dw2, db = backward_pass(
        x1,
        x2,
        y,
        prediction
    )

    print("\n--- Input and parameters ---")
    print("x1:", x1)
    print("x2:", x2)
    print("w1:", w1)
    print("w2:", w2)
    print("bias:", bias)
    print("actual answer y:", y)

    print("u = w1 * x1:", u)
    print("v = w2 * x2:", v)
    print("z = u + v + bias:", z)
    print("Prediction:", prediction)
    print("Loss:", loss)


    print("\n--- Backpropagation ---")
    print("dz:", dz)
    print("dw1:", dw1)
    print("dw2:", dw2)
    print("db:", db)


if __name__ == "__main__":
    main()