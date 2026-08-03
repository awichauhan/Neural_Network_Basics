
import math

def sigmoid(z):
    """
    Converts the raw score into a probability between 0 and 1.
    """
    return 1 / (1 + math.exp(-z))


def binary_cross_entropy(y, prediction):
    """
    Measures how different the prediction is from the actual answer.
    """
    epsilon = 1e-15
    prediction = max(epsilon, min(1 - epsilon, prediction))

    loss = -(
        y * math.log(prediction)
        + (1 - y) * math.log(1 - prediction)
    )

    return loss


def forward_pass(x1, x2, w1, w2, bias, y):
    """
    Performs forward propagation.
    """

    # Contribution of each input feature
    u = w1 * x1
    v = w2 * x2

    # Raw score produced by the neuron
    z = u + v + bias

    # Predicted probability
    prediction = sigmoid(z)

    # Error of the prediction
    loss = binary_cross_entropy(y, prediction)

    return u, v, z, prediction, loss


def backward_pass(x1, x2, y, prediction):
    """
    Calculates gradients for the weights and bias.
    """

    dz = prediction - y

    dw1 = dz * x1
    dw2 = dz * x2
    db = dz

    return dz, dw1, dw2, db


def update_parameters(
        w1,
        w2,
        bias,
        dw1,
        dw2,
        db,
        learning_rate
):
    """
    Updates the weights and bias using gradient descent.
    """

    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    bias = bias - learning_rate * db

    return w1, w2, bias


def main():
    # One training example
    x1 = 2.0
    x2 = 3.0

    # Actual correct answer
    y = 1

    # Initial model parameters
    w1 = 0.1
    w2 = -0.2
    bias = 0.5

    learning_rate = 0.1
    number_of_iterations = 1000

    for iteration in range(number_of_iterations):

        # Forward propagation
        u, v, z, prediction, loss = forward_pass(
            x1,
            x2,
            w1,
            w2,
            bias,
            y
        )

        # Backpropagation
        dz, dw1, dw2, db = backward_pass(
            x1,
            x2,
            y,
            prediction
        )

        # Gradient-descent parameter update
        w1, w2, bias = update_parameters(
            w1,
            w2,
            bias,
            dw1,
            dw2,
            db,
            learning_rate
        )

        # Print progress after every 100 iterations
        if iteration % 100 == 0:
            print(
                "Iteration:",
                iteration,
                "| Prediction:",
                round(prediction, 4),
                "| Loss:",
                round(loss, 4)
            )

    print("\n--- Final trained values ---")
    print("w1:", w1)
    print("w2:", w2)
    print("bias:", bias)
    print("final prediction:", prediction)
    print("actual answer:", y)


if __name__ == "__main__":
    main()