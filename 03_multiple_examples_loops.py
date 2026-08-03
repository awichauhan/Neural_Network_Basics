import math


def sigmoid(z):
    """
    Converts the raw neuron score into a probability between 0 and 1.
    """
    return 1 / (1 + math.exp(-z))


def binary_cross_entropy(y, prediction):
    """
    Measures the difference between the actual label and prediction.
    """

    # Prevent log(0)
    epsilon = 1e-15
    prediction = max(epsilon, min(1 - epsilon, prediction))

    loss = -(
        y * math.log(prediction)
        + (1 - y) * math.log(1 - prediction)
    )

    # Return only the loss
    return loss


def forward_pass(x1, x2, w1, w2, bias, y):
    """
    Performs forward propagation for one example.
    """

    # Contribution of input feature x1
    u = w1 * x1

    # Contribution of input feature x2
    v = w2 * x2

    # Raw score produced by the neuron
    z = u + v + bias

    # Predicted probability
    prediction = sigmoid(z)

    # Error for this training example
    loss = binary_cross_entropy(y, prediction)

    return u, v, z, prediction, loss


def backward_pass(x1, x2, y, prediction):
    """
    Calculates gradients for one training example.
    """

    # Gradient of the loss with respect to z
    dz = prediction - y

    # Gradient of the loss with respect to w1
    dw1 = dz * x1

    # Gradient of the loss with respect to w2
    dw2 = dz * x2

    # Gradient of the loss with respect to bias
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
    Updates weights and bias using gradient descent.
    """

    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    bias = bias - learning_rate * db

    return w1, w2, bias


# -------------------------------------------------
# Training data: OR-like security alert system
# -------------------------------------------------

X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

Y = [0, 1, 1, 1]


# Initial neuron parameters
w1 = 0.1
w2 = -0.2
bias = 0.5

learning_rate = 0.1
number_of_iterations = 1000
number_of_examples = len(X)


# -------------------------------------------------
# Training loop
# -------------------------------------------------

for iteration in range(number_of_iterations):

    # Reset totals at the beginning of every iteration
    total_loss = 0
    total_dw1 = 0
    total_dw2 = 0
    total_db = 0

    # Process every training example
    for example_index in range(number_of_examples):

        x1 = X[example_index][0]
        x2 = X[example_index][1]
        y = Y[example_index]

        # Forward propagation for this example
        u, v, z, prediction, loss = forward_pass(
            x1,
            x2,
            w1,
            w2,
            bias,
            y
        )

        # Backpropagation for this example
        dz, dw1, dw2, db = backward_pass(
            x1,
            x2,
            y,
            prediction
        )

        # Add this example's values to the totals
        total_loss += loss
        total_dw1 += dw1
        total_dw2 += dw2
        total_db += db

    # Average results from all examples
    average_loss = total_loss / number_of_examples

    average_dw1 = total_dw1 / number_of_examples
    average_dw2 = total_dw2 / number_of_examples
    average_db = total_db / number_of_examples

    # Update parameters once after processing all examples
    w1, w2, bias = update_parameters(
        w1,
        w2,
        bias,
        average_dw1,
        average_dw2,
        average_db,
        learning_rate
    )

    # Display training progress
    if iteration % 100 == 0:
        print(
            "Iteration:",
            iteration,
            "| Average loss:",
            round(average_loss, 4)
        )


# -------------------------------------------------
# Display final learned parameters
# -------------------------------------------------

print("\n--- Final trained parameters ---")
print("w1:", w1)
print("w2:", w2)
print("bias:", bias)


# -------------------------------------------------
# Test the trained neuron
# -------------------------------------------------

print("\n--- Final predictions ---")

for example_index in range(number_of_examples):

    x1 = X[example_index][0]
    x2 = X[example_index][1]
    y = Y[example_index]

    u, v, z, prediction, loss = forward_pass(
        x1,
        x2,
        w1,
        w2,
        bias,
        y
    )

    predicted_class = 1 if prediction >= 0.5 else 0

    print(
        "Input:",
        x1,
        x2,
        "| Probability:",
        round(prediction, 4),
        "| Predicted class:",
        predicted_class,
        "| Actual class:",
        y
    )