import numpy as np


def sigmoid(Z):
    """
    Converts raw neuron scores into probabilities between 0 and 1.

    Z shape: (1, number_of_examples)
    A shape: (1, number_of_examples)
    """
    A = 1 / (1 + np.exp(-Z))

    return A


def binary_cross_entropy(Y, A):
    """
    Calculates the average loss across all training examples.

    Y contains the actual answers.
    A contains the neuron's predictions.
    """

    # Prevent log(0)
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)

    loss = -np.mean(
        Y * np.log(A)
        + (1 - Y) * np.log(1 - A)
    )

    return loss


def forward_pass(X, Y, W, bias):
    """
    Performs forward propagation for all examples together.
    """

    # Raw score for every training example
    Z = np.dot(W.T, X) + bias

    # Probability prediction for every example
    A = sigmoid(Z)

    # Average loss across all examples
    loss = binary_cross_entropy(Y, A)

    return Z, A, loss


def backward_pass(X, Y, A):
    """
    Calculates gradients using all examples together.
    """

    number_of_examples = X.shape[1]

    # Difference between predictions and actual answers
    dZ = A - Y

    # Average gradient of the weights
    dW = (
        1 / number_of_examples
    ) * np.dot(X, dZ.T)

    # Average gradient of the bias
    db = (
        1 / number_of_examples
    ) * np.sum(dZ)

    return dZ, dW, db


def update_parameters(
        W,
        bias,
        dW,
        db,
        learning_rate
):
    """
    Updates weights and bias using gradient descent.
    """

    W = W - learning_rate * dW
    bias = bias - learning_rate * db

    return W, bias


def main():

    # -------------------------------------------------
    # Training dataset
    # -------------------------------------------------

    # Rows represent features.
    # Columns represent training examples.
    X = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]
    ])

    # Actual answer corresponding to each column of X
    Y = np.array([
        [0.0, 1.0, 1.0, 1.0]
    ])

    # -------------------------------------------------
    # Initial parameters of one neuron
    # -------------------------------------------------

    W = np.array([
        [0.1],
        [-0.2]
    ])

    bias = 0.5

    learning_rate = 0.1
    number_of_iterations = 1000

    print("--- Shapes before training ---")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("W shape:", W.shape)

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------

    for iteration in range(number_of_iterations):

        # Forward propagation
        Z, A, loss = forward_pass(
            X,
            Y,
            W,
            bias
        )

        # Backpropagation
        dZ, dW, db = backward_pass(
            X,
            Y,
            A
        )

        # Gradient-descent update
        W, bias = update_parameters(
            W,
            bias,
            dW,
            db,
            learning_rate
        )

        if iteration % 100 == 0:
            print(
                "Iteration:",
                iteration,
                "| Loss:",
                round(float(loss), 4)
            )

    # -------------------------------------------------
    # Final trained parameters
    # -------------------------------------------------

    print("\n--- Final trained parameters ---")
    print("W:")
    print(W)

    print("bias:", bias)

    # -------------------------------------------------
    # Final predictions
    # -------------------------------------------------

    Z, A, loss = forward_pass(
        X,
        Y,
        W,
        bias
    )

    predicted_classes = (A >= 0.5).astype(int)

    print("\n--- Final probability predictions ---")
    print(A)

    print("\n--- Final predicted classes ---")
    print(predicted_classes)

    print("\n--- Actual classes ---")
    print(Y.astype(int))

    print("\nFinal loss:", loss)

    # -------------------------------------------------
    # Display each example separately
    # -------------------------------------------------

    print("\n--- Example-wise results ---")

    number_of_examples = X.shape[1]

    for example_index in range(number_of_examples):

        x1 = X[0, example_index]
        x2 = X[1, example_index]

        probability = A[0, example_index]
        predicted_class = predicted_classes[0, example_index]
        actual_class = int(Y[0, example_index])

        print(
            "Input:",
            x1,
            x2,
            "| Probability:",
            round(float(probability), 4),
            "| Predicted:",
            predicted_class,
            "| Actual:",
            actual_class
        )


if __name__ == "__main__":
    main()