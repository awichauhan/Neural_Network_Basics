import numpy as np


def sigmoid(Z):
    """
    Converts raw scores into probabilities between 0 and 1.
    """
    A = 1 / (1 + np.exp(-Z))

    return A


def binary_cross_entropy(Y, A):
    """
    Calculates the average loss across all examples.
    """

    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)

    loss = -np.mean(
        Y * np.log(A)
        + (1 - Y) * np.log(1 - A)
    )

    return loss


def forward_pass(X, Y, W, bias):
    """
    Performs vectorized forward propagation.
    """

    Z = np.dot(W.T, X) + bias
    A = sigmoid(Z)
    loss = binary_cross_entropy(Y, A)

    return Z, A, loss


def backward_pass(X, Y, A):
    """
    Calculates analytical gradients using backpropagation.
    """

    number_of_examples = X.shape[1]

    dZ = A - Y

    dW = (
        1 / number_of_examples
    ) * np.dot(X, dZ.T)

    db = (
        1 / number_of_examples
    ) * np.sum(dZ)

    return dZ, dW, db


def calculate_loss(X, Y, W, bias):
    """
    Calculates only the loss.

    This is used while checking slightly changed
    weights and bias values.
    """

    Z, A, loss = forward_pass(
        X,
        Y,
        W,
        bias
    )

    return loss


def gradient_check(X, Y, W, bias):
    """
    Compares backpropagation gradients with numerical gradients.
    """

    epsilon = 1e-7

    # -----------------------------------------
    # Analytical gradients from backpropagation
    # -----------------------------------------

    Z, A, loss = forward_pass(
        X,
        Y,
        W,
        bias
    )

    dZ, dW, db = backward_pass(
        X,
        Y,
        A
    )

    # Space for storing approximate weight gradients
    approximate_dW = np.zeros_like(W)

    # -----------------------------------------
    # Numerically check every weight
    # -----------------------------------------

    for weight_index in range(W.shape[0]):

        # Create separate copies so original W remains unchanged
        W_plus = W.copy()
        W_minus = W.copy()

        # Slightly increase this weight
        W_plus[weight_index, 0] += epsilon

        # Slightly decrease this weight
        W_minus[weight_index, 0] -= epsilon

        loss_plus = calculate_loss(
            X,
            Y,
            W_plus,
            bias
        )

        loss_minus = calculate_loss(
            X,
            Y,
            W_minus,
            bias
        )

        approximate_dW[weight_index, 0] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    # -----------------------------------------
    # Numerically check the bias
    # -----------------------------------------

    bias_plus = bias + epsilon
    bias_minus = bias - epsilon

    loss_plus = calculate_loss(
        X,
        Y,
        W,
        bias_plus
    )

    loss_minus = calculate_loss(
        X,
        Y,
        W,
        bias_minus
    )

    approximate_db = (
        loss_plus - loss_minus
    ) / (2 * epsilon)

    # -----------------------------------------
    # Compare analytical and numerical gradients
    # -----------------------------------------

    analytical_gradients = np.append(
        dW.flatten(),
        db
    )

    numerical_gradients = np.append(
        approximate_dW.flatten(),
        approximate_db
    )

    numerator = np.linalg.norm(
        analytical_gradients - numerical_gradients
    )

    denominator = (
        np.linalg.norm(analytical_gradients)
        + np.linalg.norm(numerical_gradients)
    )

    difference = numerator / (denominator + 1e-15)

    print("\n--- Weight gradient checking ---")
    print("Backpropagation dW:")
    print(dW)

    print("\nNumerical approximate dW:")
    print(approximate_dW)

    print("\n--- Bias gradient checking ---")
    print("Backpropagation db:", db)
    print("Numerical approximate db:", approximate_db)

    print("\nOverall difference:", difference)

    if difference < 1e-7:
        print("Gradient checking passed.")
        print("Backpropagation appears to be correct.")
    else:
        print("Gradient checking failed.")
        print("There may be an error in backpropagation.")


def main():

    # Same OR training dataset
    X = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]
    ])

    Y = np.array([
        [0.0, 1.0, 1.0, 1.0]
    ])

    W = np.array([
        [0.1],
        [-0.2]
    ])

    bias = 0.5

    gradient_check(
        X,
        Y,
        W,
        bias
    )


if __name__ == "__main__":
    main()