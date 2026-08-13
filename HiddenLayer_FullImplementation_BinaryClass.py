import numpy as np


def sigmoid(Z):
    """
    Applies sigmoid element-wise.
    """
    return 1 / (1 + np.exp(-Z))


def binary_cross_entropy(Y, A):
    """
    Calculates average binary cross-entropy loss.
    """

    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)

    loss = -np.mean(
        Y * np.log(A)
        + (1 - Y) * np.log(1 - A)
    )

    return loss

def forward_pass(X, W1, b1, W2, b2):
    """
    Forward propagation through:
    input -> hidden layer -> output layer
    """

    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2,A1) + b2  #order in dot operation matters as matrix multiplication is not commutative
    A = sigmoid(Z2)

    return Z1,A1,Z2,A

def backward_pass(X, Y, A1, A2, W2):

    number_of_examples = X.shape[1]

    # ===================================
    # OUTPUT LAYER
    # ===================================

    # For sigmoid + binary cross entropy
    dZ2 = A2 - Y

    # Gradient of W2
    dW2 = (
        1 / number_of_examples
    ) * np.dot(dZ2, A1.T)

    # Gradient of b2
    db2 = (
        1 / number_of_examples
    ) * np.sum(
        dZ2,
        axis=1,
        keepdims=True
    )

    # ===================================
    # HIDDEN LAYER
    # ===================================

    # Move error backward through W2
    dA1 = np.dot(W2.T, dZ2)

    # Derivative of sigmoid
    dZ1 = dA1 * A1 * (1 - A1)

    # Gradient of W1
    dW1 = (
        1 / number_of_examples
    ) * np.dot(dZ1, X.T)

    # Gradient of b1
    db1 = (
        1 / number_of_examples
    ) * np.sum(
        dZ1,
        axis=1,
        keepdims=True
    )

    return dW1, db1, dW2, db2

def update_parameters(
        W1,
        b1,
        W2,
        b2,
        dW1,
        db1,
        dW2,
        db2,
        learning_rate
):

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

def main():

    X = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]
    ])

    Y = np.array([             # XOR table
        [0.0, 1.0, 1.0, 0.0]
    ])

    # 2 hidden neurons
    W1 = np.array([
        [0.1, -0.2],
        [0.7,  0.2]
    ])

    b1 = np.array([
        [0.5],
        [0.3]
    ])

    # 1 output neuron
    W2 = np.array([
        [0.4, -0.1]
    ])

    b2 = np.array([
        [0.2]
    ])

    learning_rate = 0.5
    number_of_iterations = 10000

    # ===================================
    # TRAINING LOOP
    # ===================================

    for iteration in range(number_of_iterations):

        # -------------------------------
        # Forward propagation
        # -------------------------------

        Z1, A1, Z2, A2 = forward_pass(
            X,
            W1,
            b1,
            W2,
            b2
        )

        # -------------------------------
        # Loss
        # -------------------------------

        loss = binary_cross_entropy(
            Y,
            A2
        )

        # -------------------------------
        # Backpropagation
        # -------------------------------

        dW1, db1, dW2, db2 = backward_pass(
            X,
            Y,
            A1,
            A2,
            W2
        )

        # -------------------------------
        # Gradient descent
        # -------------------------------

        W1, b1, W2, b2 = update_parameters(
            W1,
            b1,
            W2,
            b2,
            dW1,
            db1,
            dW2,
            db2,
            learning_rate
        )

        if iteration % 1000 == 0:
            print(
                "Iteration:",
                iteration,
                "| Loss:",
                round(float(loss), 4)
            )

    # ===================================
    # FINAL FORWARD PASS
    # ===================================

    Z1, A1, Z2, A2 = forward_pass(
        X,
        W1,
        b1,
        W2,
        b2
    )

    predicted_classes = (
        A2 >= 0.5
    ).astype(int)

    # ===================================
    # RESULTS
    # ===================================

    print("\nFinal probabilities:")
    print(A2)

    print("\nPredicted classes:")
    print(predicted_classes)

    print("\nActual classes:")
    print(Y.astype(int))

    print("\nFinal W1:")
    print(W1)

    print("\nFinal b1:")
    print(b1)

    print("\nFinal W2:")
    print(W2)

    print("\nFinal b2:")
    print(b2)

    # -----------------------------------
    # Example-wise results
    # -----------------------------------

    print("\n--- Example-wise results ---")

    number_of_examples = X.shape[1]

    for example_index in range(number_of_examples):

        x1 = X[0, example_index]
        x2 = X[1, example_index]

        probability = A2[0, example_index]

        predicted_class = (
            predicted_classes[
                0,
                example_index
            ]
        )

        actual_class = int(
            Y[
                0,
                example_index
            ]
        )

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
