import numpy as np
def sigmoid(Z):
    """
    Applies sigmoid element-wise.
    """
    return 1 / (1 + np.exp(-Z))

def initialize_parameter(n_x, n_h1, n_h2, n_h3, n_h4, n_y):

    W1 = np.random.randn(n_h1, n_x) * 0.5
    b1 = np.zeros((n_h1, 1))

    W2 = np.random.randn(n_h2, n_h1) * 0.5
    b2 = np.zeros((n_h2, 1))

    W3 = np.random.randn(n_h3, n_h2) * 0.5
    b3 = np.zeros((n_h3, 1))

    W4 = np.random.randn(n_h4, n_h3) * 0.5
    b4 = np.zeros((n_h4, 1))

    W = np.random.randn(n_y, n_h4) * 0.5
    b = np.zeros((n_y, 1))

    return W1, b1, W2, b2, W3, b3, W4, b4, W, b

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

def forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4, W, b):

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    Z4 = np.dot(W4, A3) + b4
    A4 = sigmoid(Z4)

    Z = np.dot(W, A4) + b
    A = sigmoid(Z)

    return Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z, A

def backward_pass(
        X, Y,
        A1, A2, A3, A4, A,
        W2, W3, W4, W
):

    m = X.shape[1]

    # ==================================
    # OUTPUT LAYER
    # ==================================

    dZ = A - Y

    dW = (1 / m) * np.dot(dZ, A4.T)

    db = (1 / m) * np.sum(
        dZ,
        axis=1,
        keepdims=True
    )

    # ==================================
    # HIDDEN LAYER 4
    # ==================================

    dA4 = np.dot(W.T, dZ)

    dZ4 = dA4 * A4 * (1 - A4)

    dW4 = (1 / m) * np.dot(dZ4, A3.T)

    db4 = (1 / m) * np.sum(
        dZ4,
        axis=1,
        keepdims=True
    )

    # ==================================
    # HIDDEN LAYER 3
    # ==================================

    dA3 = np.dot(W4.T, dZ4)

    dZ3 = dA3 * A3 * (1 - A3)

    dW3 = (1 / m) * np.dot(dZ3, A2.T)

    db3 = (1 / m) * np.sum(
        dZ3,
        axis=1,
        keepdims=True
    )

    # ==================================
    # HIDDEN LAYER 2
    # ==================================

    dA2 = np.dot(W3.T, dZ3)

    dZ2 = dA2 * A2 * (1 - A2)

    dW2 = (1 / m) * np.dot(dZ2, A1.T)

    db2 = (1 / m) * np.sum(
        dZ2,
        axis=1,
        keepdims=True
    )

    # ==================================
    # HIDDEN LAYER 1
    # ==================================

    dA1 = np.dot(W2.T, dZ2)

    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = (1 / m) * np.dot(dZ1, X.T)

    db1 = (1 / m) * np.sum(
        dZ1,
        axis=1,
        keepdims=True
    )

    return (
        dW1, db1,
        dW2, db2,
        dW3, db3,
        dW4, db4,
        dW, db
    )

def update_parameters(
        W1, b1,
        W2, b2,
        W3, b3,
        W4, b4,
        W, b,
        dW1, db1,
        dW2, db2,
        dW3, db3,
        dW4, db4,
        dW, db,
        learning_rate
):

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    W4 = W4 - learning_rate * dW4
    b4 = b4 - learning_rate * db4

    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W1, b1, W2, b2, W3, b3, W4, b4, W, b

def main():

    X = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]
    ])

    Y = np.array([
        [0.0, 1.0, 1.0, 0.0]
    ])

    n_x = 2
    n_h1 = 4
    n_h2 = 5
    n_h3 = 3
    n_h4 = 2
    n_y = 1

    W1, b1, W2, b2, W3, b3, W4, b4, W, b = initialize_parameter(n_x,n_h1,n_h2,n_h3,n_h4,n_y)

    learning_rate = 0.5
    number_of_iterations = 10000

    # ===================================
    # TRAINING LOOP
    # ===================================

    for iteration in range(number_of_iterations):

        # -------------------------------
        # Forward propagation
        # -------------------------------

        Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z, A = forward_pass(
            X,
            W1, b1,
            W2, b2,
            W3, b3,
            W4, b4,
            W, b
        )

        # -------------------------------
        # Loss
        # -------------------------------

        loss = binary_cross_entropy(
            Y,
            A
        )

        # -------------------------------
        # Backpropagation
        # -------------------------------

        dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW, db = backward_pass(
            X,
            Y,
            A1,
            A2,
            A3,
            A4,
            A,
            W2,
            W3,
            W4,
            W
        )
        # -------------------------------
        # Gradient descent
        # -------------------------------

        W1, b1, W2, b2, W3, b3, W4, b4, W, b = update_parameters(
            W1, b1,
            W2, b2,
            W3, b3,
            W4, b4,
            W, b,
            dW1, db1,
            dW2, db2,
            dW3, db3,
            dW4, db4,
            dW, db,
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

    Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z, A = forward_pass(
        X,
        W1, b1,
        W2, b2,
        W3, b3,
        W4, b4,
        W, b
    )

    predicted_classes = (
            A >= 0.5
    ).astype(int)

    # ===================================
    # RESULTS
    # ===================================

    print("\nFinal probabilities:")
    print(A)

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

    print("\nFinal W3:")
    print(W3)

    print("\nFinal b3:")
    print(b3)

    print("\nFinal W4:")
    print(W4)

    print("\nFinal b4:")
    print(b4)

    # -----------------------------------
    # Example-wise results
    # -----------------------------------

    print("\n--- Example-wise results ---")

    number_of_examples = X.shape[1]

    for example_index in range(number_of_examples):
        x1 = X[0, example_index]
        x2 = X[1, example_index]

        probability = A[0, example_index]

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


