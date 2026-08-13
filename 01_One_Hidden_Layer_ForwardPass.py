import numpy as np


def sigmoid(Z):
    """
    Converts raw neuron scores into probabilities between 0 and 1.

    Z shape: (1, number_of_examples)
    A shape: (1, number_of_examples)
    """
    A = 1 / (1 + np.exp(-Z))

    return A

def forward_pass(X, W, bias):
    """
    Performs forward propagation for all examples together.
    """

    # Raw score for every training example
    Z = np.dot(W.Tcc, X) + bias

    # Probability prediction for every example
    A = sigmoid(Z)

    # Average loss across all examples
   #  loss = binary_cross_entropy(Y, A)

    return Z, A

def output_layer_processing(W2,A1,b2):
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    return Z2, A2
def main():

    X = np.array([
        [1.2, 2.1, 2.3, 1.0],
        [0.3, 1.5, 3.0, 2.0]
    ])

    Y = np.array([
        [0.3, 1.0, 2.0, 3.0]
    ])

    # -----------------------------------------
    # Hidden layer: 2 neurons
    # -----------------------------------------

    W1 = np.array([
        [0.1, -0.2],
        [0.7, 0.2]
    ])

    b1 = np.array([
        [0.5],
        [0.3]
    ])

    # -----------------------------------------
    # Output layer: 1 neuron
    # -----------------------------------------

    W2 = np.array([
        [0.4, -0.1]
    ])

    b2 = np.array([
        [0.2]
    ])

    Z1, A1 = forward_pass(X,W1,b1)
    print("Hidden layer output: ", A1)

    Z2, A2 = output_layer_processing(W2,A1,b2)
    print("Output Layer output: ", A2)

main()