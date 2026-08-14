import numpy as np
def sigmoid(Z):
    """
    Applies sigmoid element-wise.
    """
    return 1 / (1 + np.exp(-Z))

def initialize_parameter(n_x, n_h1, n_h2, n_h3, n_h4, n_y):

    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))

    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))

    W3 = np.random.randn(n_h3, n_h2) * 0.01
    b3 = np.zeros((n_h3, 1))

    W4 = np.random.randn(n_h4, n_h3) * 0.01
    b4 = np.zeros((n_h4, 1))

    W = np.random.randn(n_y, n_h4) * 0.01   # for output layers
    b = np.zeros((n_y, 1))

    return W1, b1, W2, b2, W3, b3, W4, b4, W, b

def forward_pass(X, W1, b1, W2, b2, W3, b3, W4, b4):
    """
    Forward propagation through:
    input -> hidden layer -> output layer
    """

    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2,A1) + b2  #order in dot operation matters as matrix multiplication is not commutative
    A2 = sigmoid(Z2)

    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)

    Z4 = np.dot(W4, A3) + b4
    A4 = sigmoid(Z4)

    return Z4,A4

def output_layer_prediction(Z4,A4,b,W4):
    Z = np.dot(W4,Z4) + b
    A = sigmoid(Z)

    return Z, A

def main():

    X = np.array([
        [1.2, 2.1, 2.3, 1.0],
        [0.3, 1.5, 3.0, 2.0]
    ])

    Y = np.array([
        [0.3, 1.0, 2.0, 3.0]
    ])

    n_x = 2
    n_h1 = 4
    n_h2 = 5
    n_h3 = 3
    n_h4 = 2
    n_y = 1

    W1, b1, W2, b2, W3, b3, W4, b4, W, b = initialize_parameter(n_x,n_h1,n_h2,n_h3,n_h4,n_y)

    Z4, A4 = forward_pass(X,W1,b1,W2,b2,W3,b3,W4,b4)
    print("Hidden layer output: ", A4)

    Z,A = output_layer_prediction(Z4,A4,b,W)
    print("Output Layer output: ", A)

main()


