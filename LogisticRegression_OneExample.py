
import Computation_Graph

def main():
    # One training example with two features
    x1 = 2.0
    x2 = 3.0

    # One weight corresponding to each feature
    w1 = 0.1
    w2 = -0.2

    bias = 0.5

    # Actual correct answer from the dataset
    y = 1

    prediction, loss = Computation_Graph.forward_propagation(
        w1=w1,
        w2=w2,
        x1=x1,
        x2=x2,
        bias=bias,
        y=y
    )

    dw1, dw2, db = Computation_Graph.back_propagation(
        prediction=prediction,
        y=y,
        x1=x1,
        x2=x2
    )


if __name__ == "__main__":
    main()
