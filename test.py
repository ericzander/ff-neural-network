"""
Testing script that creates a neural network to be trained and tested with
data generated as follows:

    [x, y, x * y] where 0 <= x < 1 and 0 <= y < 1

Author:
    Eric Zander
"""

import numpy as np
import matplotlib.pyplot as plt

import ez_nn as nn


def main():
    # Create network
    network = nn.Network(loss="mse")
    network.insert(nn.Input(2))
    network.insert(nn.Connected(4, activation="relu"))
    network.insert(nn.Connected(1, activation="relu"))

    # Generate data
    training_data = generate_data(500)
    train_X, train_y = training_data[:, :2], training_data[:, 2]
    testing_data = generate_data(100)
    test_X, test_y = testing_data[:, :2], testing_data[:, 2]

    # Fit to training data
    error = network.fit(train_X, train_y, learning_rate=0.2,
                        epochs=200, logging="rmse")

    # Make predictions and print final RMSE
    pred = network.predict(test_X)
    final_RMSE = nn.rmse(pred, test_y)
    print(f"Final RMSE   : {final_RMSE}")
    print(f"NRMSE        : {final_RMSE / (np.max(test_y) - np.min(test_y))}")

    # Plot RMSE during fitting and final prediction RMSE
    plt.plot(error, label="Training RMSE")
    plt.axhline(final_RMSE, ls="--", label="Testing RMSE", color="orange")
    plt.title("RMSE by Epoch")
    plt.grid()
    plt.legend()
    plt.show()


def generate_data(m):
    """
    Generates m input-output triplets of the form:
        [x_i, y_j, x_i * y_j]
    """
    xy = np.random.rand(m, 2)
    f = np.prod(xy, axis=1).reshape(m, 1)

    return np.append(xy, f, axis=1)


if __name__ == "__main__":
    main()
