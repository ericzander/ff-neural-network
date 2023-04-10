"""
This file contains a minimalistic implementation of an artificial neural
network capable of batch gradient descent.

Example:
    import ez_nn as nn

    network = nn.Network(loss="mse")
    network.insert(nn.Input(2))
    network.insert(nn.Connected(3, activation="relu"))
    network.insert(nn.Connected(2, activation="relu"))
    network.insert(nn.Connected(1, activation="sigmoid"))

    network.fit(train_X, train_y, epoch=200, learning_rate=2)

    pred = network.predict(test_X)

Author:
    Eric Zander
"""

from typing import List, Union, Callable

import numpy as np


# -------
# Network
# -------

class Network:
    """
    Artificial neural network capable of batch gradient descent.
    """
    def __init__(self, loss="mse"):
        self.layers: List[Layer] = []
        self.loss = self._create_loss(loss)

    def insert(self, layer):
        """
        Inserts a new layer into the network
        """
        if self.layers:
            layer.prev = self.layers[-1].size
        self.layers.append(layer)

    def predict(self, input_data):
        """
        Makes predictions for the given input data
        """
        output = input_data

        for layer in self.layers:
            output = layer(output)

        return output.flatten()

    def fit(self, input_data, output_data, learning_rate=0.2, epochs=10, logging=None):
        """
        Fits the network to the given input (X) and output (y) data.

        In addition to modifying the learning rate and number of epochs,
        a type of error logging may be specified to return a list of errors
        of the specified type for each epoch.

        Logging Types:
            "rmse": Root mean square error
        """
        # Do nothing if no layers
        if not self.layers:
            return

        # Reshape actual output for consistency
        odata = output_data.reshape(-1, 1)

        # Prepare to save error for each epoch if logging enabled
        error = []

        # For each epoch
        for i in range(epochs):
            # Update network and save output error if applicable
            err = self._update_network(input_data, odata, learning_rate, logging)

            # If logging error, save
            if logging is not None:
                error.append(err)

        return error

    # ---------------
    # Fitting Helpers
    # ---------------

    def _update_network(self, input_data, output_data, learning_rate, logging):
        # Calculate and save outputs for each layer
        outputs = self.__calc_outputs(input_data)

        # Calculate loss as starting error
        error = self.loss(outputs[-1], output_data)

        # Back propagation
        # Travel through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]

            # Quit if input layer reached
            if not isinstance(layer, Connected):
                return self.__log_error(outputs[-1], output_data, logging)

            # Calculate error w.r.t. weights and biases for the layer
            # Also update error to be used in preceding layer
            de_dw, de_db, error = self.__calc_errors(layer, outputs, error, i)

            # Update weights and biases according to learning rate
            layer.w -= learning_rate * de_dw
            layer.b -= learning_rate * de_db

    def __calc_outputs(self, input_data):
        """
        Calculates and saves outputs for each layer
        """
        outputs = []

        for layer in self.layers:
            if not outputs:
                outputs.append(layer(input_data))
            else:
                outputs.append(layer(outputs[-1]))

        return outputs

    @staticmethod
    def __calc_errors(layer, outputs, error, i):
        """
        Calculate error w.r.t. weights and biases for each sample:
            gr_w(E) = (gr_o_l(E) hadprod phi'(z + b)) @ o_L_T
            gr_b(E) = gr_o_l(E) hadprod phi'(z + b)

        Also calculates error for the current layer for use in learning in
        the preceding layer:
            gr_o_l-1(E) = w.T @ (gr_o_l(E) hadprod phi'(z + b))

        Expand_dims is used to accomodate multiple training samples at once.
        """
        # Save previous layer's outputs and transpose
        o = np.expand_dims(outputs[i - 1], axis=2)
        o_t = np.transpose(o, (0, 2, 1))

        # Calc this layer's inputs
        z = np.expand_dims(outputs[i - 1], axis=2)
        z = layer.w @ z

        # Expand this layer's error to accomodate multiple samples
        err = np.expand_dims(error, axis=2)

        # Compute change in error w.r.t. biases and weights
        de_db = np.multiply(err, layer.act.deriv(z + layer.b))
        de_dw = de_db @ o_t

        # Calculate error for previous layer (for next step of back prop)
        next_err = layer.w.T @ de_db
        next_err = np.squeeze(next_err, axis=2)

        # Average across samples and adjust to match layer's shapes
        de_db = np.mean(de_db, axis=0).flatten()
        de_dw = np.mean(de_dw, axis=0)

        return de_dw, de_db, next_err

    @staticmethod
    def __log_error(pred, targ, ltype):
        if ltype is None:
            return
        elif ltype == "rmse":
            return rmse(pred, targ)
        else:
            raise ValueError("Invalid error logging type.")

    # ------------
    # Loss Helpers
    # ------------

    def _create_loss(self, ltype):
        # TODO CCE
        if ltype == "mse":
            return self.__mse
        else:
            raise ValueError("Invalid loss function alias.")

    @staticmethod
    def __mse(pred, out):
        return pred - out


# ----------
# Activation
# ----------

class Activation:
    """
    Encapsulates activation functions and their derivatives
    """
    def __init__(self, fn: Callable, deriv: Callable):
        self.fn = fn
        self.deriv = deriv


# ------
# Layers
# ------

class Layer:
    """
    Base class layer
    """
    def __init__(self, size):
        self.prev = None  # Size of previous layer
        self.size = size  # Size of layer

    def __call__(self, input_data):
        return input_data


class Input(Layer):
    """
    Input layer
    """
    def __init__(self, size: int):
        super().__init__(size)


class Connected(Layer):
    """
    Fully connected layer
    """
    def __init__(self, size: float, activation: Union[None, str] = None):
        super().__init__(size)

        self.w: Union[None, np.ndarray] = None
        self.b: Union[None, np.ndarray] = None

        self.act: Activation = self._create_activation(activation)

    def __call__(self, input_data):
        """
        Evaluates layer output for the given input data
        """
        # Initialize weights and biases if first evaluation
        if self.w is None:
            self.w = np.random.rand(self.size, self.prev)
            self.b = np.random.rand(self.size)

        # Evaluate based on shape of input data
        if input_data.ndim == 1:
            out = self._eval(input_data)
        elif input_data.ndim == 2:
            out = np.apply_along_axis(self._eval, 1, input_data)
        else:
            raise ValueError("Invalid input shape.")

        # Return neuron outputs
        return out

    def _eval(self, input_row):
        return self.act.fn((self.w @ input_row) + self.b)

    @staticmethod
    def _create_activation(atype):
        # TODO Softmax activation
        if atype is None or atype == "relu":
            return Activation(lambda x: x, lambda x: 1)
        elif atype == "sigmoid":
            return Activation(lambda x: 1 / (1 + np.exp(-x)),
                              lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2)
        else:
            raise ValueError("Invalid activation function alias.")

# ---------------
# Other Utilities
# ---------------

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))
