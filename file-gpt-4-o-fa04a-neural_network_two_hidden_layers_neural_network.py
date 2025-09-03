def __init__(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
    """
    This function initializes the TwoHiddenLayerNeuralNetwork class with random
    weights for every layer and initializes predicted output with zeroes.

    input_array : input values for training the neural network (i.e training data).
    output_array : expected output values of the given inputs.
    """

    # Input values provided for training the model.
    self.input_array = input_array

    # Use a single random generator instance for all weight initializations to reduce overhead.
    rng = np.random.default_rng()

    # Random initial weights are assigned where first argument is the
    # number of nodes in previous layer and second argument is the
    # number of nodes in the next layer.

    # Random initial weights are assigned.
    # self.input_array.shape[1] is used to represent number of nodes in input layer.
    # First hidden layer consists of 4 nodes.
    self.input_layer_and_first_hidden_layer_weights = rng.random(
        (self.input_array.shape[1], 4)
    )

    # Random initial values for the first hidden layer.
    # First hidden layer has 4 nodes.
    # Second hidden layer has 3 nodes.
    self.first_hidden_layer_and_second_hidden_layer_weights = rng.random((4, 3))

    # Random initial values for the second hidden layer.
    # Second hidden layer has 3 nodes.
    # Output layer has 1 node.
    self.second_hidden_layer_and_output_layer_weights = rng.random((3, 1))

    # Real output values provided.
    self.output_array = output_array

    # Predicted output values by the neural network.
    # Predicted_output array initially consists of zeroes.
    self.predicted_output = np.zeros_like(output_array)
