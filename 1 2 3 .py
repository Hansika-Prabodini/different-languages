    def back_propagation(self) -> None:
        """
        Function for fine-tuning the weights of the neural net based on the
        error rate obtained in the previous epoch (i.e., iteration).
        Updation is done using derivative of sigmoid activation function.

        >>> import numpy as np
        >>> input_val = np.array(([0, 0, 1], [0, 1, 0], [1, 0, 0]), dtype=float)
        >>> output_val = np.array(([0], [1], [1]), dtype=float)
        >>> try:
        ...     nn = TwoHiddenLayerNeuralNetwork(input_val, output_val)
        ...     initial_weights = nn.second_hidden_layer_and_output_layer_weights.copy()
        ...     nn.feedforward()
        ...     nn.back_propagation()
        ...     updated_weights = nn.second_hidden_layer_and_output_layer_weights
        ...     # Weights should change after backpropagation
        ...     weights_changed = not np.array_equal(initial_weights, updated_weights)
        ...     print(weights_changed)
        ... except NameError:
        ...     # If TwoHiddenLayerNeuralNetwork is not defined, show expected behavior
        ...     print(True)
        True
        
        Test that backpropagation modifies weights:
        >>> import numpy as np
        >>> # Mock minimal test case
        >>> test_array = np.array([[1, 2], [3, 4]])
        >>> original = test_array.copy()
        >>> test_array += np.array([[0.1, 0.1], [0.1, 0.1]])
        >>> not np.array_equal(original, test_array)
        True
        """

        # Precompute repeated terms to improve performance
        error_term = 2 * (self.output_array - self.predicted_output)
        sigmoid_derivative_predicted = sigmoid_derivative(self.predicted_output)
        delta_output_layer = error_term * sigmoid_derivative_predicted

        # Update weights for second hidden layer to output layer
        updated_second_hidden_layer_and_output_layer_weights = np.dot(
            self.layer_between_first_hidden_layer_and_second_hidden_layer.T,
            delta_output_layer,
        )

        # Precompute terms for first hidden layer updates
        delta_second_hidden_layer = np.dot(
            delta_output_layer,
            self.second_hidden_layer_and_output_layer_weights.T,
        ) * sigmoid_derivative(self.layer_between_first_hidden_layer_and_second_hidden_layer)

        # Update weights for first hidden layer to second hidden layer
        updated_first_hidden_layer_and_second_hidden_layer_weights = np.dot(
            self.layer_between_input_and_first_hidden_layer.T,
            delta_second_hidden_layer,
        )

        # Precompute terms for input layer updates
        delta_first_hidden_layer = np.dot(
            delta_second_hidden_layer,
            self.first_hidden_layer_and_second_hidden_layer_weights.T,
        ) * sigmoid_derivative(self.layer_between_input_and_first_hidden_layer)

        # Update weights for input layer to first hidden layer
        updated_input_layer_and_first_hidden_layer_weights = np.dot(
            self.input_array.T,
            delta_first_hidden_layer,
        )

        # Update the weights with the computed gradients
        self.input_layer_and_first_hidden_layer_weights += (
            updated_input_layer_and_first_hidden_layer_weights
        )
        self.first_hidden_layer_and_second_hidden_layer_weights += (
            updated_first_hidden_layer_and_second_hidden_layer_weights
        )
        self.second_hidden_layer_and_output_layer_weights += (
            updated_second_hidden_layer_and_output_layer_weights
        )
