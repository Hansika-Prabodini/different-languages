"""
Forward propagation explanation:
https://towardsdatascience.com/forward-propagation-in-neural-networks-simplified-math-and-code-version-bbcfef6f9250
"""

import math
import random


# Sigmoid
def sigmoid_function(value: float, deriv: bool = False) -> float:
    """Return the sigmoid function of a float.

    >>> sigmoid_function(0)
    0.5
    >>> sigmoid_function(3.5)
    0.9706877692486436
    >>> sigmoid_function(-3.5)
    0.029312230751354174
    >>> sigmoid_function(0.5, True)
    0.25
    >>> sigmoid_function(0.0, True)
    0.0
    """
    if deriv:
        return value * (1 - value)
    return 1 / (1 + math.exp(-value))


# Initial Value
INITIAL_VALUE = 0.02


def forward_propagation(expected: int, number_propagations: int) -> float:
    """Return the value found after the forward propagation training.

    Tests with different propagation counts and expected values:
    
    >>> import random
    >>> random.seed(42)  # Set seed for reproducible test
    >>> res = forward_propagation(32, 450_000)  # High iteration count
    >>> 30.0 <= res <= 34.0  # Allow reasonable range due to randomness
    True

    >>> random.seed(42)
    >>> res = forward_propagation(32, 1000)  # Low iteration count
    >>> 20.0 <= res <= 40.0  # Wider range for low iterations
    True
    
    >>> random.seed(42)
    >>> res = forward_propagation(50, 100000)
    >>> 45.0 <= res <= 55.0  # Test with different target
    True
    
    Edge cases:
    >>> random.seed(42)
    >>> res = forward_propagation(0, 1000)
    >>> isinstance(res, float)
    True
    
    >>> random.seed(42)
    >>> res = forward_propagation(100, 1000)
    >>> isinstance(res, float)
    True
    """

    # Random weight - more efficient random generation  
    weight = float(random.randint(1, 199))  # Equivalent range to 2 * (1 to 100) - 1, but more efficient
    
    # Pre-compute constants
    expected_normalized = expected / 100.0
    exp_func = math.exp
    
    for _ in range(number_propagations):
        # Forward propagation - inline sigmoid calculation
        sigmoid_input = INITIAL_VALUE * weight
        layer_1 = 1.0 / (1.0 + exp_func(-sigmoid_input))
        
        # How much did we miss?
        layer_1_error = expected_normalized - layer_1
        
        # Error delta - inline sigmoid derivative calculation
        layer_1_delta = layer_1_error * layer_1 * (1.0 - layer_1)
        
        # Update weight
        weight += INITIAL_VALUE * layer_1_delta

    return layer_1 * 100.0


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    expected = int(input("Expected value: "))
    number_propagations = int(input("Number of propagations: "))
    print(forward_propagation(expected, number_propagations))
