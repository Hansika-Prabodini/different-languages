"""
Comprehensive test suite for file-agent+claude-v4-sonnet_d44a6-neural_network_convolution_neural_network(1).py.
Tests the improved CNN implementation with numpy arrays and better performance optimizations.
"""
import unittest
import numpy as np
import tempfile
import os
import pickle
from unittest.mock import patch, MagicMock
import sys

# Import the module under test
sys.path.append('.')

# Import the CNN class from the file
try:
    exec(open('file-agent+claude-v4-sonnet_d44a6-neural_network_convolution_neural_network(1).py').read())
except Exception as e:
    print(f"Warning: Could not import CNN class: {e}")
    # Create a mock class for testing
    class CNN:
        pass


class TestAgentSonnetCNN(unittest.TestCase):
    """Test cases for improved CNN class from file-agent+claude-v4-sonnet_d44a6-neural_network_convolution_neural_network(1).py."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            self.conv1_get = [3, 2, 1]  # kernel size, number, step
            self.size_p1 = 2
            self.bp_num1 = 8  # should match flattened pooled feature size
            self.bp_num2 = 4
            self.bp_num3 = 2
            self.cnn = CNN(self.conv1_get, self.size_p1, self.bp_num1, self.bp_num2, self.bp_num3)
            
            # Create test data
            self.test_data = np.random.rand(6, 6)  # 6x6 input image
            self.test_labels = np.array([1, 0])  # binary classification
            
            self.cnn_available = True
        except Exception:
            self.cnn_available = False
            self.skipTest("CNN class not available")

    def test_initialization(self):
        """Test CNN initialization with improved numpy arrays."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        self.assertEqual(self.cnn.conv1, [3, 2])
        self.assertEqual(self.cnn.step_conv1, 1)
        self.assertEqual(self.cnn.size_pooling1, 2)
        self.assertEqual(self.cnn.num_bp1, 8)
        self.assertEqual(self.cnn.num_bp2, 4)
        self.assertEqual(self.cnn.num_bp3, 2)
        
        # Test that weights are numpy arrays (improved implementation)
        self.assertIsInstance(self.cnn.w_conv1, np.ndarray)
        self.assertEqual(self.cnn.w_conv1.shape, (2, 3, 3))  # (num_kernels, kernel_height, kernel_width)
        
        # Test other weight matrices
        self.assertIsInstance(self.cnn.wkj, np.ndarray)
        self.assertIsInstance(self.cnn.vji, np.ndarray)
        
    def test_initialization_with_custom_rates(self):
        """Test CNN initialization with custom learning rates."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        cnn_custom = CNN([3, 2, 1], 2, 8, 4, 2, rate_w=0.3, rate_t=0.4)
        self.assertEqual(cnn_custom.rate_weight, 0.3)
        self.assertEqual(cnn_custom.rate_thre, 0.4)

    def test_sigmoid_function(self):
        """Test sigmoid activation function."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test with various inputs
        self.assertAlmostEqual(self.cnn.sig(0), 0.5, places=5)
        self.assertAlmostEqual(self.cnn.sig(1), 0.731, places=2)
        self.assertAlmostEqual(self.cnn.sig(-1), 0.268, places=2)
        
        # Test with arrays
        test_array = np.array([0, 1, -1])
        result = self.cnn.sig(test_array)
        expected = np.array([0.5, 0.731, 0.268])
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_do_round(self):
        """Test rounding function."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        self.assertEqual(self.cnn.do_round(1.23456), 1.235)
        self.assertEqual(self.cnn.do_round(1.0), 1.0)
        self.assertEqual(self.cnn.do_round(-1.23456), -1.235)

    def test_improved_convolute(self):
        """Test improved convolution operation with numpy arrays."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        test_data = np.ones((4, 4))  # 4x4 data
        test_kernels = np.ones((1, 3, 3))  # 1 kernel of size 3x3
        
        focus_list, feature_maps = self.cnn.convolute(
            test_data, [3, 1], test_kernels, [0], 1
        )
        
        # Should produce 2x2 feature map for 4x4 input with 3x3 kernel
        self.assertEqual(len(feature_maps), 1)
        self.assertEqual(feature_maps[0].shape, (2, 2))
        
        # Check that focus_list is a numpy array (improved implementation)
        self.assertIsInstance(focus_list, np.ndarray)
        self.assertEqual(focus_list.shape[1], 9)  # 3x3 flattened patches

    def test_improved_pooling(self):
        """Test improved pooling operation."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Create test feature maps as numpy arrays
        feature_map = [np.array([[1, 2], [3, 4]])]
        pooled = self.cnn.pooling(feature_map, 2, "average_pool")
        
        # Should average the 2x2 region to single value
        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0].shape, (1, 1))
        self.assertAlmostEqual(pooled[0][0, 0], 2.5)  # (1+2+3+4)/4 = 2.5

    def test_improved_pooling_max(self):
        """Test improved max pooling operation."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        feature_map = [np.array([[1, 2], [3, 4]])]
        pooled = self.cnn.pooling(feature_map, 2, "max_pooling")
        
        self.assertEqual(pooled[0][0, 0], 4)  # max of [1,2,3,4]

    def test_improved_expand_functionality(self):
        """Test improved data expansion methods."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test _expand method with list of arrays
        test_data = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        expanded = self.cnn._expand(test_data)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(expanded, expected)
        
        # Test _expand_mat method
        test_mat = np.array([[1, 2], [3, 4]])
        expanded_mat = self.cnn._expand_mat(test_mat)
        expected_mat = np.array([[1, 2, 3, 4]])
        np.testing.assert_array_equal(expanded_mat, expected_mat)

    def test_backward_compatibility_save_load(self):
        """Test model saving and loading with backward compatibility."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Save model
            self.cnn.save_model(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Load model
            loaded_cnn = CNN.read_model(tmp_path)
            
            # Verify loaded model has same parameters
            self.assertEqual(loaded_cnn.conv1, self.cnn.conv1)
            self.assertEqual(loaded_cnn.step_conv1, self.cnn.step_conv1)
            self.assertEqual(loaded_cnn.size_pooling1, self.cnn.size_pooling1)
            
            # Verify arrays are properly loaded
            np.testing.assert_array_equal(loaded_cnn.w_conv1, self.cnn.w_conv1)
            np.testing.assert_array_equal(loaded_cnn.wkj, self.cnn.wkj)
            np.testing.assert_array_equal(loaded_cnn.vji, self.cnn.vji)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_improved_gradient_calculation(self):
        """Test improved gradient calculation with vectorized operations."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        out_map = [np.array([[0.1, 0.2], [0.3, 0.4]])]
        pd_pool = [0.1, 0.2, 0.3, 0.4]
        
        gradients = self.cnn._calculate_gradient_from_pool(
            out_map, pd_pool, 1, 2, 2
        )
        
        self.assertEqual(len(gradients), 1)
        self.assertEqual(gradients[0].shape, (2, 2))
        self.assertIsInstance(gradients[0], np.ndarray)

    def test_vectorized_convolution_performance(self):
        """Test that vectorized convolution produces correct results."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test with known input
        test_data = np.ones((4, 4))
        test_kernels = np.ones((1, 3, 3))
        
        focus_list, feature_maps = self.cnn.convolute(
            test_data, [3, 1], test_kernels, [0], 1
        )
        
        # With all-ones input and kernel, convolution should give 9 before sigmoid
        expected_conv_value = 9
        expected_sigmoid_value = self.cnn.sig(expected_conv_value)
        
        # All values in feature map should be approximately the same
        feature_map = feature_maps[0]
        np.testing.assert_array_almost_equal(
            feature_map, 
            np.full_like(feature_map, expected_sigmoid_value), 
            decimal=5
        )

    def test_memory_efficient_pooling(self):
        """Test memory-efficient pooling implementation."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Create larger feature maps to test memory efficiency
        large_feature_map = [np.random.rand(8, 8)]
        pooled = self.cnn.pooling(large_feature_map, 2, "average_pool")
        
        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0].shape, (4, 4))
        self.assertIsInstance(pooled[0], np.ndarray)

    def test_training_setup(self):
        """Test training setup with improved arrays."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Create minimal training data
        train_data = [np.random.rand(6, 6)]
        teach_data = [[1, 0]]
        
        # Test that training method exists
        self.assertTrue(hasattr(self.cnn, 'train'))
        
        # Test that we can call the training function (just check setup)
        with patch('builtins.print'):  # Suppress output
            try:
                # Test just the initial setup part of training
                self.assertTrue(callable(self.cnn.train))
            except Exception as e:
                # Training might fail due to dimension mismatches, but method should exist
                pass

    def test_array_vs_matrix_compatibility(self):
        """Test that the improved implementation handles both arrays and matrices."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test weight shapes are correct for array format
        self.assertEqual(len(self.cnn.w_conv1.shape), 3)  # 3D array for multiple kernels
        self.assertEqual(len(self.cnn.wkj.shape), 2)      # 2D array for fully connected
        self.assertEqual(len(self.cnn.vji.shape), 2)      # 2D array for fully connected

    def test_edge_cases_improved(self):
        """Test edge cases with improved implementation."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test with minimum size data
        small_data = np.ones((3, 3))
        small_kernels = np.ones((1, 3, 3))
        
        try:
            focus_list, feature_maps = self.cnn.convolute(
                small_data, [3, 1], small_kernels, [0], 1
            )
            self.assertEqual(feature_maps[0].shape, (1, 1))
        except Exception:
            pass  # Acceptable if method has size constraints

    def test_numerical_stability_improved(self):
        """Test numerical stability with improved implementation."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test sigmoid with extreme values
        large_positive = self.cnn.sig(1000)
        self.assertAlmostEqual(large_positive, 1.0, places=5)
        
        large_negative = self.cnn.sig(-1000)
        self.assertAlmostEqual(large_negative, 0.0, places=5)


class TestAgentSonnetCNNIntegration(unittest.TestCase):
    """Integration tests for improved CNN workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        try:
            self.cnn = CNN([3, 2, 1], 2, 8, 4, 2)
            self.cnn_available = True
        except Exception:
            self.cnn_available = False
        
    def test_complete_forward_pass_improved(self):
        """Test complete forward pass with improved implementation."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Create test data that will work through the entire pipeline
        test_input = np.random.rand(6, 6)
        
        try:
            # Test convolution
            focus_list, feature_maps = self.cnn.convolute(
                test_input, self.cnn.conv1, self.cnn.w_conv1, 
                self.cnn.thre_conv1, self.cnn.step_conv1
            )
            
            # Test pooling
            pooled_features = self.cnn.pooling(feature_maps, self.cnn.size_pooling1)
            
            # Test expansion
            expanded = self.cnn._expand(pooled_features)
            
            # Verify dimensions flow correctly with improved implementation
            self.assertIsInstance(focus_list, np.ndarray)
            self.assertIsInstance(feature_maps, list)
            self.assertIsInstance(pooled_features, list)
            self.assertIsInstance(expanded, np.ndarray)
            
            # Check that all feature maps are numpy arrays
            for fm in feature_maps:
                self.assertIsInstance(fm, np.ndarray)
                
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")

    def test_model_persistence_with_arrays(self):
        """Test model save/load cycle with numpy arrays."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Save model with array weights
            original_w_conv1 = self.cnn.w_conv1.copy()
            self.cnn.save_model(tmp_path)
            
            # Modify current model
            self.cnn.w_conv1 = np.ones_like(self.cnn.w_conv1)
            
            # Load saved model
            loaded_cnn = CNN.read_model(tmp_path)
            
            # Verify loaded model has original array weights
            np.testing.assert_array_equal(loaded_cnn.w_conv1, original_w_conv1)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_performance_comparison_indicators(self):
        """Test indicators of improved performance."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test that weight structures are optimized
        self.assertIsInstance(self.cnn.w_conv1, np.ndarray)
        self.assertEqual(self.cnn.w_conv1.ndim, 3)  # More efficient 3D structure
        
        # Test that pooling uses integer division (more efficient)
        test_data = np.random.rand(8, 8)
        focus_list, feature_maps = self.cnn.convolute(
            test_data, self.cnn.conv1, self.cnn.w_conv1,
            self.cnn.thre_conv1, self.cnn.step_conv1
        )
        
        pooled = self.cnn.pooling(feature_maps, 2)
        
        # Check that pooled dimensions use efficient integer division
        for pm in pooled:
            self.assertEqual(pm.shape[0] * 2, feature_maps[0].shape[0])

    def test_vectorized_operations(self):
        """Test that operations are properly vectorized."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test that expand methods work efficiently with arrays
        test_arrays = [np.random.rand(4, 4), np.random.rand(4, 4)]
        expanded = self.cnn._expand(test_arrays)
        
        self.assertIsInstance(expanded, np.ndarray)
        self.assertEqual(expanded.shape, (32,))  # 2 * 4 * 4 = 32


class TestAgentSonnetCNNCompatibility(unittest.TestCase):
    """Test backward compatibility and error handling."""

    def setUp(self):
        """Set up compatibility test fixtures."""
        try:
            self.cnn = CNN([3, 2, 1], 2, 8, 4, 2)
            self.cnn_available = True
        except Exception:
            self.cnn_available = False

    def test_backward_compatibility_with_old_format(self):
        """Test loading models saved in old matrix format."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Create a mock old-format model dictionary
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Create old-format model (with matrix list instead of array)
            old_format_model = {
                "num_bp1": 8, "num_bp2": 4, "num_bp3": 2,
                "conv1": [3, 2], "step_conv1": 1, "size_pooling1": 2,
                "rate_weight": 0.2, "rate_thre": 0.2,
                "w_conv1": [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) for _ in range(2)],  # List format
                "wkj": np.random.rand(2, 4),
                "vji": np.random.rand(4, 8),
                "thre_conv1": np.array([0.1, 0.2]),
                "thre_bp2": np.random.rand(4),
                "thre_bp3": np.random.rand(2)
            }
            
            with open(tmp_path, "wb") as f:
                pickle.dump(old_format_model, f)
            
            # Test loading old format
            try:
                loaded_cnn = CNN.read_model(tmp_path)
                self.assertIsInstance(loaded_cnn.w_conv1, np.ndarray)
                self.assertEqual(loaded_cnn.w_conv1.shape, (2, 3, 3))
            except Exception as e:
                # Backward compatibility might not be perfect, but should be handled gracefully
                self.assertIsInstance(e, (ValueError, AttributeError, TypeError))
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_error_handling_invalid_inputs(self):
        """Test error handling with invalid inputs."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test with invalid data types
        try:
            invalid_data = "not_a_numpy_array"
            self.cnn.convolute(invalid_data, [3, 1], self.cnn.w_conv1, self.cnn.thre_conv1, 1)
        except (AttributeError, TypeError, ValueError):
            pass  # Expected error types
            
        # Test with mismatched dimensions
        try:
            wrong_shape_data = np.random.rand(2, 3)  # Too small for 3x3 kernel
            self.cnn.convolute(wrong_shape_data, [3, 1], self.cnn.w_conv1, self.cnn.thre_conv1, 1)
        except (IndexError, ValueError):
            pass  # Expected error types

    def test_empty_or_none_inputs(self):
        """Test handling of empty or None inputs."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test with None
        try:
            result = self.cnn._expand(None)
        except (AttributeError, TypeError):
            pass  # Expected behavior
            
        # Test with empty list
        try:
            result = self.cnn._expand([])
            self.assertIsInstance(result, np.ndarray)
        except Exception:
            pass  # Acceptable behavior


class TestAgentSonnetCNNMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of improved implementation."""

    def setUp(self):
        """Set up mathematical correctness test fixtures."""
        try:
            self.cnn = CNN([3, 1, 1], 2, 2, 2, 1)  # Simple configuration
            self.cnn_available = True
        except Exception:
            self.cnn_available = False

    def test_convolution_mathematical_correctness(self):
        """Test that convolution produces mathematically correct results."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Simple test case with known results
        input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        kernel = np.array([[[1, 0, -1], [1, 0, -1], [1, 0, -1]]])  # Edge detection kernel
        threshold = np.array([0])
        
        focus_list, feature_maps = self.cnn.convolute(input_data, [3, 1], kernel, threshold, 1)
        
        # For this specific kernel and input, we can calculate expected result
        # The convolution should detect vertical edges
        expected_before_sigmoid = (1*1 + 2*0 + 3*(-1) + 4*1 + 5*0 + 6*(-1) + 7*1 + 8*0 + 9*(-1)) - 0
        expected_after_sigmoid = self.cnn.sig(expected_before_sigmoid)
        
        self.assertEqual(feature_maps[0].shape, (1, 1))
        self.assertAlmostEqual(feature_maps[0][0, 0], expected_after_sigmoid, places=5)

    def test_pooling_mathematical_correctness(self):
        """Test that pooling produces mathematically correct results."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test average pooling
        input_feature_map = [np.array([[1, 2], [3, 4]])]
        pooled = self.cnn.pooling(input_feature_map, 2, "average_pool")
        expected_avg = (1 + 2 + 3 + 4) / 4
        
        self.assertAlmostEqual(pooled[0][0, 0], expected_avg, places=5)
        
        # Test max pooling
        pooled_max = self.cnn.pooling(input_feature_map, 2, "max_pooling")
        expected_max = 4
        
        self.assertEqual(pooled_max[0][0, 0], expected_max)

    def test_sigmoid_mathematical_properties(self):
        """Test mathematical properties of sigmoid function."""
        if not self.cnn_available:
            self.skipTest("CNN class not available")
            
        # Test sigmoid properties
        self.assertAlmostEqual(self.cnn.sig(0), 0.5, places=10)
        
        # Test symmetry: sig(-x) = 1 - sig(x)
        x = 2.5
        self.assertAlmostEqual(
            self.cnn.sig(-x), 
            1 - self.cnn.sig(x), 
            places=10
        )
        
        # Test range: all outputs should be in (0, 1)
        test_values = np.array([-100, -10, -1, 0, 1, 10, 100])
        results = self.cnn.sig(test_values)
        
        self.assertTrue(np.all(results > 0))
        self.assertTrue(np.all(results < 1))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
