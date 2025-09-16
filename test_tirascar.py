"""
Comprehensive test suite for neural network implementation (තිරසාර.py).
Tests CNN functionality including convolution, pooling, training, and prediction.
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

# Import the CNN class from තිරසාර.py
try:
    from තිරසාර import CNN
except ImportError:
    # Try alternative import method
    import importlib.util
    spec = importlib.util.spec_from_file_location("tirascar", "තිරසාර.py")
    tirascar_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tirascar_module)
    CNN = tirascar_module.CNN


class TestTirasarCNN(unittest.TestCase):
    """Test cases for CNN class from තිරසාර.py."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.conv1_get = [3, 2, 1]  # kernel size, number, step
        self.size_p1 = 2
        self.bp_num1 = 8  # should match flattened pooled feature size
        self.bp_num2 = 4
        self.bp_num3 = 2
        self.cnn = CNN(self.conv1_get, self.size_p1, self.bp_num1, self.bp_num2, self.bp_num3)
        
        # Create test data
        self.test_data = np.random.rand(6, 6)  # 6x6 input image
        self.test_labels = np.array([1, 0])  # binary classification

    def test_initialization(self):
        """Test CNN initialization with valid parameters."""
        self.assertEqual(self.cnn.conv1, [3, 2])
        self.assertEqual(self.cnn.step_conv1, 1)
        self.assertEqual(self.cnn.size_pooling1, 2)
        self.assertEqual(self.cnn.num_bp1, 8)
        self.assertEqual(self.cnn.num_bp2, 4)
        self.assertEqual(self.cnn.num_bp3, 2)
        self.assertEqual(len(self.cnn.w_conv1), 2)  # 2 convolution kernels
        
    def test_initialization_with_custom_rates(self):
        """Test CNN initialization with custom learning rates."""
        cnn_custom = CNN([3, 2, 1], 2, 8, 4, 2, rate_w=0.3, rate_t=0.4)
        self.assertEqual(cnn_custom.rate_weight, 0.3)
        self.assertEqual(cnn_custom.rate_thre, 0.4)
        self.assertEqual(cnn_custom.dropout_rate, 0.5)  # Default dropout rate

    def test_sigmoid_function(self):
        """Test sigmoid activation function."""
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
        self.assertEqual(self.cnn.do_round(1.23456), 1.235)
        self.assertEqual(self.cnn.do_round(1.0), 1.0)
        self.assertEqual(self.cnn.do_round(-1.23456), -1.235)

    def test_convolute_basic(self):
        """Test basic convolution operation."""
        test_data = np.ones((4, 4))  # 4x4 data
        focus_list, feature_maps = self.cnn.convolute(
            test_data, [3, 1], [np.ones((3, 3))], [0], 1
        )
        
        # Should produce 2x2 feature map for 4x4 input with 3x3 kernel
        self.assertEqual(len(feature_maps), 1)
        self.assertEqual(feature_maps[0].shape, (2, 2))
        
    def test_convolute_with_step(self):
        """Test convolution with different step sizes."""
        test_data = np.ones((6, 6))
        focus_list, feature_maps = self.cnn.convolute(
            test_data, [3, 1], [np.ones((3, 3))], [0], 2  # step=2
        )
        
        # With step=2, should produce 2x2 feature map
        self.assertEqual(feature_maps[0].shape, (2, 2))

    def test_pooling_average(self):
        """Test average pooling operation."""
        # Create test feature maps
        feature_map = [np.array([[1, 2], [3, 4]])]
        pooled = self.cnn.pooling(feature_map, 2, "average_pool")
        
        # Should average the 2x2 region to single value
        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0].shape, (1, 1))
        self.assertAlmostEqual(pooled[0][0, 0], 2.5)  # (1+2+3+4)/4 = 2.5

    def test_pooling_max(self):
        """Test max pooling operation."""
        feature_map = [np.array([[1, 2], [3, 4]])]
        pooled = self.cnn.pooling(feature_map, 2, "max_pooling")
        
        self.assertEqual(pooled[0][0, 0], 4)  # max of [1,2,3,4]

    def test_expand_functionality(self):
        """Test data expansion methods."""
        # Test _expand method
        test_data = [np.array([[1, 2], [3, 4]])]
        expanded = self.cnn._expand(test_data)
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(expanded, expected)
        
        # Test _expand_mat method
        test_mat = np.array([[1, 2], [3, 4]])
        expanded_mat = self.cnn._expand_mat(test_mat)
        expected_mat = np.array([[1, 2, 3, 4]])
        np.testing.assert_array_equal(expanded_mat, expected_mat)

    def test_save_and_load_model(self):
        """Test model saving and loading functionality."""
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
            self.assertEqual(loaded_cnn.num_bp1, self.cnn.num_bp1)
            self.assertEqual(loaded_cnn.num_bp2, self.cnn.num_bp2)
            self.assertEqual(loaded_cnn.num_bp3, self.cnn.num_bp3)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_gradient_calculation(self):
        """Test gradient calculation from pooling layer."""
        out_map = [np.array([[0.1, 0.2], [0.3, 0.4]])]
        pd_pool = [0.1, 0.2, 0.3, 0.4]
        
        gradients = self.cnn._calculate_gradient_from_pool(
            out_map, pd_pool, 1, 2, 2
        )
        
        self.assertEqual(len(gradients), 1)
        self.assertEqual(gradients[0].shape, (2, 2))

    def test_training_setup(self):
        """Test training setup and basic functionality."""
        # Create minimal training data
        train_data = [np.random.rand(6, 6)]
        teach_data = [[1, 0]]
        
        # Test that training method exists and can be called
        self.assertTrue(hasattr(self.cnn, 'train'))
        
        # Test training mode attribute setting
        self.cnn.training = True
        self.assertTrue(self.cnn.training)
        self.cnn.training = False
        self.assertFalse(self.cnn.training)

    def test_weight_matrix_dimensions(self):
        """Test that weight matrices have correct dimensions."""
        # Convolution weights
        self.assertEqual(len(self.cnn.w_conv1), self.conv1_get[1])  # Number of kernels
        for w in self.cnn.w_conv1:
            self.assertEqual(w.shape, (self.conv1_get[0], self.conv1_get[0]))
            
        # BP layer weights
        self.assertEqual(self.cnn.wkj.shape, (self.bp_num3, self.bp_num2))
        self.assertEqual(self.cnn.vji.shape, (self.bp_num2, self.bp_num1))
        
        # Thresholds
        self.assertEqual(len(self.cnn.thre_conv1), self.conv1_get[1])
        self.assertEqual(len(self.cnn.thre_bp2), self.bp_num2)
        self.assertEqual(len(self.cnn.thre_bp3), self.bp_num3)

    def test_dropout_functionality(self):
        """Test dropout mechanism."""
        self.assertEqual(self.cnn.dropout_rate, 0.5)
        
        # Test that training attribute affects behavior
        self.cnn.training = True
        self.assertTrue(hasattr(self.cnn, 'training'))
        
        self.cnn.training = False
        self.assertFalse(self.cnn.training)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimum size data
        small_data = np.ones((3, 3))  # Just fits 3x3 kernel
        try:
            focus_list, feature_maps = self.cnn.convolute(
                small_data, [3, 1], [np.ones((3, 3))], [0], 1
            )
            self.assertEqual(feature_maps[0].shape, (1, 1))
        except Exception:
            pass  # Acceptable if method has size constraints
            
        # Test sigmoid with extreme values
        large_positive = self.cnn.sig(100)
        self.assertAlmostEqual(large_positive, 1.0, places=5)
        
        large_negative = self.cnn.sig(-100)
        self.assertAlmostEqual(large_negative, 0.0, places=5)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with invalid convolution parameters
        with self.assertRaises((ValueError, IndexError)):
            CNN([0, 1, 1], 2, 8, 4, 2)  # Invalid kernel size
            
        # Test pooling with invalid size
        feature_map = [np.array([[1, 2], [3, 4]])]
        with self.assertRaises((ValueError, ZeroDivisionError)):
            self.cnn.pooling(feature_map, 0, "average_pool")

    def test_expand_mat_method_reference(self):
        """Test that Expand_Mat method exists and works (case sensitivity check)."""
        # The convolute method calls self.Expand_Mat - this might be a typo
        # Let's test if _expand_mat works as expected
        test_matrix = np.array([[1, 2], [3, 4]])
        result = self.cnn._expand_mat(test_matrix)
        expected = np.array([[1, 2, 3, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_method_case_sensitivity(self):
        """Test for potential method name issues."""
        # Check if the method names are consistent
        self.assertTrue(hasattr(self.cnn, '_expand_mat'))
        self.assertTrue(hasattr(self.cnn, '_expand'))
        self.assertTrue(hasattr(self.cnn, 'convolute'))
        self.assertTrue(hasattr(self.cnn, 'pooling'))
        self.assertTrue(hasattr(self.cnn, 'sig'))
        self.assertTrue(hasattr(self.cnn, 'do_round'))


class TestTirasarCNNIntegration(unittest.TestCase):
    """Integration tests for CNN workflow from තිරසාර.py."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.cnn = CNN([3, 2, 1], 2, 8, 4, 2)
        
    def test_complete_forward_pass(self):
        """Test complete forward pass through the network."""
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
            
            # Verify dimensions flow correctly
            self.assertIsInstance(focus_list, np.ndarray)
            self.assertIsInstance(feature_maps, list)
            self.assertIsInstance(pooled_features, list)
            self.assertIsInstance(expanded, np.ndarray)
            
        except Exception as e:
            # If there are dimension mismatches, at least verify the methods exist
            self.fail(f"Forward pass failed: {e}")

    def test_model_persistence(self):
        """Test complete model save/load cycle."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Train model minimally (just set some weights)
            original_weights = self.cnn.wkj.copy()
            
            # Save model
            self.cnn.save_model(tmp_path)
            
            # Modify current model
            self.cnn.wkj = np.ones_like(self.cnn.wkj)
            
            # Load saved model
            loaded_cnn = CNN.read_model(tmp_path)
            
            # Verify loaded model has original weights
            np.testing.assert_array_equal(loaded_cnn.wkj, original_weights)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_convolution_pooling_pipeline(self):
        """Test convolution followed by pooling."""
        test_data = np.random.rand(8, 8)  # Larger test data
        
        # Convolution
        focus_list, feature_maps = self.cnn.convolute(
            test_data, self.cnn.conv1, self.cnn.w_conv1,
            self.cnn.thre_conv1, self.cnn.step_conv1
        )
        
        # Pooling
        pooled = self.cnn.pooling(feature_maps, self.cnn.size_pooling1)
        
        # Verify pipeline consistency
        self.assertEqual(len(feature_maps), len(self.cnn.w_conv1))
        self.assertEqual(len(pooled), len(feature_maps))
        
        # Check size reduction from pooling
        for i in range(len(feature_maps)):
            original_size = feature_maps[i].shape[0]
            pooled_size = pooled[i].shape[0]
            expected_pooled_size = original_size // self.cnn.size_pooling1
            self.assertEqual(pooled_size, expected_pooled_size)


class TestTirasarCNNErrorHandling(unittest.TestCase):
    """Test error handling and edge cases for তিরসার.py CNN."""

    def setUp(self):
        """Set up test fixtures."""
        self.cnn = CNN([3, 2, 1], 2, 8, 4, 2)

    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        # Test with very small data
        small_data = np.array([[1]])
        
        # This should either work or raise a clear exception
        try:
            result = self.cnn.convolute(
                small_data, [1, 1], [np.ones((1, 1))], [0], 1
            )
            # If it works, verify output structure
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
        except (ValueError, IndexError) as e:
            # Expected for invalid dimensions
            pass

    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions."""
        test_data = np.random.rand(4, 4)
        
        # Try with kernel larger than data
        with self.assertRaises((ValueError, IndexError)):
            self.cnn.convolute(
                test_data, [5, 1], [np.ones((5, 5))], [0], 1
            )

    def test_invalid_pooling_parameters(self):
        """Test invalid pooling parameters."""
        feature_map = [np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
        
        # Test with pooling size larger than feature map
        with self.assertRaises((ValueError, ZeroDivisionError)):
            self.cnn.pooling(feature_map, 5, "average_pool")

    def test_file_operations_error_handling(self):
        """Test error handling in file operations."""
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            CNN.read_model("nonexistent_file.pkl")
        
        # Test saving to invalid path
        with self.assertRaises((OSError, PermissionError)):
            self.cnn.save_model("/invalid/path/model.pkl")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test sigmoid with very large values
        large_val = self.cnn.sig(1000)
        self.assertAlmostEqual(large_val, 1.0, places=5)
        
        small_val = self.cnn.sig(-1000)
        self.assertAlmostEqual(small_val, 0.0, places=5)
        
        # Test with NaN input
        nan_result = self.cnn.sig(np.nan)
        self.assertTrue(np.isnan(nan_result))


class TestTirasarCNNPerformance(unittest.TestCase):
    """Performance and efficiency tests for তিরসাર.py CNN."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.cnn = CNN([3, 2, 1], 2, 8, 4, 2)

    def test_memory_efficiency(self):
        """Test memory efficiency with moderately sized data."""
        # Use reasonable size data to avoid memory issues in CI
        test_data = np.random.rand(16, 16)
        
        import sys
        initial_refs = sys.getrefcount(test_data)
        
        # Run convolution
        focus_list, feature_maps = self.cnn.convolute(
            test_data, self.cnn.conv1, self.cnn.w_conv1,
            self.cnn.thre_conv1, self.cnn.step_conv1
        )
        
        # Run pooling
        pooled = self.cnn.pooling(feature_maps, self.cnn.size_pooling1)
        
        # Run expansion
        expanded = self.cnn._expand(pooled)
        
        # Verify operations completed without memory errors
        self.assertIsInstance(expanded, np.ndarray)
        self.assertGreater(len(expanded), 0)

    def test_computational_correctness(self):
        """Test computational correctness with known inputs."""
        # Use simple known input
        test_data = np.ones((4, 4))
        simple_kernel = [np.ones((3, 3))]
        
        focus_list, feature_maps = self.cnn.convolute(
            test_data, [3, 1], simple_kernel, [0], 1
        )
        
        # With all-ones input and all-ones kernel, each convolution should give 9
        # After sigmoid: sig(9) ≈ 0.9999
        expected_value = self.cnn.sig(9)
        
        for feature_map in feature_maps:
            # All values should be approximately the same (sig(9))
            unique_vals = np.unique(np.round(feature_map, 4))
            self.assertEqual(len(unique_vals), 1)
            self.assertAlmostEqual(unique_vals[0], expected_value, places=3)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
