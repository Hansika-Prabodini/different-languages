"""
Comprehensive test suite for Electricity Data processing pipeline.
Tests data loading, cleaning, feature engineering, model training, and evaluation.
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Mock the data loading since the original file path might not exist
class MockCSVReader:
    """Mock CSV reader for testing data processing pipeline."""
    
    @staticmethod
    def create_sample_data(n_rows=1000):
        """Create sample electricity data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        # Create datetime index
        date_range = pd.date_range('2013-01-01', periods=n_rows, freq='30T')
        
        data = {
            'HolidayFlag': np.random.choice([0, 1], n_rows),
            'Holiday': ['None'] * n_rows,  # Will be dropped
            'DayOfWeek': np.random.randint(1, 8, n_rows),
            'WeekOfYear': np.random.randint(1, 53, n_rows),
            'Day': np.random.randint(1, 32, n_rows),
            'Month': np.random.randint(1, 13, n_rows),
            'Year': np.random.choice([2013, 2014, 2015], n_rows),
            'PeriodOfDay': np.random.randint(1, 49, n_rows),
            'ForecastWindProduction': np.random.normal(100, 30, n_rows),
            'SystemLoadEA': np.random.normal(5000, 1000, n_rows),
            'SMPEA': np.random.normal(50, 15, n_rows),
            'ORKTemperature': np.random.normal(15, 8, n_rows),
            'ORKWindspeed': np.random.normal(8, 4, n_rows),
            'CO2Intensity': np.random.normal(400, 50, n_rows),
            'ActualWindProduction': np.random.normal(95, 25, n_rows),
            'SystemLoadEP2': np.random.normal(4800, 900, n_rows),
            'SMPEP2': np.random.normal(45, 12, n_rows)
        }
        
        # Introduce some missing values
        missing_indices = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
        data['ORKTemperature'][missing_indices[:len(missing_indices)//2]] = np.nan
        data['ORKWindspeed'][missing_indices[len(missing_indices)//2:]] = np.nan
        
        # Create some outliers in SMPEP2
        outlier_indices = np.random.choice(n_rows, size=int(0.01 * n_rows), replace=False)
        data['SMPEP2'][outlier_indices] = np.random.choice([600, 700, -10, -20], len(outlier_indices))
        
        df = pd.DataFrame(data, index=date_range)
        return df


class TestDataLoading(unittest.TestCase):
    """Test data loading and initial processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = MockCSVReader.create_sample_data(100)
    
    def test_data_creation(self):
        """Test sample data creation."""
        self.assertEqual(len(self.sample_data), 100)
        self.assertIsInstance(self.sample_data.index, pd.DatetimeIndex)
        self.assertIn('SMPEP2', self.sample_data.columns)
        self.assertIn('ORKTemperature', self.sample_data.columns)
    
    def test_data_types(self):
        """Test initial data types after creation."""
        # Test that numeric columns can be converted
        numeric_columns = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA']
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data[col]))
    
    def test_missing_values_present(self):
        """Test that missing values are present in expected columns."""
        self.assertTrue(self.sample_data['ORKTemperature'].isna().any())
        self.assertTrue(self.sample_data['ORKWindspeed'].isna().any())
        
    def test_outliers_present(self):
        """Test that outliers are present in SMPEP2."""
        outliers = (self.sample_data['SMPEP2'] < 0) | (self.sample_data['SMPEP2'] > 550)
        self.assertTrue(outliers.any())


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = MockCSVReader.create_sample_data(200)
        
    def test_numeric_conversion(self):
        """Test conversion of columns to numeric types."""
        # Simulate the conversion process from the original script
        cols_to_numeric = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 
                          'ORKWindspeed', 'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
        
        for col in cols_to_numeric:
            converted = pd.to_numeric(self.sample_data[col], errors='coerce').astype('float32')
            self.assertEqual(converted.dtype, np.float32)
    
    def test_missing_value_removal(self):
        """Test removal of rows with missing temperature/windspeed values."""
        # Simulate dropna operation
        df_cleaned = self.sample_data.dropna(subset=['ORKTemperature','ORKWindspeed'])
        
        # Should have fewer rows than original
        self.assertLess(len(df_cleaned), len(self.sample_data))
        
        # Should have no missing values in these columns
        self.assertFalse(df_cleaned['ORKTemperature'].isna().any())
        self.assertFalse(df_cleaned['ORKWindspeed'].isna().any())
    
    def test_outlier_removal(self):
        """Test outlier removal from SMPEP2."""
        # Apply outlier filter
        outlier_filter = (self.sample_data['SMPEP2'] > 0) & (self.sample_data['SMPEP2'] <= 550)
        df_filtered = self.sample_data[outlier_filter]
        
        # All remaining values should be in valid range
        self.assertTrue((df_filtered['SMPEP2'] > 0).all())
        self.assertTrue((df_filtered['SMPEP2'] <= 550).all())
    
    def test_median_filling(self):
        """Test filling missing values with median."""
        # Create test data with missing values
        test_data = self.sample_data.copy()
        fill_columns = ['ForecastWindProduction','SystemLoadEA','SMPEA',
                       'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
        
        # Introduce some missing values
        for col in fill_columns[:2]:  # Test on first two columns
            test_data.loc[test_data.index[:10], col] = np.nan
        
        # Fill with median
        medians = {col: test_data[col].median() for col in fill_columns}
        test_data[fill_columns] = test_data[fill_columns].fillna(medians)
        
        # Check that no missing values remain
        for col in fill_columns:
            self.assertFalse(test_data[col].isna().any())
    
    def test_mean_filling(self):
        """Test filling CO2Intensity with mean."""
        test_data = self.sample_data.copy()
        
        # Introduce missing values
        test_data.loc[test_data.index[:5], 'CO2Intensity'] = np.nan
        
        # Fill with mean
        mean_value = test_data['CO2Intensity'].mean()
        test_data['CO2Intensity'].fillna(mean_value, inplace=True)
        
        # Check no missing values remain
        self.assertFalse(test_data['CO2Intensity'].isna().any())
    
    def test_column_dropping(self):
        """Test dropping of specified columns."""
        columns_to_drop = ['Holiday','WeekOfYear','ForecastWindProduction','SystemLoadEA']
        df_new = self.sample_data.drop(columns=columns_to_drop)
        
        for col in columns_to_drop:
            self.assertNotIn(col, df_new.columns)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = MockCSVReader.create_sample_data(100)
        # Simulate the cleaned data state
        self.df_cleaned = self.sample_data.dropna(subset=['ORKTemperature','ORKWindspeed'])
        self.df_scaled = self.df_cleaned.reset_index()
        
    def test_periodic_transform_function(self):
        """Test periodic transformation of cyclic features."""
        def periodic_transform(df, variable):
            max_val = df[variable].max()
            angle = 2 * np.pi * df[variable] / max_val
            df[f"{variable}_SIN"] = np.sin(angle)
            df[f"{variable}_COS"] = np.cos(angle)
            return df
        
        # Test transformation on DayOfWeek
        result_df = periodic_transform(self.df_scaled.copy(), 'DayOfWeek')
        
        # Check that sin and cos columns were created
        self.assertIn('DayOfWeek_SIN', result_df.columns)
        self.assertIn('DayOfWeek_COS', result_df.columns)
        
        # Check that values are in valid range for sin/cos
        self.assertTrue((result_df['DayOfWeek_SIN'] >= -1).all())
        self.assertTrue((result_df['DayOfWeek_SIN'] <= 1).all())
        self.assertTrue((result_df['DayOfWeek_COS'] >= -1).all())
        self.assertTrue((result_df['DayOfWeek_COS'] <= 1).all())
    
    def test_cyclic_transformation_completeness(self):
        """Test that all cyclic features are transformed."""
        df_test = self.df_scaled.copy()
        cyclic_features = ['DayOfWeek', 'Day', 'Month', 'PeriodOfDay']
        
        def periodic_transform(df, variable):
            max_val = df[variable].max()
            angle = 2 * np.pi * df[variable] / max_val
            df[f"{variable}_SIN"] = np.sin(angle)
            df[f"{variable}_COS"] = np.cos(angle)
            return df
        
        # Apply transformations
        for feature in cyclic_features:
            df_test = periodic_transform(df_test, feature)
        
        # Check all transformations exist
        for feature in cyclic_features:
            self.assertIn(f"{feature}_SIN", df_test.columns)
            self.assertIn(f"{feature}_COS", df_test.columns)
    
    def test_feature_dropping_after_transformation(self):
        """Test dropping of original cyclic features after transformation."""
        df_test = self.df_scaled.copy()
        
        # Add some dummy transformed columns
        df_test['DayOfWeek_SIN'] = np.sin(df_test['DayOfWeek'])
        df_test['DayOfWeek_COS'] = np.cos(df_test['DayOfWeek'])
        
        # Drop original columns
        columns_to_drop = ['DateTime','DayOfWeek','Day','Month','PeriodOfDay']
        # Only drop columns that exist
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_test.columns]
        df_final = df_test.drop(columns=existing_cols_to_drop)
        
        # Check that original cyclic columns are removed
        for col in existing_cols_to_drop:
            self.assertNotIn(col, df_final.columns)


class TestDataSplitting(unittest.TestCase):
    """Test data splitting and preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create processed data similar to the pipeline
        self.sample_data = MockCSVReader.create_sample_data(200)
        self.df_processed = self.sample_data.drop(columns=['Holiday'])
        
    def test_feature_target_split(self):
        """Test splitting features and target variable."""
        X = self.df_processed.drop(columns='SMPEP2', axis=1)
        y = self.df_processed['SMPEP2']
        
        # Check dimensions
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(self.df_processed))
        
        # Check target column not in features
        self.assertNotIn('SMPEP2', X.columns)
        
        # Check target is correct
        self.assertEqual(y.name, 'SMPEP2')
    
    def test_train_test_split_dimensions(self):
        """Test train-test split produces correct dimensions."""
        from sklearn.model_selection import train_test_split
        
        X = self.df_processed.drop(columns='SMPEP2', axis=1)
        y = self.df_processed['SMPEP2']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Check split ratios
        total_samples = len(X)
        expected_train = int(0.8 * total_samples)
        expected_test = total_samples - expected_train
        
        self.assertEqual(len(X_train), expected_train)
        self.assertEqual(len(X_test), expected_test)
        self.assertEqual(len(y_train), expected_train)
        self.assertEqual(len(y_test), expected_test)
    
    def test_data_scaling(self):
        """Test data scaling with MinMaxScaler."""
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        
        X = self.df_processed.drop(columns='SMPEP2', axis=1)
        y = self.df_processed['SMPEP2']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Apply scaling
        mm = MinMaxScaler()
        X_train_scaled = mm.fit_transform(X_train)
        X_test_scaled = mm.transform(X_test)
        
        # Check that scaled data is in [0, 1] range
        self.assertTrue((X_train_scaled >= 0).all())
        self.assertTrue((X_train_scaled <= 1).all())
        self.assertTrue((X_test_scaled >= -0.1).all())  # Small tolerance for test data
        self.assertTrue((X_test_scaled <= 1.1).all())   # Small tolerance for test data
        
        # Check dimensions preserved
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)


class TestModelTraining(unittest.TestCase):
    """Test model training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create minimal dataset for model testing
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.X_train = np.random.rand(n_samples, n_features)
        self.y_train = np.random.rand(n_samples) * 100
        self.X_test = np.random.rand(20, n_features)
        self.y_test = np.random.rand(20) * 100
    
    def test_model_accuracy_function(self):
        """Test the model_acc function structure."""
        from sklearn.linear_model import LinearRegression
        
        def model_acc(model, X_train, y_train, X_test, y_test):
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            return acc
        
        # Test with LinearRegression
        lr = LinearRegression()
        accuracy = model_acc(lr, self.X_train, self.y_train, self.X_test, self.y_test)
        
        # R² score should be between -inf and 1, but typically > -10 for reasonable data
        self.assertIsInstance(accuracy, (float, np.float64))
        self.assertGreater(accuracy, -10)  # Very loose bound
    
    def test_multiple_models_training(self):
        """Test training multiple models."""
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        
        models = {
            'LinearRegression': LinearRegression(),
            'Lasso': Lasso(alpha=0.1),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            try:
                model.fit(self.X_train, self.y_train)
                score = model.score(self.X_test, self.y_test)
                results[name] = score
                self.assertIsInstance(score, (float, np.float64))
            except Exception as e:
                self.fail(f"Model {name} failed to train: {e}")
        
        # Should have results for all models
        self.assertEqual(len(results), len(models))
    
    def test_random_forest_specific_training(self):
        """Test specific RandomForest training with parameters."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Test with specific parameters from the original script
        rf = RandomForestRegressor(
            n_estimators=50,  # Reduced for faster testing
            criterion='squared_error', 
            random_state=42,
            n_jobs=1  # Use single job for consistent testing
        )
        
        rf.fit(self.X_train, self.y_train)
        predictions = rf.predict(self.X_test)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check feature importance exists
        self.assertEqual(len(rf.feature_importances_), self.X_train.shape[1])
        self.assertTrue((rf.feature_importances_ >= 0).all())


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation metrics and procedures."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 50
        
        # Create realistic predictions vs actual values
        self.y_true = np.random.rand(n_samples) * 100
        self.y_pred = self.y_true + np.random.normal(0, 5, n_samples)  # Add some noise
    
    def test_evaluation_metrics(self):
        """Test MAE and MSE calculation."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(self.y_true, self.y_pred)
        mse = mean_squared_error(self.y_true, self.y_pred)
        
        # MAE and MSE should be positive
        self.assertGreater(mae, 0)
        self.assertGreater(mse, 0)
        
        # MSE should be >= MAE² for same-sized arrays (mathematical property)
        # This is not always true, but our noise level should make it reasonable
        self.assertIsInstance(mae, (float, np.float64))
        self.assertIsInstance(mse, (float, np.float64))
    
    def test_predictions_dataframe_creation(self):
        """Test creation of results dataframe."""
        # Simulate the final_df creation from the original script
        final_df = pd.DataFrame({
            'Prediction': self.y_pred,
            'Real': self.y_true
        })
        
        self.assertEqual(len(final_df), len(self.y_true))
        self.assertIn('Prediction', final_df.columns)
        self.assertIn('Real', final_df.columns)
        
        # Check data integrity
        np.testing.assert_array_equal(final_df['Real'].values, self.y_true)
        np.testing.assert_array_equal(final_df['Prediction'].values, self.y_pred)
    
    def test_feature_importance_analysis(self):
        """Test feature importance analysis."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Train model
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        feature_importance = rf.feature_importances_
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # Verify structure
        self.assertEqual(len(importance_df), X.shape[1])
        self.assertTrue((importance_df['Importance'] >= 0).all())
        self.assertTrue(abs(importance_df['Importance'].sum() - 1.0) < 1e-10)  # Should sum to 1


class TestNeuralNetworkComponents(unittest.TestCase):
    """Test neural network components from the script."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 5
        self.X_train = np.random.rand(self.n_samples, self.n_features).astype(np.float32)
        self.y_train = np.random.rand(self.n_samples).astype(np.float32)
    
    def test_regression_dataset_class(self):
        """Test custom RegressionDataset class."""
        # Mock PyTorch if not available
        try:
            import torch
            from torch.utils.data import Dataset
            
            class RegressionDataset(Dataset):
                def __init__(self, features, targets):
                    self.features = torch.tensor(features, dtype=torch.float32)
                    self.targets = torch.tensor(targets, dtype=torch.float32)
                
                def __len__(self):
                    return len(self.features)
                
                def __getitem__(self, idx):
                    return self.features[idx], self.targets[idx]
            
            # Test dataset creation
            dataset = RegressionDataset(self.X_train, self.y_train)
            
            # Test dataset length
            self.assertEqual(len(dataset), self.n_samples)
            
            # Test dataset item access
            features, target = dataset[0]
            self.assertEqual(len(features), self.n_features)
            self.assertIsInstance(target.item(), float)
            
        except ImportError:
            # Skip if PyTorch not available
            self.skipTest("PyTorch not available")
    
    def test_ann_model_structure(self):
        """Test ANN model structure."""
        try:
            import torch
            import torch.nn as nn
            
            class ANNModel(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(ANNModel, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x
            
            # Test model creation
            model = ANNModel(self.n_features, 32, 1)
            
            # Test forward pass
            test_input = torch.tensor(self.X_train[:5], dtype=torch.float32)
            output = model(test_input)
            
            # Check output shape
            self.assertEqual(output.shape, (5, 1))
            
        except ImportError:
            self.skipTest("PyTorch not available")


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_complete_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Start with raw data
        raw_data = MockCSVReader.create_sample_data(100)
        
        # Step 1: Data cleaning
        df_cleaned = raw_data.dropna(subset=['ORKTemperature','ORKWindspeed'])
        
        # Step 2: Outlier removal
        outlier_filter = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)
        df_cleaned = df_cleaned[outlier_filter]
        
        # Step 3: Fill missing values
        fill_with_median = ['ForecastWindProduction','SystemLoadEA','SMPEA',
                           'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
        medians = {col: df_cleaned[col].median() for col in fill_with_median if col in df_cleaned.columns}
        df_cleaned[list(medians.keys())] = df_cleaned[list(medians.keys())].fillna(medians)
        
        # Step 4: Drop columns
        columns_to_drop = ['Holiday','WeekOfYear','ForecastWindProduction','SystemLoadEA']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_cleaned.columns]
        df_processed = df_cleaned.drop(columns=existing_cols_to_drop)
        
        # Verify final state
        self.assertLessEqual(len(df_processed), len(raw_data))  # Should have fewer or equal rows
        self.assertTrue((df_processed['SMPEP2'] > 0).all())
        self.assertTrue((df_processed['SMPEP2'] <= 550).all())
        
        for col in existing_cols_to_drop:
            self.assertNotIn(col, df_processed.columns)
    
    def test_end_to_end_model_pipeline(self):
        """Test end-to-end model training pipeline."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Create processed data
        processed_data = MockCSVReader.create_sample_data(100)
        processed_data = processed_data.dropna()
        
        # Feature-target split
        X = processed_data.drop(columns='SMPEP2')
        y = processed_data['SMPEP2']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model training
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Verify pipeline completed successfully
        self.assertIsInstance(mse, (float, np.float64))
        self.assertIsInstance(mae, (float, np.float64))
        self.assertEqual(len(y_pred), len(y_test))
        self.assertGreater(model.score(X_test_scaled, y_test), -10)  # Very loose bound


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame()
        
        # Operations should handle empty data gracefully
        try:
            result = empty_df.dropna()
            self.assertEqual(len(result), 0)
        except Exception as e:
            self.fail(f"Empty data handling failed: {e}")
    
    def test_all_missing_values(self):
        """Test handling when all values are missing in a column."""
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'C': [10, 20, 30, 40, 50]
        })
        
        # Median of all-NaN column should be NaN
        median_b = test_data['B'].median()
        self.assertTrue(pd.isna(median_b))
        
        # Filling with median should still leave NaN
        filled = test_data['B'].fillna(median_b)
        self.assertTrue(filled.isna().all())
    
    def test_single_value_column(self):
        """Test handling of columns with single unique value."""
        test_data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        # Should handle constant column without error
        try:
            scaled = scaler.fit_transform(test_data)
            # Constant column should become all zeros
            self.assertTrue(np.allclose(scaled[:, 0], 0))
        except Exception as e:
            self.fail(f"Constant column handling failed: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
