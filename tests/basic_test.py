import unittest
import pandas as pd
from data_processing import preprocess_data
from train_model import train_model
from evaluate_model import evaluate_model

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing"""
        self.test_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9],
            'sepal_width': [3.5, 3.0],
            'petal_length': [1.4, 1.4],
            'petal_width': [0.2, 0.2],
            'species': ['setosa', 'setosa']
        })
        
    def test_preprocess_data(self):
        """Test if preprocessing returns correct format"""
        # Save test data
        self.test_data.to_csv('/tmp/test_input.csv', index=False)
        
        # Run preprocessing
        processed_file = preprocess_data('/tmp/test_input.csv', 'processed.csv')
        
        # Load and check processed data
        processed_df = pd.read_csv(processed_file)
        
        # Simple checks
        self.assertEqual(len(processed_df.columns), 5)  # Check number of columns
        self.assertTrue('species' in processed_df.columns)  # Check target column exists
        
    def test_train_model(self):
        """Test if model training works"""
        # Save test data
        self.test_data.to_csv('/tmp/test_train.csv', index=False)
        
        # Train model
        model_file = train_model('/tmp/test_train.csv', 'test_model.pkl')
        
        # Check if model file was created
        self.assertTrue(model_file.endswith('test_model.pkl'))
        
    def test_evaluate_model(self):
        """Test if model evaluation returns valid accuracy"""
        # Prepare test data
        self.test_data.to_csv('/tmp/test_eval.csv', index=False)
        
        # Train and save a test model
        model_file = train_model('/tmp/test_eval.csv', 'eval_model.pkl')
        
        # Evaluate model
        accuracy = evaluate_model(model_file, '/tmp/test_eval.csv')
        
        # Check if accuracy is valid
        self.assertTrue(0 <= accuracy <= 1)