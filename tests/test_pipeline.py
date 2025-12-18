"""
Test suite for StyleSense machine learning pipeline.

This module contains tests to verify the functionality of the StyleSense
pipeline including data loading, feature transformations, and model training.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "StyleSense" / "data" / "reviews.csv"


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_data_file_exists(self):
        """Test that the data file exists."""
        assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"
    
    def test_data_loads_successfully(self):
        """Test that the data loads without errors."""
        df = pd.read_csv(DATA_PATH)
        assert df is not None
        assert len(df) > 0
    
    def test_data_has_expected_columns(self):
        """Test that the data has all expected columns."""
        df = pd.read_csv(DATA_PATH)
        expected_columns = [
            'Clothing ID', 'Age', 'Title', 'Review Text',
            'Positive Feedback Count', 'Division Name',
            'Department Name', 'Class Name', 'Recommended IND'
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_data_has_no_missing_values(self):
        """Test that there are no missing values in the dataset."""
        df = pd.read_csv(DATA_PATH)
        assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
    
    def test_target_column_is_binary(self):
        """Test that the target column contains only 0 and 1."""
        df = pd.read_csv(DATA_PATH)
        unique_values = df['Recommended IND'].unique()
        assert set(unique_values) == {0, 1}, "Target should contain only 0 and 1"


class TestFeatureTransformations:
    """Tests for feature transformation functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Load sample data for testing."""
        df = pd.read_csv(DATA_PATH)
        # Use a small sample for faster testing
        return df.sample(n=min(100, len(df)), random_state=42)
    
    @pytest.fixture
    def prepared_features(self, sample_data):
        """Prepare features for testing."""
        data = sample_data.copy()
        # Combine text columns
        data['Full Review'] = data['Title'].fillna('') + ' ' + data['Review Text'].fillna('')
        return data
    
    def test_numerical_pipeline(self, prepared_features):
        """Test that numerical features are processed correctly."""
        num_features = ['Age', 'Positive Feedback Count']
        X_num = prepared_features[num_features]
        
        # Create numerical pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        
        # Transform
        X_transformed = num_pipeline.fit_transform(X_num)
        
        # Check output shape
        assert X_transformed.shape[0] == len(prepared_features)
        assert X_transformed.shape[1] == len(num_features)
        
        # Check that values are scaled between 0 and 1
        assert X_transformed.min() >= 0
        assert X_transformed.max() <= 1
    
    def test_categorical_pipeline(self, prepared_features):
        """Test that categorical features are processed correctly."""
        cat_features = ['Division Name', 'Department Name', 'Class Name']
        X_cat = prepared_features[cat_features]
        
        # Create categorical pipeline
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Transform
        X_transformed = cat_pipeline.fit_transform(X_cat)
        
        # Check output shape
        assert X_transformed.shape[0] == len(prepared_features)
        # One-hot encoding should create multiple columns
        assert X_transformed.shape[1] > len(cat_features)
    
    def test_text_vectorization(self, prepared_features):
        """Test that text features are vectorized correctly."""
        X_text = prepared_features['Full Review']
        
        # Create text vectorizer
        vectorizer = TfidfVectorizer(max_features=100, min_df=2)
        
        # Transform
        X_transformed = vectorizer.fit_transform(X_text)
        
        # Check output shape
        assert X_transformed.shape[0] == len(prepared_features)
        assert X_transformed.shape[1] > 0


class TestPipelineTraining:
    """Tests for model training functionality."""
    
    @pytest.fixture
    def train_test_data(self):
        """Prepare train/test split for testing."""
        df = pd.read_csv(DATA_PATH)
        # Use a small sample for faster testing
        df_sample = df.sample(n=min(500, len(df)), random_state=42)
        
        # Prepare features
        df_sample['Full Review'] = df_sample['Title'].fillna('') + ' ' + df_sample['Review Text'].fillna('')
        
        # Separate features and target
        X = df_sample.drop('Recommended IND', axis=1)
        y = df_sample['Recommended IND']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_pipeline_fits_without_error(self, train_test_data):
        """Test that the pipeline can be trained without errors."""
        X_train, _, y_train, _ = train_test_data
        
        # Define features
        num_features = ['Age', 'Positive Feedback Count']
        cat_features = ['Division Name', 'Department Name', 'Class Name']
        
        # Create simplified pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]), num_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_features),
            ]
        )
        
        # Create pipeline with model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # This should not raise any errors
        pipeline.fit(X_train, y_train)
        
        assert pipeline is not None
    
    def test_model_can_predict(self, train_test_data):
        """Test that the trained model can make predictions."""
        X_train, X_test, y_train, y_test = train_test_data
        
        # Define features
        num_features = ['Age', 'Positive Feedback Count']
        cat_features = ['Division Name', 'Department Name', 'Class Name']
        
        # Create simplified pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]), num_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_features),
            ]
        )
        
        # Create pipeline with model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        predictions = pipeline.predict(X_test)
        
        # Check predictions
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1}), "Predictions should be 0 or 1"
    
    def test_model_accuracy_above_baseline(self, train_test_data):
        """Test that model accuracy is above random baseline."""
        X_train, X_test, y_train, y_test = train_test_data
        
        # Define features
        num_features = ['Age', 'Positive Feedback Count']
        cat_features = ['Division Name', 'Department Name', 'Class Name']
        
        # Create simplified pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]), num_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_features),
            ]
        )
        
        # Create pipeline with model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Train and score
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        # Model should be better than random (50%)
        assert score > 0.5, f"Model accuracy ({score:.2f}) should be better than random baseline"
