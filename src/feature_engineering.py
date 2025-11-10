"""
Feature Engineering Module

This module provides functions for creating, transforming, and selecting features.
Includes encoding, scaling, and feature creation techniques.

Author: AIDD Final Project
Date: November 9, 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class for feature engineering operations.
    
    Provides methods for encoding categorical variables, scaling numerical features,
    creating new features, and selecting the most important features.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FeatureEngineer.
        
        Args:
            df (pd.DataFrame): The DataFrame to engineer features for
        """
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}
        logger.info(f"FeatureEngineer initialized with DataFrame of shape {self.df.shape}")
    
    def encode_categorical(self,
                          columns: List[str],
                          method: str = 'onehot',
                          drop_first: bool = True) -> 'FeatureEngineer':
        """
        Encode categorical variables.
        
        Args:
            columns (list): List of column names to encode
            method (str): Encoding method ('onehot', 'label', 'ordinal')
            drop_first (bool): Drop first category in one-hot encoding to avoid multicollinearity
        
        Returns:
            FeatureEngineer: Self for method chaining
        """
        logger.info(f"Encoding categorical columns: {columns} using {method} method")
        
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            if method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=drop_first)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(columns=[col])
                logger.info(f"One-hot encoded '{col}' into {len(dummies.columns)} columns")
            
            elif method == 'label':
                # Label encoding
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Label encoded '{col}'")
            
            elif method == 'ordinal':
                # Ordinal encoding (maintains order)
                unique_values = sorted(self.df[col].unique())
                ordinal_map = {val: idx for idx, val in enumerate(unique_values)}
                self.df[col] = self.df[col].map(ordinal_map)
                logger.info(f"Ordinal encoded '{col}'")
            
            else:
                raise ValueError(f"Unknown encoding method: {method}")
        
        return self
    
    def scale_features(self,
                      columns: List[str],
                      method: str = 'standard') -> 'FeatureEngineer':
        """
        Scale numerical features.
        
        Args:
            columns (list): List of column names to scale
            method (str): Scaling method ('standard', 'minmax', 'robust')
        
        Returns:
            FeatureEngineer: Self for method chaining
        """
        logger.info(f"Scaling columns: {columns} using {method} method")
        
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler
            logger.info(f"Scaled '{col}' using {method} scaling")
        
        return self
    
    def create_polynomial_features(self,
                                   columns: List[str],
                                   degree: int = 2) -> 'FeatureEngineer':
        """
        Create polynomial features from numerical columns.
        
        Args:
            columns (list): List of column names
            degree (int): Degree of polynomial features (2 = squared, 3 = cubed, etc.)
        
        Returns:
            FeatureEngineer: Self for method chaining
        """
        logger.info(f"Creating polynomial features (degree={degree}) for: {columns}")
        
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue
            
            for d in range(2, degree + 1):
                new_col_name = f"{col}_pow{d}"
                self.df[new_col_name] = self.df[col] ** d
                logger.info(f"Created feature '{new_col_name}'")
        
        return self
    
    def create_interaction_features(self,
                                   column_pairs: List[Tuple[str, str]],
                                   operation: str = 'multiply') -> 'FeatureEngineer':
        """
        Create interaction features between column pairs.
        
        Args:
            column_pairs (list): List of tuples containing column name pairs
            operation (str): Operation to perform ('multiply', 'add', 'divide', 'subtract')
        
        Returns:
            FeatureEngineer: Self for method chaining
        """
        logger.info(f"Creating interaction features using '{operation}' operation")
        
        for col1, col2 in column_pairs:
            if col1 not in self.df.columns or col2 not in self.df.columns:
                logger.warning(f"One or both columns not found: {col1}, {col2}")
                continue
            
            if not (pd.api.types.is_numeric_dtype(self.df[col1]) and 
                   pd.api.types.is_numeric_dtype(self.df[col2])):
                logger.warning(f"Both columns must be numeric: {col1}, {col2}")
                continue
            
            new_col_name = f"{col1}_{operation}_{col2}"
            
            if operation == 'multiply':
                self.df[new_col_name] = self.df[col1] * self.df[col2]
            elif operation == 'add':
                self.df[new_col_name] = self.df[col1] + self.df[col2]
            elif operation == 'divide':
                # Avoid division by zero
                self.df[new_col_name] = self.df[col1] / (self.df[col2] + 1e-10)
            elif operation == 'subtract':
                self.df[new_col_name] = self.df[col1] - self.df[col2]
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            logger.info(f"Created interaction feature '{new_col_name}'")
        
        return self
    
    def create_binned_features(self,
                              column: str,
                              bins: Union[int, List],
                              labels: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Create binned (discretized) features from continuous variables.
        
        Args:
            column (str): Column name to bin
            bins (int or list): Number of bins or bin edges
            labels (list): Labels for the bins (optional)
        
        Returns:
            FeatureEngineer: Self for method chaining
        """
        if column not in self.df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return self
        
        new_col_name = f"{column}_binned"
        self.df[new_col_name] = pd.cut(self.df[column], bins=bins, labels=labels)
        logger.info(f"Created binned feature '{new_col_name}'")
        
        return self
    
    def create_datetime_features(self,
                                column: str,
                                features: List[str] = ['year', 'month', 'day', 'dayofweek']) -> 'FeatureEngineer':
        """
        Extract datetime features from a datetime column.
        
        Args:
            column (str): Column name containing datetime data
            features (list): List of features to extract
                Options: 'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 
                        'dayofyear', 'quarter', 'is_weekend'
        
        Returns:
            FeatureEngineer: Self for method chaining
        """
        if column not in self.df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return self
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(self.df[column]):
            try:
                self.df[column] = pd.to_datetime(self.df[column])
            except Exception as e:
                logger.error(f"Could not convert '{column}' to datetime: {str(e)}")
                return self
        
        logger.info(f"Extracting datetime features from '{column}': {features}")
        
        for feature in features:
            new_col_name = f"{column}_{feature}"
            
            if feature == 'year':
                self.df[new_col_name] = self.df[column].dt.year
            elif feature == 'month':
                self.df[new_col_name] = self.df[column].dt.month
            elif feature == 'day':
                self.df[new_col_name] = self.df[column].dt.day
            elif feature == 'hour':
                self.df[new_col_name] = self.df[column].dt.hour
            elif feature == 'minute':
                self.df[new_col_name] = self.df[column].dt.minute
            elif feature == 'dayofweek':
                self.df[new_col_name] = self.df[column].dt.dayofweek
            elif feature == 'dayofyear':
                self.df[new_col_name] = self.df[column].dt.dayofyear
            elif feature == 'quarter':
                self.df[new_col_name] = self.df[column].dt.quarter
            elif feature == 'is_weekend':
                self.df[new_col_name] = (self.df[column].dt.dayofweek >= 5).astype(int)
            else:
                logger.warning(f"Unknown datetime feature: {feature}")
                continue
            
            logger.info(f"Created datetime feature '{new_col_name}'")
        
        return self
    
    def select_features(self,
                       target: str,
                       k: int = 10,
                       method: str = 'f_classif') -> List[str]:
        """
        Select the k best features based on statistical tests.
        
        Args:
            target (str): Target column name
            k (int): Number of features to select
            method (str): Selection method ('f_classif', 'chi2', 'mutual_info')
        
        Returns:
            list: List of selected feature names
        """
        if target not in self.df.columns:
            logger.error(f"Target column '{target}' not found")
            return []
        
        # Separate features and target
        X = self.df.drop(columns=[target])
        y = self.df[target]
        
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]
        
        logger.info(f"Selecting top {k} features using {method} method")
        
        # Choose scoring function
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'chi2':
            score_func = chi2
            # Chi2 requires non-negative features
            X_numeric = X_numeric.abs()
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Perform feature selection
        selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_cols)))
        selector.fit(X_numeric, y)
        
        # Get selected feature names
        selected_features = X_numeric.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected features: {selected_features}")
        return selected_features
    
    def get_feature_data(self) -> pd.DataFrame:
        """
        Get the DataFrame with engineered features.
        
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        logger.info(f"Returning engineered DataFrame with shape {self.df.shape}")
        return self.df


# Utility functions
def quick_feature_engineering(df: pd.DataFrame,
                              categorical_cols: Optional[List[str]] = None,
                              numerical_cols: Optional[List[str]] = None,
                              scale: bool = True,
                              encode: bool = True) -> pd.DataFrame:
    """
    Perform quick feature engineering with common transformations.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_cols (list): List of categorical columns to encode
        numerical_cols (list): List of numerical columns to scale
        scale (bool): Whether to scale numerical features
        encode (bool): Whether to encode categorical features
    
    Returns:
        pd.DataFrame: Transformed DataFrame
    """
    logger.info("Performing quick feature engineering")
    engineer = FeatureEngineer(df)
    
    if encode and categorical_cols:
        engineer.encode_categorical(categorical_cols, method='onehot')
    
    if scale and numerical_cols:
        engineer.scale_features(numerical_cols, method='standard')
    
    return engineer.get_feature_data()


# Example usage
if __name__ == "__main__":
    print("\n=== Feature Engineering Example ===")
    
    # Create sample data
    sample_data = {
        'age': [25, 30, 35, 40, 45, 50],
        'income': [50000, 60000, 75000, 80000, 90000, 100000],
        'education': ['HS', 'BS', 'MS', 'BS', 'PhD', 'MS'],
        'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'SF'],
        'target': [0, 0, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(sample_data)
    print("\nOriginal Data:")
    print(df)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(df)
    
    # Perform feature engineering
    print("\n=== Applying Feature Engineering ===")
    
    # Encode categorical variables
    engineer.encode_categorical(['education', 'city'], method='onehot')
    print("\nAfter encoding:")
    print(engineer.df.head())
    
    # Create polynomial features
    engineer.create_polynomial_features(['age'], degree=2)
    print("\nAfter polynomial features:")
    print(engineer.df[['age', 'age_pow2']].head())
    
    # Create interaction features
    engineer.create_interaction_features([('age', 'income')], operation='multiply')
    print("\nAfter interaction features:")
    print(engineer.df[['age', 'income', 'age_multiply_income']].head())
    
    # Scale numerical features
    engineer.scale_features(['age', 'income'], method='standard')
    print("\nAfter scaling:")
    print(engineer.df[['age', 'income']].head())
    
    # Get final data
    final_df = engineer.get_feature_data()
    print("\n=== Final Engineered DataFrame ===")
    print(f"Shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")
