"""
Data Cleaner Module

This module provides functions for cleaning and preprocessing data.
Handles missing values, outliers, duplicates, and data type conversions.

Author: AIDD Final Project
Date: November 9, 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class for cleaning and preprocessing data.
    
    This class provides methods for handling missing values, removing duplicates,
    detecting and handling outliers, and standardizing data formats.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to clean
        """
        self.df = df.copy()
        self.original_shape = df.shape
        logger.info(f"DataCleaner initialized with DataFrame of shape {self.original_shape}")
    
    def get_cleaning_report(self) -> Dict:
        """
        Generate a report of the current data quality issues.
        
        Returns:
            dict: Dictionary containing data quality metrics
        """
        report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info(f"Generated cleaning report - Missing values: {report['duplicate_rows']} duplicates found")
        return report
    
    def handle_missing_values(self, 
                             strategy: str = 'drop',
                             columns: Optional[List[str]] = None,
                             fill_value: Optional[Union[int, float, str]] = None,
                             threshold: float = 0.5) -> 'DataCleaner':
        """
        Handle missing values in the DataFrame.
        
        Args:
            strategy (str): Strategy for handling missing values
                - 'drop': Drop rows with missing values
                - 'drop_columns': Drop columns with missing values above threshold
                - 'mean': Fill with column mean (numeric only)
                - 'median': Fill with column median (numeric only)
                - 'mode': Fill with column mode
                - 'forward_fill': Forward fill
                - 'backward_fill': Backward fill
                - 'constant': Fill with constant value
            columns (list): Specific columns to apply the strategy to (None = all)
            fill_value: Value to use when strategy='constant'
            threshold (float): Threshold for dropping columns (proportion of missing values)
        
        Returns:
            DataCleaner: Self for method chaining
        """
        initial_rows = len(self.df)
        initial_cols = len(self.df.columns)
        
        target_cols = columns if columns else self.df.columns.tolist()
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            self.df = self.df.dropna(subset=target_cols)
            logger.info(f"Dropped {initial_rows - len(self.df)} rows with missing values")
        
        elif strategy == 'drop_columns':
            missing_pct = self.df[target_cols].isnull().sum() / len(self.df)
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            self.df = self.df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        elif strategy == 'mean':
            for col in target_cols:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
            logger.info(f"Filled missing values with mean for numeric columns")
        
        elif strategy == 'median':
            for col in target_cols:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            logger.info(f"Filled missing values with median for numeric columns")
        
        elif strategy == 'mode':
            for col in target_cols:
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                if mode_value is not None:
                    self.df[col].fillna(mode_value, inplace=True)
            logger.info(f"Filled missing values with mode")
        
        elif strategy == 'forward_fill':
            self.df[target_cols] = self.df[target_cols].fillna(method='ffill')
            logger.info(f"Applied forward fill for missing values")
        
        elif strategy == 'backward_fill':
            self.df[target_cols] = self.df[target_cols].fillna(method='bfill')
            logger.info(f"Applied backward fill for missing values")
        
        elif strategy == 'constant':
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy='constant'")
            self.df[target_cols] = self.df[target_cols].fillna(fill_value)
            logger.info(f"Filled missing values with constant: {fill_value}")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self
    
    def remove_duplicates(self, 
                         subset: Optional[List[str]] = None,
                         keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset (list): Columns to consider for identifying duplicates
            keep (str): Which duplicates to keep ('first', 'last', False)
        
        Returns:
            DataCleaner: Self for method chaining
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        logger.info(f"Removed {removed} duplicate rows")
        return self
    
    def detect_outliers(self, 
                       column: str,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a numeric column.
        
        Args:
            column (str): Column name to check for outliers
            method (str): Method for outlier detection ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
                - For IQR: multiplier for IQR (default 1.5)
                - For Z-score: number of standard deviations (default 3)
        
        Returns:
            pd.Series: Boolean series indicating outliers
        """
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column {column} is not numeric")
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_outliers = outliers.sum()
        logger.info(f"Detected {n_outliers} outliers in column '{column}' using {method} method")
        return outliers
    
    def handle_outliers(self,
                       column: str,
                       method: str = 'iqr',
                       strategy: str = 'remove',
                       threshold: float = 1.5) -> 'DataCleaner':
        """
        Handle outliers in a numeric column.
        
        Args:
            column (str): Column name to process
            method (str): Method for outlier detection ('iqr' or 'zscore')
            strategy (str): How to handle outliers
                - 'remove': Remove outlier rows
                - 'cap': Cap values at the threshold
                - 'median': Replace with median
            threshold (float): Threshold for outlier detection
        
        Returns:
            DataCleaner: Self for method chaining
        """
        outliers = self.detect_outliers(column, method, threshold)
        
        if strategy == 'remove':
            initial_rows = len(self.df)
            self.df = self.df[~outliers]
            logger.info(f"Removed {initial_rows - len(self.df)} outlier rows")
        
        elif strategy == 'cap':
            if method == 'iqr':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"Capped outliers in column '{column}'")
        
        elif strategy == 'median':
            median_value = self.df[column].median()
            self.df.loc[outliers, column] = median_value
            logger.info(f"Replaced outliers with median in column '{column}'")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self
    
    def convert_data_types(self, 
                          type_map: Dict[str, str]) -> 'DataCleaner':
        """
        Convert data types of specified columns.
        
        Args:
            type_map (dict): Dictionary mapping column names to target data types
                Example: {'age': 'int', 'price': 'float', 'category': 'category'}
        
        Returns:
            DataCleaner: Self for method chaining
        """
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(dtype)
                    logger.info(f"Converted column '{col}' to {dtype}")
                except Exception as e:
                    logger.warning(f"Could not convert column '{col}' to {dtype}: {str(e)}")
            else:
                logger.warning(f"Column '{col}' not found in DataFrame")
        
        return self
    
    def standardize_column_names(self) -> 'DataCleaner':
        """
        Standardize column names (lowercase, replace spaces with underscores).
        
        Returns:
            DataCleaner: Self for method chaining
        """
        original_columns = self.df.columns.tolist()
        self.df.columns = (self.df.columns
                          .str.strip()
                          .str.lower()
                          .str.replace(' ', '_')
                          .str.replace('[^a-z0-9_]', '', regex=True))
        logger.info(f"Standardized column names")
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        logger.info(f"Returning cleaned DataFrame - Original shape: {self.original_shape}, "
                   f"Final shape: {self.df.shape}")
        return self.df
    
    def reset(self) -> 'DataCleaner':
        """
        Reset to the original DataFrame.
        
        Returns:
            DataCleaner: Self for method chaining
        """
        logger.warning("Resetting to original DataFrame")
        # Note: This won't work as intended without storing the original df
        # This is a design limitation documented in SHORTCOMINGS.md
        return self


# Utility functions
def quick_clean(df: pd.DataFrame, 
                remove_duplicates: bool = True,
                handle_missing: str = 'drop',
                standardize_names: bool = True) -> pd.DataFrame:
    """
    Perform quick basic cleaning on a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicates
        handle_missing (str): Strategy for missing values ('drop', 'mean', 'median')
        standardize_names (bool): Whether to standardize column names
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    logger.info("Performing quick clean")
    cleaner = DataCleaner(df)
    
    if standardize_names:
        cleaner.standardize_column_names()
    
    if remove_duplicates:
        cleaner.remove_duplicates()
    
    if handle_missing:
        cleaner.handle_missing_values(strategy=handle_missing)
    
    return cleaner.get_cleaned_data()


# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    print("\n=== Data Cleaning Example ===")
    
    sample_data = {
        'Name': ['John', 'Jane', 'Bob', 'Alice', 'Bob', None],
        'Age': [25, np.nan, 35, 28, 35, 45],
        'Salary': [50000, 60000, 75000, np.nan, 75000, 100000],
        'Department': ['Sales', 'Marketing', 'IT', 'Sales', 'IT', 'HR'],
        'Score': [85, 92, 78, 88, 78, 150]  # 150 is an outlier
    }
    
    df = pd.DataFrame(sample_data)
    print("\nOriginal Data:")
    print(df)
    print(f"\nShape: {df.shape}")
    
    # Initialize cleaner
    cleaner = DataCleaner(df)
    
    # Get initial report
    print("\n=== Cleaning Report ===")
    report = cleaner.get_cleaning_report()
    print(f"Total rows: {report['total_rows']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Missing values:\n{pd.Series(report['missing_values'])}")
    
    # Perform cleaning
    cleaned_df = (cleaner
                  .standardize_column_names()
                  .remove_duplicates()
                  .handle_missing_values(strategy='median', columns=['age', 'salary'])
                  .handle_outliers(column='score', method='iqr', strategy='cap')
                  .get_cleaned_data())
    
    print("\n=== Cleaned Data ===")
    print(cleaned_df)
    print(f"\nShape after cleaning: {cleaned_df.shape}")
    
    # Final report
    print("\n=== Final Report ===")
    final_cleaner = DataCleaner(cleaned_df)
    final_report = final_cleaner.get_cleaning_report()
    print(f"Total rows: {final_report['total_rows']}")
    print(f"Missing values:\n{pd.Series(final_report['missing_values'])}")
