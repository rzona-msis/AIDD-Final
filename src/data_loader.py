"""
Data Loader Module

This module handles loading data from various sources into pandas DataFrames.
Supports CSV, Excel, JSON, and other common formats.

Author: AIDD Final Project
Date: November 9, 2025
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class for loading data from various file formats.
    
    Attributes:
        data_dir (Path): Path to the data directory
        raw_data_dir (Path): Path to raw data subdirectory
        processed_data_dir (Path): Path to processed data subdirectory
    """
    
    def __init__(self, data_dir: str = '../data'):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Path to the main data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    def load_csv(self, 
                 filename: str, 
                 raw: bool = True,
                 **kwargs) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.
        
        Args:
            filename (str): Name of the CSV file
            raw (bool): If True, load from raw directory; else from processed
            **kwargs: Additional arguments to pass to pd.read_csv()
        
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.raw_data_dir / filename if raw else self.processed_data_dir / filename
        
        try:
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def load_excel(self, 
                   filename: str, 
                   raw: bool = True,
                   sheet_name: Union[str, int] = 0,
                   **kwargs) -> pd.DataFrame:
        """
        Load an Excel file into a pandas DataFrame.
        
        Args:
            filename (str): Name of the Excel file
            raw (bool): If True, load from raw directory; else from processed
            sheet_name (str or int): Sheet name or index to load
            **kwargs: Additional arguments to pass to pd.read_excel()
        
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.raw_data_dir / filename if raw else self.processed_data_dir / filename
        
        try:
            logger.info(f"Loading Excel file: {file_path}, sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise
    
    def load_json(self, 
                  filename: str, 
                  raw: bool = True,
                  **kwargs) -> pd.DataFrame:
        """
        Load a JSON file into a pandas DataFrame.
        
        Args:
            filename (str): Name of the JSON file
            raw (bool): If True, load from raw directory; else from processed
            **kwargs: Additional arguments to pass to pd.read_json()
        
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.raw_data_dir / filename if raw else self.processed_data_dir / filename
        
        try:
            logger.info(f"Loading JSON file: {file_path}")
            df = pd.read_json(file_path, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            raise
    
    def save_data(self, 
                  df: pd.DataFrame, 
                  filename: str,
                  processed: bool = True,
                  file_format: str = 'csv',
                  **kwargs) -> None:
        """
        Save a DataFrame to file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Name of the output file
            processed (bool): If True, save to processed directory; else to raw
            file_format (str): Format to save ('csv', 'excel', 'json', 'parquet')
            **kwargs: Additional arguments for the save function
        """
        save_dir = self.processed_data_dir if processed else self.raw_data_dir
        file_path = save_dir / filename
        
        try:
            logger.info(f"Saving data to: {file_path}")
            
            if file_format == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_format == 'excel':
                df.to_excel(file_path, index=False, **kwargs)
            elif file_format == 'json':
                df.to_json(file_path, **kwargs)
            elif file_format == 'parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully saved {len(df)} rows to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
        
        Returns:
            dict: Dictionary containing data information
        """
        info = {
            'shape': df.shape,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info(f"Data info - Rows: {info['rows']}, Columns: {info['columns']}")
        return info
    
    def list_files(self, raw: bool = True) -> list:
        """
        List all files in the data directory.
        
        Args:
            raw (bool): If True, list raw directory; else processed
        
        Returns:
            list: List of filenames
        """
        target_dir = self.raw_data_dir if raw else self.processed_data_dir
        
        try:
            files = [f.name for f in target_dir.iterdir() if f.is_file()]
            logger.info(f"Found {len(files)} files in {target_dir}")
            return files
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []


def load_sample_data(dataset_name: str = 'iris') -> pd.DataFrame:
    """
    Load a sample dataset for testing purposes.
    
    Args:
        dataset_name (str): Name of the sample dataset
            Options: 'iris', 'tips', 'titanic', 'diamonds'
    
    Returns:
        pd.DataFrame: Sample dataset
    """
    try:
        import seaborn as sns
        logger.info(f"Loading sample dataset: {dataset_name}")
        df = sns.load_dataset(dataset_name)
        logger.info(f"Loaded sample dataset with {len(df)} rows")
        return df
    except ImportError:
        logger.error("Seaborn not installed. Cannot load sample datasets.")
        raise
    except Exception as e:
        logger.error(f"Error loading sample dataset: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    # Initialize the data loader
    loader = DataLoader(data_dir='../data')
    
    # Example: Load a sample dataset
    print("\n=== Loading Sample Dataset ===")
    try:
        df = load_sample_data('iris')
        print(f"\nLoaded Iris dataset:")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Get data info
        info = loader.get_data_info(df)
        print(f"\nData Info:")
        print(f"Rows: {info['rows']}")
        print(f"Columns: {info['columns']}")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        
        # Save example
        loader.save_data(df, 'iris_sample.csv', processed=True)
        print("\nSaved sample data to processed directory")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # List files
    print("\n=== Listing Files ===")
    raw_files = loader.list_files(raw=True)
    processed_files = loader.list_files(raw=False)
    print(f"Raw files: {raw_files}")
    print(f"Processed files: {processed_files}")
