"""
Configuration Module

This module contains project configuration settings.
Modify these values according to your specific needs.

Author: AIDD Final Project
Date: November 9, 2025
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Data Settings
DATA_FILE = 'your_data.csv'  # Change this to your data file
TARGET_COLUMN = 'target'  # Change this to your target variable name
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature Engineering Settings
CATEGORICAL_FEATURES = []  # List your categorical features
NUMERICAL_FEATURES = []  # List your numerical features
FEATURES_TO_SCALE = []  # Features that need scaling
FEATURES_TO_ENCODE = []  # Features that need encoding

# Model Training Settings
TASK_TYPE = 'classification'  # or 'regression'
CROSS_VALIDATION_FOLDS = 5
MODELS_TO_TRAIN = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting'
]

# Model Hyperparameters
HYPERPARAMETERS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7]
    }
}

# Evaluation Settings
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
REGRESSION_METRICS = ['mse', 'rmse', 'mae', 'r2', 'mape']

# Visualization Settings
FIGURE_SIZE = (12, 6)
PLOT_STYLE = 'whitegrid'
DPI = 300
COLOR_PALETTE = 'Set2'

# Logging Settings
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = PROJECT_ROOT / 'project.log'

# Processing Settings
N_JOBS = -1  # Number of CPU cores to use (-1 = all available)
VERBOSE = True

# Data Cleaning Settings
MISSING_VALUE_STRATEGY = 'median'  # 'drop', 'mean', 'median', 'mode'
OUTLIER_METHOD = 'iqr'  # 'iqr' or 'zscore'
OUTLIER_THRESHOLD = 1.5
REMOVE_DUPLICATES = True

# Feature Selection Settings
FEATURE_SELECTION_METHOD = 'f_classif'  # 'f_classif', 'chi2', 'mutual_info'
N_FEATURES_TO_SELECT = 10

# Model Persistence Settings
SAVE_BEST_MODEL = True
BEST_MODEL_PATH = MODELS_DIR / 'best_model.pkl'

# Reproducibility
import numpy as np
import random

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Example usage
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nTask Type: {TASK_TYPE}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Random State: {RANDOM_STATE}")
