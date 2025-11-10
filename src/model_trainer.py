"""
Model Trainer Module

This module provides functions for training machine learning models.
Supports classification and regression tasks with multiple algorithms.

Author: AIDD Final Project
Date: November 9, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import joblib
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class for training and managing machine learning models.
    
    Supports various classification and regression algorithms with
    hyperparameter tuning and cross-validation capabilities.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the ModelTrainer.
        
        Args:
            task_type (str): Type of ML task ('classification' or 'regression')
        """
        self.task_type = task_type
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info(f"ModelTrainer initialized for {task_type} task")
    
    def prepare_data(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    stratify: bool = True) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify the split (for classification)
        """
        stratify_param = y if stratify and self.task_type == 'classification' else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        logger.info(f"Data split - Training: {len(self.X_train)}, Testing: {len(self.X_test)}")
    
    def get_default_models(self) -> Dict[str, Any]:
        """
        Get a dictionary of default models based on task type.
        
        Returns:
            dict: Dictionary of model name to model object
        """
        if self.task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }
        else:  # regression
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor()
            }
        
        logger.info(f"Retrieved {len(models)} default models for {self.task_type}")
        return models
    
    def train_model(self,
                   model_name: str,
                   model: Any = None,
                   **kwargs) -> Any:
        """
        Train a single model.
        
        Args:
            model_name (str): Name of the model
            model: Model object (if None, uses default model)
            **kwargs: Additional parameters for the model
        
        Returns:
            Trained model object
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if model is None:
            default_models = self.get_default_models()
            if model_name not in default_models:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(default_models.keys())}")
            model = default_models[model_name]
        
        # Update model parameters if provided
        if kwargs:
            model.set_params(**kwargs)
        
        logger.info(f"Training {model_name}...")
        model.fit(self.X_train, self.y_train)
        
        self.models[model_name] = model
        logger.info(f"Successfully trained {model_name}")
        
        return model
    
    def train_multiple_models(self,
                            model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Args:
            model_names (list): List of model names to train (if None, trains all default models)
        
        Returns:
            dict: Dictionary of trained models
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        default_models = self.get_default_models()
        
        if model_names is None:
            model_names = list(default_models.keys())
        
        logger.info(f"Training {len(model_names)} models...")
        
        for model_name in model_names:
            try:
                self.train_model(model_name)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        logger.info(f"Completed training {len(self.models)} models")
        return self.models
    
    def cross_validate(self,
                      model_name: str,
                      cv: int = 5,
                      scoring: Optional[str] = None) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name (str): Name of the model to cross-validate
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric (if None, uses default for task type)
        
        Returns:
            dict: Dictionary with mean and std of cross-validation scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        logger.info(f"Performing {cv}-fold cross-validation for {model_name}...")
        
        model = self.models[model_name]
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        logger.info(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        return results
    
    def hyperparameter_tuning(self,
                             model_name: str,
                             param_grid: Dict,
                             cv: int = 5,
                             scoring: Optional[str] = None) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name (str): Name of the model
            param_grid (dict): Dictionary of parameters to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
        
        Returns:
            tuple: (best_model, best_params)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        default_models = self.get_default_models()
        if model_name not in default_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        base_model = default_models[model_name]
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        self.models[f"{model_name} (Tuned)"] = best_model
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        
        return best_model, best_params
    
    def set_best_model(self, model_name: str) -> None:
        """
        Set a model as the best model.
        
        Args:
            model_name (str): Name of the model to set as best
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self.best_model = self.models[model_name]
        self.best_model_name = model_name
        logger.info(f"Set '{model_name}' as the best model")
    
    def predict(self,
               X: pd.DataFrame,
               model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction
            model_name (str): Name of model to use (if None, uses best model)
        
        Returns:
            np.ndarray: Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model set. Specify model_name or call set_best_model()")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            model = self.models[model_name]
        
        logger.info(f"Making predictions...")
        predictions = model.predict(X)
        return predictions
    
    def predict_proba(self,
                     X: pd.DataFrame,
                     model_name: Optional[str] = None) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X (pd.DataFrame): Feature matrix
            model_name (str): Name of model to use
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model set")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            model = self.models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model does not support predict_proba")
        
        logger.info(f"Getting prediction probabilities...")
        probabilities = model.predict_proba(X)
        return probabilities
    
    def save_model(self,
                  model_name: str,
                  filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model '{model_name}' to {filepath}")
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model saved successfully")
    
    def load_model(self,
                  model_name: str,
                  filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to load the model from
        """
        logger.info(f"Loading model from {filepath}")
        model = joblib.load(filepath)
        self.models[model_name] = model
        logger.info(f"Model loaded successfully as '{model_name}'")
    
    def get_feature_importance(self,
                              model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance from a tree-based model.
        
        Args:
            model_name (str): Name of the model
        
        Returns:
            pd.DataFrame: DataFrame with feature names and importance scores
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model set")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model does not have feature_importances_ attribute")
        
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Retrieved feature importance for {len(importance_df)} features")
        return importance_df


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    
    print("\n=== Model Training Example (Classification) ===")
    
    # Create sample classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Initialize trainer
    trainer = ModelTrainer(task_type='classification')
    
    # Prepare data
    trainer.prepare_data(X_df, y_series, test_size=0.2, random_state=42)
    
    # Train multiple models
    print("\n=== Training Multiple Models ===")
    models = trainer.train_multiple_models(model_names=['Logistic Regression', 'Random Forest', 'Decision Tree'])
    
    # Cross-validate
    print("\n=== Cross-Validation ===")
    for model_name in models.keys():
        cv_results = trainer.cross_validate(model_name, cv=5)
        print(f"{model_name}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    
    # Set best model
    trainer.set_best_model('Random Forest')
    
    # Make predictions
    print("\n=== Making Predictions ===")
    predictions = trainer.predict(trainer.X_test)
    print(f"Made {len(predictions)} predictions")
    
    # Get feature importance
    print("\n=== Feature Importance ===")
    importance_df = trainer.get_feature_importance()
    print(importance_df.head(10))
    
    # Save model
    print("\n=== Saving Model ===")
    trainer.save_model('Random Forest', '../models/random_forest_model.pkl')
    print("Model saved successfully")
