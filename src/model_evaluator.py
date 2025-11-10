"""
Model Evaluator Module

This module provides functions for evaluating machine learning models.
Supports various metrics for classification and regression tasks.

Author: AIDD Final Project
Date: November 9, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    A class for evaluating machine learning model performance.
    
    Provides metrics and visualization tools for both classification
    and regression tasks.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the ModelEvaluator.
        
        Args:
            task_type (str): Type of ML task ('classification' or 'regression')
        """
        self.task_type = task_type
        self.results = {}
        logger.info(f"ModelEvaluator initialized for {task_type} task")
    
    def evaluate_classification(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               model_name: str = 'Model') -> Dict[str, float]:
        """
        Evaluate a classification model.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities (optional)
            model_name (str): Name of the model
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating classification model: {model_name}")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if probabilities are provided
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        self.results[model_name] = metrics
        
        logger.info(f"Results for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_regression(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           model_name: str = 'Model') -> Dict[str, float]:
        """
        Evaluate a regression model.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating regression model: {model_name}")
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Add MAPE if no zero values in y_true
        if not np.any(y_true == 0):
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not calculate MAPE: {str(e)}")
        
        self.results[model_name] = metrics
        
        logger.info(f"Results for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            np.ndarray: Confusion matrix
        """
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification")
        
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Generated confusion matrix of shape {cm.shape}")
        return cm
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (8, 6),
                             save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (list): Class labels
            figsize (tuple): Figure size
            save_path (str): Path to save the plot (optional)
        """
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification")
        
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      figsize: Tuple[int, int] = (8, 6),
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve (binary classification only).
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            figsize (tuple): Figure size
            save_path (str): Path to save the plot (optional)
        """
        if self.task_type != 'classification':
            raise ValueError("ROC curve only available for classification")
        
        if len(np.unique(y_true)) != 2:
            raise ValueError("ROC curve plotting only supports binary classification")
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      figsize: Tuple[int, int] = (10, 4),
                      save_path: Optional[str] = None) -> None:
        """
        Plot residuals for regression models.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            figsize (tuple): Figure size
            save_path (str): Path to save the plot (optional)
        """
        if self.task_type != 'regression':
            raise ValueError("Residual plot only available for regression")
        
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residual plot
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residual Plot')
        ax1.grid(alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_scatter(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               figsize: Tuple[int, int] = (8, 6),
                               save_path: Optional[str] = None) -> None:
        """
        Plot predicted vs actual values for regression.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            figsize (tuple): Figure size
            save_path (str): Path to save the plot (optional)
        """
        if self.task_type != 'regression':
            raise ValueError("Prediction scatter plot only available for regression")
        
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction scatter plot saved to {save_path}")
        
        plt.show()
    
    def get_classification_report(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 labels: Optional[List[str]] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (list): Class labels
        
        Returns:
            str: Classification report
        """
        if self.task_type != 'classification':
            raise ValueError("Classification report only available for classification")
        
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        logger.info("Generated classification report")
        return report
    
    def compare_models(self,
                      metric: Optional[str] = None,
                      figsize: Tuple[int, int] = (10, 6),
                      save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            metric (str): Specific metric to compare (if None, compares all)
            figsize (tuple): Figure size for the plot
            save_path (str): Path to save the plot (optional)
        
        Returns:
            pd.DataFrame: Comparison DataFrame
        """
        if not self.results:
            raise ValueError("No results to compare. Evaluate models first.")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        
        logger.info(f"Comparing {len(self.results)} models")
        
        # Plot comparison
        if metric:
            if metric not in comparison_df.columns:
                raise ValueError(f"Metric '{metric}' not found in results")
            
            plt.figure(figsize=figsize)
            comparison_df[metric].sort_values().plot(kind='barh', color='steelblue')
            plt.xlabel(metric.upper())
            plt.ylabel('Model')
            plt.title(f'Model Comparison - {metric.upper()}')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
        else:
            # Plot all metrics
            comparison_df.plot(kind='bar', figsize=figsize, width=0.8)
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.title('Model Comparison - All Metrics')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        return comparison_df
    
    def get_best_model(self,
                      metric: str) -> Tuple[str, float]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric (str): Metric to use for comparison
        
        Returns:
            tuple: (model_name, metric_value)
        """
        if not self.results:
            raise ValueError("No results available. Evaluate models first.")
        
        comparison_df = pd.DataFrame(self.results).T
        
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        # For MSE, MAE, RMSE - lower is better
        if metric in ['mse', 'mae', 'rmse', 'mape']:
            best_idx = comparison_df[metric].idxmin()
        else:  # For accuracy, precision, recall, f1, r2, roc_auc - higher is better
            best_idx = comparison_df[metric].idxmax()
        
        best_model = best_idx
        best_score = comparison_df.loc[best_idx, metric]
        
        logger.info(f"Best model based on {metric}: {best_model} ({best_score:.4f})")
        return best_model, best_score


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    print("\n=== Model Evaluation Example (Classification) ===")
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train two models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    gb_pred = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)
    
    # Evaluate models
    evaluator = ModelEvaluator(task_type='classification')
    
    print("\n=== Random Forest Results ===")
    rf_metrics = evaluator.evaluate_classification(y_test, rf_pred, rf_proba, 'Random Forest')
    
    print("\n=== Gradient Boosting Results ===")
    gb_metrics = evaluator.evaluate_classification(y_test, gb_pred, gb_proba, 'Gradient Boosting')
    
    # Compare models
    print("\n=== Model Comparison ===")
    comparison = evaluator.compare_models(metric='accuracy')
    print(comparison)
    
    # Get best model
    best_model, best_score = evaluator.get_best_model('f1_score')
    print(f"\nBest model based on F1-score: {best_model} ({best_score:.4f})")
    
    # Plot confusion matrix
    print("\n=== Plotting Confusion Matrix ===")
    evaluator.plot_confusion_matrix(y_test, rf_pred, labels=['Class 0', 'Class 1'])
