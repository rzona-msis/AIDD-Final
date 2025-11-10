# AIDD Final Project

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Educational-green.svg)](LICENSE)

A comprehensive machine learning project for the AI in Data-Driven Decision Making (AIDD) course at Indiana University's MSIS program. This project demonstrates end-to-end implementation of AI/ML techniques including data preprocessing, exploratory analysis, model development, evaluation, and deployment considerations.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project applies machine learning techniques to solve real-world business problems through data-driven decision making. The implementation showcases:

- **Data Processing**: Comprehensive data cleaning, preprocessing, and feature engineering
- **Exploratory Analysis**: Statistical analysis and visualization of patterns and relationships
- **Model Development**: Implementation of multiple ML algorithms with comparison
- **Model Evaluation**: Rigorous testing using appropriate metrics and cross-validation
- **Best Practices**: Clean code, documentation, version control, and reproducibility

### Problem Statement

Organizations face increasing complexity in decision-making processes due to large data volumes, need for timely predictions, and limited capacity to analyze complex patterns. This project demonstrates how AI and machine learning can develop accurate predictive models that enable better business decisions.

### Objectives

1. Perform comprehensive data collection, cleaning, and preprocessing
2. Conduct exploratory data analysis to identify patterns and insights
3. Develop and compare multiple machine learning models
4. Evaluate and optimize model performance
5. Provide actionable insights and recommendations

## ✨ Features

- **Modular Code Structure**: Well-organized, reusable Python modules
- **Data Processing Pipeline**: Automated data loading, cleaning, and transformation
- **Feature Engineering**: Advanced feature creation and selection techniques
- **Multiple ML Models**: Implementation of various algorithms (Logistic Regression, Random Forest, Gradient Boosting, etc.)
- **Comprehensive Evaluation**: Multiple metrics, confusion matrices, ROC curves
- **Jupyter Notebooks**: Interactive analysis and visualization
- **Documentation**: Detailed documentation of methodology and results

## 📁 Project Structure

```
AIDD-Final/
│
├── data/                          # Data directory
│   ├── raw/                       # Original, immutable data
│   └── processed/                 # Cleaned, processed data
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb   # (To be created)
│   └── 03_model_evaluation.ipynb # (To be created)
│
├── src/                           # Source code
│   ├── data_loader.py            # Data loading utilities
│   ├── data_cleaner.py           # Data cleaning functions
│   ├── feature_engineering.py    # Feature engineering tools
│   ├── model_trainer.py          # Model training pipeline
│   └── model_evaluator.py        # Model evaluation tools
│
├── models/                        # Trained model files
│   └── .gitkeep
│
├── results/                       # Results, figures, tables
│   └── .gitkeep
│
├── docs/                          # Documentation
│   ├── PROJECT_PROPOSAL.md       # Detailed project proposal
│   ├── METHODOLOGY.md            # (To be created)
│   └── SHORTCOMINGS.md           # Limitations and improvements
│
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/rzona-msis/AIDD-Final.git
cd AIDD-Final
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas; import sklearn; import numpy; print('Installation successful!')"
```

## 💻 Usage

### Using Python Scripts

#### 1. Load and Clean Data

```python
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner

# Load data
loader = DataLoader(data_dir='data')
df = loader.load_csv('your_data.csv')

# Clean data
cleaner = DataCleaner(df)
cleaned_df = (cleaner
              .standardize_column_names()
              .remove_duplicates()
              .handle_missing_values(strategy='median')
              .get_cleaned_data())
```

#### 2. Engineer Features

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(cleaned_df)
processed_df = (engineer
                .encode_categorical(['category_col'], method='onehot')
                .scale_features(['numeric_col'], method='standard')
                .get_feature_data())
```

#### 3. Train Models

```python
from src.model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(task_type='classification')

# Prepare data
trainer.prepare_data(X, y, test_size=0.2)

# Train multiple models
trainer.train_multiple_models()

# Cross-validate
for model_name in trainer.models.keys():
    trainer.cross_validate(model_name, cv=5)
```

#### 4. Evaluate Models

```python
from src.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(task_type='classification')

# Evaluate model
metrics = evaluator.evaluate_classification(y_test, y_pred, y_pred_proba)

# Compare models
comparison = evaluator.compare_models()

# Get best model
best_model, best_score = evaluator.get_best_model('f1_score')
```

### Using Jupyter Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `notebooks/` directory

3. Open and run notebooks in order:
   - `01_exploratory_data_analysis.ipynb` - EDA and data understanding
   - `02_model_training.ipynb` - Model development
   - `03_model_evaluation.ipynb` - Results and evaluation

## 🔬 Methodology

### 1. Data Collection & Preparation
- Data sourcing and loading
- Data quality assessment
- Handling missing values and outliers
- Data type conversions

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Feature relationships
- Target variable analysis

### 3. Feature Engineering
- Categorical encoding (One-Hot, Label)
- Numerical scaling (Standard, MinMax)
- Feature creation (Polynomial, Interactions)
- Feature selection

### 4. Model Development
- Train/test split
- Baseline model establishment
- Multiple algorithm implementation:
  - Linear models
  - Tree-based models (Decision Tree, Random Forest)
  - Ensemble methods (Gradient Boosting)
  - Support Vector Machines
  - K-Nearest Neighbors
- Hyperparameter tuning
- Cross-validation

### 5. Model Evaluation
- Performance metrics calculation
- Confusion matrix analysis
- ROC curve and AUC
- Model comparison
- Final model selection

## 📊 Results

Results will be documented in the `results/` directory and include:

- Model performance comparison tables
- Confusion matrices and ROC curves
- Feature importance visualizations
- Final model metrics and insights
- Business recommendations

*(Results will be generated after running the analysis)*

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[PROJECT_PROPOSAL.md](docs/PROJECT_PROPOSAL.md)**: Comprehensive project proposal including objectives, methodology, timeline, and expected outcomes
- **METHODOLOGY.md**: Detailed methodology and technical approach *(to be created)*
- **[SHORTCOMINGS.md](docs/SHORTCOMINGS.md)**: Known limitations, assumptions, and areas for improvement

Additional documentation:
- Code documentation via docstrings in all modules
- Inline comments for complex logic
- Jupyter notebooks with markdown explanations

## 🤝 Contributing

This is an academic project for the AIDD course. However, suggestions and feedback are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 👥 Author

**[Your Name]**
- Course: AI in Data-Driven Decision Making (AIDD)
- Institution: Indiana University - Master of Science in Information Systems (MSIS)
- Date: November 2025

## 📄 License

This project is for educational purposes as part of the AIDD course at Indiana University.

## 🙏 Acknowledgments

- Indiana University MSIS Program
- AIDD Course Instructors
- Open-source community for the excellent libraries used in this project

## 📞 Contact

For questions or feedback regarding this project:
- GitHub Issues: [Create an issue](https://github.com/rzona-msis/AIDD-Final/issues)
- Email: [Your email]

---

**Last Updated:** November 9, 2025