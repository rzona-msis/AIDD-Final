# Results Directory

This directory contains analysis results, visualizations, and output files.

## Contents

This directory will contain:

- **Figures and Plots** (`.png`, `.jpg`, `.pdf`)
  - Confusion matrices
  - ROC curves
  - Feature importance plots
  - Distribution plots
  - Correlation heatmaps

- **Tables and Reports** (`.csv`, `.xlsx`)
  - Model performance comparisons
  - Feature importance rankings
  - Prediction results
  - Statistical summaries

- **HTML Reports** (`.html`)
  - Interactive visualizations
  - Dashboard reports

## Usage

Results are typically saved from notebooks or scripts using matplotlib/seaborn:

```python
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
```

Or from pandas DataFrames:

```python
results_df.to_csv('results/model_comparison.csv', index=False)
```

## Note

Result files are excluded from git (see `.gitignore`) to keep repository size manageable.
Store only the most important final results in git if needed.
