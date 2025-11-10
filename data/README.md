# Data Directory

This directory contains all project data files.

## Structure

```
data/
├── raw/           # Original, immutable data files
└── processed/     # Cleaned and processed data files
```

## Raw Data (`raw/`)

Store original, unmodified data files here:
- CSV files
- Excel spreadsheets
- JSON files
- Database exports

**Important:** Never modify files in this directory. Keep original data intact.

## Processed Data (`processed/`)

Store cleaned and processed data files here:
- Cleaned datasets
- Feature-engineered data
- Train/test splits
- Transformed data

## Usage

### Loading Raw Data
```python
from src.data_loader import DataLoader

loader = DataLoader(data_dir='data')
df = loader.load_csv('your_data.csv', raw=True)
```

### Saving Processed Data
```python
loader.save_data(cleaned_df, 'cleaned_data.csv', processed=True)
```

## Data Sources

Document your data sources here:
- Source name and URL
- Date acquired
- Version/snapshot information
- License and usage terms

## Note

Data files are excluded from git (see `.gitignore`) due to:
- Large file sizes
- Privacy/security concerns
- Licensing restrictions

To share the project, provide:
1. Instructions for obtaining the data
2. Data download scripts (if public)
3. Sample/dummy data for testing
