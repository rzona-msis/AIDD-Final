# Models Directory

This directory contains trained machine learning models.

## Contents

After training models, this directory will contain:

- `*.pkl` - Scikit-learn models saved with joblib
- `*.h5` - Keras/TensorFlow models
- `*.pt` / `*.pth` - PyTorch models
- `*.joblib` - Alternative format for scikit-learn models

## Usage

Models are saved using the `ModelTrainer.save_model()` method:

```python
trainer.save_model('Random Forest', 'models/random_forest_model.pkl')
```

And loaded using:

```python
trainer.load_model('Random Forest', 'models/random_forest_model.pkl')
```

## Note

Model files are typically large and are excluded from git (see `.gitignore`).
