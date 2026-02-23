# Malware Classification with XGBoost

A machine learning project for binary file classification using byte-level analysis and file size features, powered by XGBoost and hyperparameter optimization.

## What This Project Does

This project implements a multi-class malware classification system that analyzes binary files at the byte level to identify file types or malware families. It extracts statistical features from file contents (byte frequency distributions) combined with file size metadata, then trains ensemble machine learning models (XGBoost and Random Forest) to classify unknown files.

Key capabilities:
- **Byte-level feature extraction**: Analyzes raw binary content by counting hexadecimal byte occurrences (256 possible values + unknown bytes)
- **File size analysis**: Extracts size information from compressed archives
- **Multi-class classification**: Supports classification into multiple file categories/malware families
- **GPU acceleration**: Optional CUDA support for faster model training on compatible hardware

## Why This Project is Useful

- **Fast inference**: Trained models can classify files in milliseconds
- **Lightweight features**: No complex static/dynamic analysis required—just binary content and size
- **Hyperparameter optimized**: Includes RandomizedSearchCV for tuning XGBoost parameters
- **Comparative evaluation**: Trains both XGBoost and Random Forest for performance comparison
- **GPU-ready**: Can leverage NVIDIA GPUs to accelerate training on large datasets

## Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU + CUDA Toolkit (optional but recommended for `HyperParameterXG.py`)
- 7z archive support (for file extraction from `.7z` files)

### Installation

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd path/to/Licenta
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install pandas scikit-learn xgboost numpy matplotlib py7zr cupy
   ```

   For GPU support with XGBoost, ensure CUDA is installed, then reinstall XGBoost:
   ```bash
   pip install --upgrade xgboost
   ```

### Verify GPU Setup (Optional)

If using GPU acceleration, verify CUDA support:
```bash
python test.py
```

This will confirm your GPU is properly configured and accessible to XGBoost.

### Data Preparation

1. **Extract byte data**: Place your binary files in the `parsed_bytes/` directory as hex-encoded `.bytes` files
2. **Prepare size data**: Extract file sizes from your 7z archive:
   ```bash
   python filesize.py
   ```
   This creates `filesizes.csv`

3. **Generate byte features**:
   ```bash
   python bytescsv.py
   ```
   This creates `ByteCount_Size_Labels.csv` with byte frequency features

4. **Add labels**: Ensure your CSV contains a `Class` column with integer class labels

### Training Models

#### Quick Start: Standard Training
Train both XGBoost and Random Forest models and evaluate them:
```bash
python Initial_Training.py
```

This will:
- Load training data from `ByteCount_Size_Labels.csv`
- Split data 80/20 (stratified by class)
- Train both classifiers
- Display accuracy, classification reports, and confusion matrices
- Generate feature importance plots for XGBoost

#### Hyperparameter Tuning: Optimized XGBoost
For better performance, use RandomizedSearchCV to find optimal hyperparameters:
```bash
python HyperParameterXG.py
```

This performs a randomized grid search over:
- Learning rate
- Max depth
- Subsample ratio
- Colsample by tree
- Number of estimators
- Regularization (gamma, lambda, alpha)

Results are saved to `Best Parameters XGBoost from Tuning` file.

### Project Structure

```
Licenta/
├── bytescsv.py                      # Extract byte frequency features from binary files
├── filesize.py                      # Extract file sizes from 7z archive
├── Initial_Training.py              # Train XGBoost and Random Forest models
├── HyperParameterXG.py              # Hyperparameter tuning with RandomizedSearchCV
├── test.py                          # GPU/CUDA verification script
├── organizer.py                     # Sort CSV by class label
├── ByteCount_Size_Labels.csv        # Training dataset with features and labels
├── parsed_bytes/                    # Binary files in hex format (.bytes files)
├── train/                           # (Optional) Source training data
└── graphs/                          # Output directory for visualization plots
```

### Usage Example

```python
# Load and use a trained model
import pickle
import pandas as pd
from xgboost import XGBClassifier

# Load trained model
model = XGBClassifier()
# model.load_model('xgboost_model.bin')

# Prepare features (matching training data format)
test_data = pd.read_csv('ByteCount_Size_Labels.csv')
X_test = test_data.drop(columns=['Id', 'Class'])

# Make predictions
predictions = model.predict(X_test)
```

## Features and Architecture

### Data Pipeline
1. **Raw binary files** → Hex-encoded text in `parsed_bytes/`
2. **Byte counting** → Creates vocabulary of 256 hex byte values + unknown marker
3. **Feature extraction** → Generates frequency vector for each file
4. **Size features** → Appends file size metadata
5. **Model training** → XGBoost/RF on combined feature set

### Machine Learning Models
- **XGBoost**: Gradient boosted trees with custom loss function for multi-class (`mlogloss`)
- **Random Forest**: Ensemble of 100 decision trees
- Both models use stratified train/test split (80/20) to maintain class distributions

### Key Hyperparameters (Tuned via HyperParameterXG.py)
- `learning_rate`: Controls gradient descent step size
- `max_depth`: Maximum tree depth
- `subsample`: Fraction of samples used per tree
- `colsample_bytree`: Fraction of features used per tree
- `n_estimators`: Number of boosting rounds
- `gamma`, `lambda`, `alpha`: Regularization parameters

## Where to Get Help

- **GPU Issues**: Check [XGBoost GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- **CUDA Setup**: See [NVIDIA CUDA Toolkit installation guide](https://developer.nvidia.com/cuda-toolkit)
- **Scikit-learn**: Refer to [scikit-learn documentation](https://scikit-learn.org/stable/)
- **Data preparation issues**: Run individual scripts (`bytescsv.py`, `filesize.py`) in isolation to debug

## Maintainer

This project is maintained as part of a thesis/research work on malware classification.

## Contributing

To extend this project:
1. Improve feature engineering (entropy, n-grams, etc.)
2. Add more model variants (SVM, Neural Networks)
3. Implement cross-validation strategies
4. Create visualization tools for decision boundaries
5. Optimize batch processing for large datasets

## License

See LICENSE file for details.

---

**Note**: This project is optimized for GPU training. The `HyperParameterXG.py` script uses `tree_method='hist'` and `device='cuda'` for maximum performance on NVIDIA GPUs. For CPU-only systems, modify the XGBoost initialization parameters accordingly.
