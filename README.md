# Malware Classification via Multi-Modal Ensemble Learning

> **📚 Final Year Diploma Project** - A comprehensive malware classification system using Deep Learning and Machine Learning approaches.

## 🎯 Project Overview

This project implements a **multi-modal ensemble learning system** for malware classification, addressing the challenge of accurately identifying and categorizing various malware families. The system combines convolutional neural networks (CNNs) with traditional machine learning models to achieve robust classification across **9 malware families** from the Microsoft BIG 2015 Malware Classification Challenge.

### Problem Statement

Malware classification is critical for cybersecurity defense but presents significant challenges:
- Diverse malware representations (binary code, assembly, byte sequences)
- High-dimensional feature spaces
- Need for real-time inference
- Trade-off between accuracy and computational efficiency

### Solution

A **weighted ensemble approach** combining:
- 🖼️ **Visual CNNs**: Byte and assembly code visualizations
- 📊 **Statistical Models**: XGBoost with engineered features
- 🔄 **Ensemble Voting**: Weighted predictions from multiple models

---

## 📋 System Architecture

### Multi-Modal Approach

```
Raw Malware Files
    ├── Byte Extraction → Byte Images → CNN (EfficientNet-B4) [40%]
    ├── Assembly Code  → ASM Images  → CNN (EfficientNet-B3) [30%]
    └── Feature Extraction → XGBoost Classifier [30%]
    
    └─→ Ensemble Voting → Final Classification
```

### Key Components

| Component | Role | Technology |
|-----------|------|-----------|
| **Data Processing** | Byte extraction, image generation | Python, PIL, NumPy |
| **Byte CNN** | Visual classification from hex dumps | PyTorch, EfficientNet-B4 |
| **ASM CNN** | Visual classification from assembly code | PyTorch, EfficientNet-B3 |
| **Feature ML** | Statistical classification | XGBoost, scikit-learn |
| **Ensemble** | Weighted prediction combination | Custom voting logic |

### Target Classes (9 Malware Families)

1. **Ramnit** - Worm/trojan hybrid
2. **Lollipop** - Download adware
3. **Kelihos_v3** - Botnet
4. **Vundo** - Rogue software/spyware
5. **Simda** - Trojan banker
6. **Tracur** - Trojan banker
7. **Kelihos_v1** - Botnet (older version)
8. **Obfuscator** - Packed/encrypted malware
9. **Gatak** - Trojan downloader

---

## 📁 Project Structure

```
Licenta2026/
├── cod_cnn/                          # CNN model implementations
│   ├── TorchCNN_V1_ASM.py           # EfficientNet-B3 for ASM images
│   ├── TorchCNN_V2.py               # EfficientNet-B4 for byte images
│   └── Image_Training_CNN_*.py      # Training scripts
│
├── cod_ml/                          # Traditional ML models
│   ├── Final_Set_Trainings.py      # XGBoost, RF, SVM, KNN, LightGBM
│   ├── Final_HP_Tuning_XGBoost.py  # Hyperparameter tuning
│   └── Initial_Training.py          # Baseline model training
│
├── cod_csv/                         # Data processing & CSV manipulation
│   ├── ASM_Handler.py               # Assembly code extraction
│   ├── csv_modifier.py              # Feature engineering
│   ├── bytescsv.py                  # Byte feature extraction
│   └── filesize.py                  # File size analysis
│
├── cod_fisiere/                     # Image generation from binary data
│   ├── asm_img.py                   # Convert ASM to grayscale images
│   ├── hex_img.py                   # Convert bytes to grayscale images
│   ├── image_class_sort.py          # Organize images by class
│   └── data_graph.py                # Visualization utilities
│
├── Mega-Model/                      # Ensemble model
│   ├── Mega-Model_WholeSET.py      # Final ensemble classifier
│   ├── Mega-Model_V2_80-20.py      # Train/test split variant
│   ├── Mega-Model_V2_HiddenData.py # External validation
│   └── *.pth                        # Pre-trained model weights
│
├── csv/                             # Feature-based datasets
│   ├── Full_Features_vocab*.csv     # Different feature vocabularies
│   ├── Full_Features_FINAL.csv      # Final feature set
│   ├── trainLabels.csv              # Training labels
│   └── ByteCount_Size_Labels.csv    # Size-based features
│
├── models_cnn/                      # Saved neural network weights
│   ├── malware_cnn_best_v1_asm.pth    # Best ASM model
│   ├── malware_cnn_best_v2.pth        # Best byte model
│   └── malware_cnn_final_*.pth        # Final checkpoints
│
├── results/                         # Classification results & confusion matrices
├── graphs/                          # Visualization outputs
├── tuning_results/                  # Hyperparameter tuning logs
├── final_results/                   # Final evaluation metrics
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🎨 Visualization Components

### Graph Types Generated

1. **Confusion Matrices**
   - Per-model and ensemble confusion matrices
   - Shows classification accuracy across all 9 malware families
   - Located in: `results/`

2. **Training Curves**
   - Loss and accuracy over epochs
   - Model convergence behavior
   - Located in: `cnn_graphs/`

3. **Feature Distribution**
   - Byte distribution histograms
   - Feature importance from XGBoost
   - Class imbalance visualization

4. **Ensemble Analysis**
   - Prediction confidence distributions
   - Stress test results
   - Weighted voting impact visualization
   - Located in root directory (`.png` files)

### Key Visualizations

- `confusion_matrix.png` - Ensemble model accuracy breakdown
- `confidence_distribution.png` - Prediction confidence analysis
- `Stress_Test_Ensemble.png` - Model robustness evaluation

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (for GPU acceleration - optional but recommended)
- **8GB+ RAM** (minimum; 16GB+ recommended)
- **50GB+ disk space** (for images and models)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Licenta2026
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA (optional):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Dependencies

| Package | Purpose |
|---------|---------|
| **torch, torchvision** | Deep learning framework, pre-trained models |
| **xgboost** | Gradient boosting classifier |
| **scikit-learn** | ML utilities, metrics, preprocessing |
| **pandas, numpy** | Data manipulation and numerical computing |
| **pillow** | Image processing |
| **matplotlib, seaborn** | Visualization and plotting |
| **tqdm** | Progress bars |

---

## 📊 Usage Guide

### 1. Data Preparation

**Step 1a: Extract and prepare raw data**
```bash
# Place malware files in ./train directory
# Ensure labels are available in CSV format

python cod_csv/filesize.py              # Extract file sizes
python cod_csv/bytescsv.py              # Extract byte statistics
```

**Step 1b: Generate images from binary data**
```bash
# Generate byte visualization images
python cod_fisiere/hex_img.py
# Output: ./images/*.png

# Generate assembly code visualization images
python cod_fisiere/asm_img.py
# Output: ./asm_images/*.png
```

**Step 1c: Organize and process features**
```bash
# Create feature datasets
python cod_csv/csv_modifier.py

# Organize images by class
python cod_fisiere/image_class_sort.py
```

### 2. Individual Model Training

**Train Byte CNN (EfficientNet-B4):**
```bash
cd cod_cnn
python TorchCNN_V2.py
# Saves best model to: ../models_cnn/malware_cnn_best_v2.pth
```

**Train ASM CNN (EfficientNet-B3):**
```bash
cd cod_cnn
python TorchCNN_V1_ASM.py
# Saves best model to: ../models_cnn/malware_cnn_best_v1_asm.pth
```

**Train XGBoost classifier:**
```bash
cd cod_ml
python Final_Set_Trainings.py
# Saves model to: ../Mega-Model/xgb_malware_model.pkl
```

**Hyperparameter tuning (optional):**
```bash
cd cod_ml
python Final_HP_Tuning_XGBoost.py
# Generates best parameters for the XGBoost model
```

### 3. Ensemble Model Inference

**Run the complete ensemble on your dataset:**
```bash
cd Mega-Model
python Mega-Model_WholeSET.py
```

**Expected Output:**
```
Using device: cuda
Bytes CNN loaded successfully.
ASM CNN loaded successfully.

Processing: [████████████████] 100%
Accuracy: 0.8234
Classification Report:
              precision    recall  f1-score   support
      Ramnit     0.85      0.82      0.83      450
    Lollipop     0.81      0.79      0.80      380
  Kelihos_v3     0.88      0.85      0.86      520
       Vundo     0.92      0.90      0.91      410
       Simda     0.79      0.81      0.80      350
      Tracur     0.86      0.84      0.85      470
  Kelihos_v1     0.84      0.86      0.85      440
  Obfuscator     0.87      0.88      0.87      520
       Gatak     0.80      0.82      0.81      460
```

### 4. Custom Predictions

**Classify a single file:**
```python
from Mega_Model_WholeSET import CNN, CNN_ASM, CLASS_NAMES
import torch
from PIL import Image
import numpy as np

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ... (model loading code)

# Prepare input
byte_image = Image.open('path/to/byte_image.png')
transforms = ...  # Apply preprocessing

# Get predictions
with torch.no_grad():
    byte_pred = byte_cnn(byte_tensor)
    asm_pred = asm_cnn(asm_tensor)
    xgb_pred = xgb_model.predict_proba(features)
    
# Weighted ensemble
ensemble_pred = (0.40 * byte_pred + 
                0.30 * asm_pred + 
                0.30 * xgb_pred)

malware_class = CLASS_NAMES[ensemble_pred.argmax()]
print(f"Predicted: {malware_class}")
```

---


## 🔬 Experimental Variations

The project includes several experimental variants to test robustness:

| Script | Purpose |
|--------|---------|
| `Mega-Model_V2_80-20.py` | Standard 80/20 train-test split |
| `Mega-Model_V2_HiddenData.py` | External validation on separate dataset |
| `Mega-Model_V2_Images_Split.py` | Image-only evaluation |
| `Mega-Model_V2_Leakage_Test.py` | Data leakage detection |
| `Mega-Model_V2_Noise.py` | Robustness to noisy data |

---

## 📂 Key Datasets

| Dataset | Features | Purpose |
|---------|----------|---------|
| `Full_Features_FINAL.csv` | 1000+ engineered features | Baseline feature set |
| `Full_Features_vocab*.csv` | Variable vocabulary sizes | Testing vocabulary impact |
| `ByteCount_Size_Labels.csv` | File size, byte statistics | Size-based features |
| Image directories | Grayscale visualizations | CNN input data |

---

## 🛠️ Customization

### Adjust Ensemble Weights

Edit the weighting in `Mega-Model_WholeSET.py`:
```python
W_XGB      = 0.30     # XGBoost weight
W_BYTE_CNN = 0.40     # Byte CNN weight
W_ASM_CNN  = 0.30     # ASM CNN weight
```

### Modify CNN Architecture

Edit EfficientNet versions in `cod_cnn/TorchCNN_V*.py`:
```python
# Change model backbone
self.base = models.efficientnet_b5(weights=None)  # Different EfficientNet version
```

### Tune XGBoost Parameters

Modify in `cod_ml/Final_HP_Tuning_XGBoost.py`:
```python
xgb_params = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 300,
    # ... other parameters
}
```

---

## 📝 File Naming Convention

- **Model weights**: `malware_cnn_best_v*.pth`, `malware_cnn_final_*.pth`
- **CSV datasets**: `Full_Features_*.csv`, `*_modified.csv`
- **Results**: `Results_*.txt`, `confusion_matrix_*.png`
- **Pickled models**: `xgb_malware_model.pkl`

---

## 🔍 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in model training scripts or use CPU
```python
device = torch.device('cpu')  # Force CPU usage
```

### Issue: Missing image files
**Solution**: Regenerate images from binary data
```bash
python cod_fisiere/hex_img.py
python cod_fisiere/asm_img.py
```

### Issue: Feature dimension mismatch
**Solution**: Rerun CSV processing with consistent feature extraction
```bash
python cod_csv/csv_modifier.py
```

### Issue: Model loading fails
**Solution**: Ensure model files are in correct directories and PyTorch version matches
```bash
pip install --upgrade torch torchvision
```

---

## 📚 Technical Stack

- **Deep Learning**: PyTorch, EfficientNet
- **Machine Learning**: XGBoost, scikit-learn, LightGBM
- **Data Processing**: Pandas, NumPy, PIL
- **Visualization**: Matplotlib, Seaborn
- **Hardware**: NVIDIA CUDA 11.8+ (GPU support)

---

## 📊 Dataset Reference

This project uses the **Microsoft BIG 2015 Malware Classification Challenge** dataset:
- **Training samples**: ~11,000 files
- **Test samples**: ~2,000 files
- **Classes**: 9 malware families
- **File formats**: Binary executables (.exe, .scr, etc.)

---

## 🎓 Project Context

**Diploma Project Details:**
- **Type**: Final year degree project (Licență)
- **Focus**: Multi-modal machine learning for cybersecurity
- **Innovation**: Ensemble approach combining visual and statistical methods
- **Scope**: Complete pipeline from data extraction to production-ready inference

---

## 👨‍💻 Development Notes

### Code Organization Philosophy

1. **Separation of concerns**: Data processing, modeling, and evaluation are in separate directories
2. **Reproducibility**: Version control for models (`_v1`, `_v2`, `_best`, `_final`)
3. **Experimentation**: Multiple variants for robustness testing
4. **Documentation**: Inline comments in Python files (Romanian and English)

### Best Practices Applied

- ✅ Modular code structure
- ✅ Balanced class handling with WeightedRandomSampler
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Cross-validation for model evaluation
- ✅ Ensemble voting for robustness
- ✅ GPU acceleration support
- ✅ Progress tracking with tqdm

---

## 📄 Additional Resources

- **Papers**: Microsoft BIG 2015 challenge documentation
- **Models**: Pre-trained EfficientNet weights included
- **Results**: Complete confusion matrices and classification reports in `results/`
- **Graphs**: Visualizations in `graphs/` and `cnn_graphs/`

---

## 📞 Support

For issues or questions:

1. Check the relevant `*.py` script documentation
2. Review inline comments (mostly in English)
3. Check the `results/` directory for detailed outputs
4. Examine classification reports for per-class performance

---

## 📜 License

This project is provided as-is for educational and research purposes.

---

**Last Updated**: March 2026  
**Status**: Diploma Project - Complete  
**Python Version**: 3.8+  
**PyTorch Version**: 2.0+
