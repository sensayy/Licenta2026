# Malware Classification — Diploma Project

This repository contains code, scripts, and processed artifacts used for a final‑year diploma project on malware family classification. It explores both classical machine learning (feature‑based) models and a small convolutional neural network trained on image representations of binary files.

Key goals:
- Preprocess raw `.bytes` files and extract features (byte counts, sizes, labels).
- Convert byte streams into grayscale images for CNN experiments.
- Train and evaluate ML models (Random Forest, XGBoost) and a PyTorch CNN.
- Produce visualisations and reports used in analysis.

---

**Quick start**

- Activate the included virtual environment (PowerShell):

  ```powershell
  & .\malware_env\Scripts\Activate.ps1
  ```

- Note: This environment has the dependencies necessary ONLY for the CNN training and evaluation. Install the requirements.txt dependecies for everything else.

- If you don't use the included environment, create one and install dependencies:

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt  # if you add one
  ```

---

**Typical workflow**

1. Preprocessing
   - Parse `.bytes` files and remove address columns: use scripts under `cod_fisiere/` (e.g. `HexCount.py`).
   - Generate CSV feature files with `cod_csv/bytescsv.py` and `filesizecsv_create.py`.

2. Feature-based training
   - Use `cod_ml/Initial_Training.py` to train baseline models and produce evaluation plots.
   - Use `cod_ml/HyperParameterXG.py` to tune XGBoost hyperparameters.

3. Image-based training (CNN)
   - Convert parsed bytes to images with `cod_fisiere/hex_img.py` and manage images in `images/` or `images_resized/`.
   - Train the CNN with `cod_cnn/Image_Training.py` or the reformatted variants.
   - Evaluate using `cod_cnn/CNN_Report.py`.

---

**Repository layout (high level)**

- `parsed_bytes/` — parsed `.bytes` files used as input to feature/image generators.
- `images/`, `images_resized/` — generated grayscale images for CNN training.
- `csv/`, `modified_csv/`, `backup/` — CSV datasets and backups.
- `cod_csv/` — CSV generation and preprocessing helpers.
- `cod_fisiere/` — file utilities and image conversion scripts.
- `cod_cnn/` — CNN model training, evaluation, and reporting code.
- `cod_ml/` — classical ML experiments and hyperparameter tuning.
- `models_cnn/` — saved PyTorch model weights (checkpoints).

---

**Usage examples**

- Generate byte‑count CSVs (example):

  ```powershell
  python cod_csv/bytescsv.py
  ```

- Train XGBoost (example):

  ```powershell
  python cod_ml/Initial_Training.py
  ```

- Train CNN (example):

  ```powershell
  python cod_cnn/Image_Training.py
  ```

Adjust script arguments in the headers of each script — many scripts accept path/width/limit parameters.

---

**Notes & recommendations**

- This codebase was written for experimentation and reproducible analysis — not production use.
- Consider adding a `requirements.txt` or `environment.yml` to pin dependencies for reproducibility.
- Add a `LICENSE` if you plan to publish the repository.

---

If you'd like, I can:
- generate a `requirements.txt` by scanning imports,
- create a `CONTRIBUTING.md` and `LICENSE`,
- or produce a short usage tutorial showing the end‑to‑end pipeline.

