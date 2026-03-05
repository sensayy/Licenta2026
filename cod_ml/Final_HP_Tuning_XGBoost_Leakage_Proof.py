#TRAIN TEST SPLIT ADAPTAT PENTRU CNN-URI CA SA EVIT DATA LEAKAGE.
#DATORITA PROCENTAJULUI SUSPECT DE ACURATETE 100%-99.9999% AM DECIS SA FAC ACEASTA SCHIMBARE.

import pandas as pd
import os
os.makedirs("tuning_results", exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import time

import pickle as pk
import json
import joblib
import torch

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, precision_recall_fscore_support, log_loss

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

start_time = time.time()


def get_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return p, r, f1

def save_results(dataset, model_name, accuracy, classification_report, log_loss_value):
    with open(f"tuning_results/{dataset}/Results_{dataset}.txt", encoding="utf-8", mode="a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Log Loss: {log_loss_value}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report)
    print(f"Results of {model_name} saved to tuning_results/{dataset}/Results_{dataset.split('.')[0]}.txt")


df = pd.read_csv("csv/Full_Features_vocab2_sorted.csv")
dataset = "Full_Features_vocab2_sorted.csv"
print("Using dataset: Full_Features_vocab2_sorted.csv")

os.makedirs(f"tuning_results/{dataset}", exist_ok=True)

X = df.drop(columns=["Id", "Class"]).astype(np.float32)
y = df["Class"]

if y.min() == 1:
    y = y - 1

# =============================================================================
# SPLIT IDENTIC CU CNN-URILE
# Folosim torch.randperm cu acelasi seed (42) si aceleasi ratii (80/10/10)
# pentru a reproduce exact acelasi split ca in scripturile CNN.
# IMPORTANT: acest lucru functioneaza DOAR daca randurile din CSV sunt in
# aceeasi ordine ca imaginile incarcate de ImageFolder (sortate alfabetic
# dupa folder de clasa, apoi dupa numele fisierului). Foloseste scriptul
# sort_csv.py pentru a sorta CSV-ul inainte de a rula acest script.
# =============================================================================
total   = len(df)
n_train = int(0.8 * total)
n_val   = int(0.1 * total)
n_test  = total - n_train - n_val

indices       = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
train_indices = indices[:n_train]
val_indices   = indices[n_train:n_train + n_val]
test_indices  = indices[n_train + n_val:]

X_train = X.iloc[train_indices].reset_index(drop=True)
X_val   = X.iloc[val_indices].reset_index(drop=True)
X_test  = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_val   = y.iloc[val_indices].reset_index(drop=True)
y_test  = y.iloc[test_indices].reset_index(drop=True)

print(f"Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")


def objective(trial):
    param = {
        'device': 'cuda',
        'tree_method': 'hist',
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        #'n_estimators':     trial.suggest_int('n_estimators', 100, 2000),
        #'n_estimators':     1000,
        #'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2),
        #'max_depth':        trial.suggest_int('max_depth', 3, 10),
        #'subsample':        trial.suggest_float('subsample', 0.1, 1.0),
        #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        "n_estimators": 1174,
        "learning_rate": 0.048476600922859735,
        "max_depth": 3,
        "subsample": 0.9751250144753952,
        "colsample_bytree": 0.21001510994302897
    }

    model = XGBClassifier(**param)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Trial {trial.number}: Accuracy = {cv_scores.mean():.4f} with params: {param}")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return np.mean(cv_scores)


print("Starting Optuna Optimization...")
print(f"Time since init is: {time.time() - start_time:.2f} seconds")

study_name   = "malware_xgb_optimization"
storage_name = "sqlite:///tuning_results/optuna_study.db"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    load_if_exists=True
)
study.optimize(objective, n_trials=1)

print("\nOptimization Finished!")
print(f"Best Accuracy: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")

best_params = study.best_params
best_params['device']      = 'cuda'
best_params['tree_method'] = 'hist'

final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

final_preds = final_model.predict(X_test)
final_probs = final_model.predict_proba(X_test)

acc    = accuracy_score(y_test, final_preds)
l_loss = log_loss(y_test, final_probs)
report = classification_report(y_test, final_preds)
save_results(dataset, "XGBoost_Optuna_Tuned", acc, report, l_loss)

importance    = final_model.feature_importances_
feature_names = X.columns
indices_plot  = np.argsort(importance)[-20:]

plt.figure(figsize=(10, 8))
plt.title('Top 20 Most Important Features (Tuned XGBoost)')
plt.barh(range(len(indices_plot)), importance[indices_plot], align='center', color='skyblue')
plt.yticks(range(len(indices_plot)), [feature_names[i] for i in indices_plot])
plt.xlabel('Relative Importance')
plt.savefig("tuning_results/XGBoost_Feature_Importance.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_predictions(
    y_test, final_preds,
    display_labels=range(1, 10),
    cmap='Blues',
    ax=ax
)
plt.title('Confusion Matrix: Tuned XGBoost')
plt.savefig("tuning_results/XGBoost_Confusion_Matrix.png")
plt.close()

pk.dump(final_model, open("tuning_results/xgb_malware_model.pkl", "wb"))
final_model.save_model("tuning_results/xgb_malware_model.json")
joblib.dump(final_model, "tuning_results/xgb_model_joblib.pkl")

with open("tuning_results/best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)