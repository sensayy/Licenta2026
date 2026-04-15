import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import pickle as pk
import json
import joblib
import torch

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support, log_loss
from sklearn.utils.class_weight import compute_sample_weight

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

start_time = time.time()


################## Functii ###################################



def save_results(dataset, model_name, accuracy, classification_report, log_loss_value):
    with open(f"Organized_Modelling/XGBoost/training_results/{dataset}/Results_{dataset}.txt", encoding="utf-8", mode="a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Log Loss: {log_loss_value}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report)
    print(f"Results of {model_name} saved to Organized_Modelling/XGBoost/training_results/{dataset}/Results_{dataset.split('.')[0]}.txt")


    #########################


df = pd.read_csv("csv/Full_Features_vocab2_sorted.csv")
dataset = "Full_Features_vocab2_sorted.csv"
print("Using dataset: Full_Features_vocab2_sorted.csv")

os.makedirs(f"Organized_Modelling/XGBoost/training_results/{dataset}", exist_ok=True)

X = df.drop(columns=["Id", "Class"]).astype(np.float32)
y = df["Class"]
y = y - 1 #Pentru a avea clasele de la 0 la 8 in loc de 1 la 9

indices = np.arange(len(df))
y_labels = df["Class"].values


############### Facem split 80/20 ##################
train_indices, tval_indices, y_train_strat, y_tval_strat = train_test_split(
    indices, y_labels, 
    train_size=0.8, 
    stratify=y_labels, 
    random_state=42
)

################## Facem split 50/50 pentru val/test din cei 20% ramasi ##################
val_indices, test_indices, y_val_strat, y_test_strat = train_test_split(
    tval_indices, y_tval_strat, 
    test_size=0.5, 
    stratify=y_tval_strat, 
    random_state=42
)

X_train = X.iloc[train_indices].reset_index(drop=True)
X_val   = X.iloc[val_indices].reset_index(drop=True)
X_test  = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_val   = y.iloc[val_indices].reset_index(drop=True)
y_test  = y.iloc[test_indices].reset_index(drop=True)


split_indices = {
    "train": train_indices.tolist(),
    "val": val_indices.tolist(),
    "test": test_indices.tolist()
}
with open("Organized_Modelling/split_indices.json", "w") as f:
    json.dump(split_indices, f)

print(f"Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', device = 'cuda', random_state=42)

xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
xgb_preds = xgb_model.predict(X_test)


importance = xgb_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importance)[-20:]

plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Bytes (XGBoost) {dataset}')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"Organized_Modelling/XGBoost/training_results/{dataset}/XGBoost_Feature_Importance_{dataset.split('.')[0]}.png")

cmxgb = confusion_matrix(y_test, xgb_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=range(0, 9)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class XGBoost {dataset}')
plt.savefig(f"Organized_Modelling/XGBoost/training_results/{dataset}/XGBoost_Confusion_Matrix_{dataset.split('.')[0]}.png")

xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_classification_report = classification_report(y_test, xgb_preds)
xgb_log_loss = log_loss(y_test, xgb_model.predict_proba(X_test))


print("\n--- XGBoost Accuracy ---")
print(xgb_accuracy)

print("\n--- XGBoost Log Loss ---")
print(xgb_log_loss)

print("\nDetailed XGBoost Report:")
print(xgb_classification_report)



save_results(dataset, "XGBoost", xgb_accuracy, xgb_classification_report, xgb_log_loss)


model_save_path = f"Organized_Modelling/XGBoost/models/{dataset.split('.')[0]}"
os.makedirs(model_save_path, exist_ok=True)
xgb_model.save_model(f"{model_save_path}/xgb_malware_model.json")
joblib.dump(xgb_model, f"{model_save_path}/xgb_model_joblib.pkl")

with open(f"{model_save_path}/xgb_model_pickle.pkl", "wb") as f:
    pk.dump(xgb_model, f)

print(f"\nModel saved successfully in multiple formats at: {model_save_path}")