import json

import pandas as pd
import os

from sklearn.utils import compute_sample_weight
import numpy as np


from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, log_loss
import matplotlib.pyplot as plt

import lightgbm as lgb


def get_metrics(y_true, y_pred):
    # Returns precision, recall, and f1 for each class
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return p, r, f1

def save_results(dataset, model_name, accuracy, classification_report, log_loss_value):
    with open(f"Organized_Modelling/vocabulary_testing/results/{dataset}/Results_{dataset}.txt",encoding="utf-8", mode="a") as f:
        f.write("\n"+"="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Log Loss: {log_loss_value}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report)
    print(f"Results of {model_name} saved to Organized_Modelling/vocabulary_testing/results/{dataset}/Results_{dataset.split('.')[0]}.txt")




#=================================================================================================================================#
#================= Load Data ===================


df = pd.read_csv("csv/Full_Features_vocab6.csv")
dataset = "Full_Features_vocab6.csv"
print("Using dataset: Full_Features_vocab6.csv")


            
os.makedirs(f"Organized_Modelling/vocabulary_testing/results/{dataset}", exist_ok=True)


os.makedirs(f"Organized_Modelling/vocabulary_testing/results/{dataset}", exist_ok=True)

with open("Organized_Modelling/split_indices.json", "r") as f:
    indices = json.load(f)

train_idx = indices["train"]
val_idx = indices["val"]
test_idx = indices["test"]

X = df.drop(columns=["Id", "Class"]).astype(np.float32)
y = df["Class"]

if y.min() == 1:
    y = y - 1 #Pentru a avea clasele de la 0 la 8 in loc de 1 la 9

X_train = X.iloc[train_idx].reset_index(drop=True)
X_val   = X.iloc[val_idx].reset_index(drop=True)
X_test  = X.iloc[test_idx].reset_index(drop=True)

y_train = y.iloc[train_idx].reset_index(drop=True)
y_val   = y.iloc[val_idx].reset_index(drop=True)
y_test  = y.iloc[test_idx].reset_index(drop=True)

print(f"Indices loaded! Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)


#=================================================================================================================================#
#==========Random Forest===========
print("Training Random Forest Classifier...")
rf_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42) #initializam
rf_model.fit(X_train, y_train, sample_weight=sample_weights) #antrenam
rf_preds = rf_model.predict(X_test) # testam

importance = rf_model.feature_importances_ #luam valorile cele mai importante pentru decizii
feature_names = X.columns #Luam numele coloanelor

indices = np.argsort(importance)[-20:] #sortam dupa importanta, primele 20 cele mai importante. Argsort sorteaza de la mic la mare deci luam cu - ca sa fie ultimele 20 cele mai importante
## np.argsort practic scoate doar pozitiile nu si valorile, nu sorteaza in sine matricea ci doar scoate indicii in ordine. Sorting the arguments!

plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Bytes (Random Forest) {dataset}')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/Random_Forest_Feature_Importance_{dataset.split('.')[0]}.png")


cmrf = confusion_matrix(y_test, rf_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmrf, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class Random Forest {dataset}')
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/Random_Forest_Confusion_Matrix_{dataset.split('.')[0]}.png")


rf_accuracy = accuracy_score(y_test, rf_preds)
rf_classification_report = classification_report(y_test, rf_preds)
rf_log_loss = log_loss(y_test, rf_model.predict_proba(X_test)) #Calculam log loss pentru Random Forest
save_results(dataset, "Random Forest", rf_accuracy, rf_classification_report, rf_log_loss)


print("\n--- Random Forest Accuracy ---")
print(rf_accuracy)

print("\nDetailed Random Forest Report:")
print(rf_classification_report)

#=================================================================================================================================#
#==========XGBoost===========
print("Training XGBoost Classifier...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', device = 'cuda', random_state=42) #label_encoder -> ii spunem ca deja avem label-urile separate;  mlogloss -> Multi-class Logarithmic Loss
xgb_model.fit(X_train, y_train, sample_weight=sample_weights) #antrenam
xgb_preds = xgb_model.predict(X_test) #testam

importance = xgb_model.feature_importances_ #luam valorile cele mai importante pentru decizii
feature_names = X.columns #Luam numele coloanelor

indices = np.argsort(importance)[-20:] #sortam dupa importanta, primele 20 cele mai importante. Argsort sorteaza de la mic la mare deci luam cu - ca sa fie ultimele 20 cele mai importante
## np.argsort practic scoate doar pozitiile nu si valorile, nu sorteaza in sine matricea ci doar scoate indicii in ordine. Sorting the arguments!


plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Bytes (XGBoost) {dataset}')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/XGBoost_Feature_Importance_{dataset.split('.')[0]}.png")


cmxgb = confusion_matrix(y_test, xgb_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class XGBoost {dataset}')
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/XGBoost_Confusion_Matrix_{dataset.split('.')[0]}.png")


xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_classification_report = classification_report(y_test, xgb_preds)
xgb_log_loss = log_loss(y_test, xgb_model.predict_proba(X_test)) #Calculam log loss pentru XGBoost

save_results(dataset, "XGBoost", xgb_accuracy, xgb_classification_report, xgb_log_loss)



print("\n--- XGBoost Accuracy ---")
print(xgb_accuracy)

print("\nDetailed XGBoost Report:")
print(xgb_classification_report)

#######################################################################################
#=================LightGBM===========
print("Training LightGBM Classifier...")
lgb_model = lgb.LGBMClassifier(random_state=42) #initializam
lgb_model.fit(X_train, y_train, sample_weight=sample_weights) #antrenam
lgb_preds = lgb_model.predict(X_test) # testam
importance = lgb_model.feature_importances_ #luam valorile cele mai importante pentru decizii
feature_names = X.columns #Luam numele coloanelor
indices = np.argsort(importance)[-20:] #sortam dupa importanta, primele 20 cele mai importante. Argsort sorteaza de la mic la mare deci luam cu - ca sa fie ultimele 20 cele mai importante
plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Bytes (LightGBM) {dataset}')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/LightGBM_Feature_Importance_{dataset.split('.')[0]}.png")

cm_lgb = confusion_matrix(y_test, lgb_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lgb, display_labels=range(1, 10))
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class LightGBM {dataset}')
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/LightGBM_Confusion_Matrix_{dataset.split('.')[0]}.png")
lgb_accuracy = accuracy_score(y_test, lgb_preds)
lgb_classification_report = classification_report(y_test, lgb_preds)
lgb_log_loss = log_loss(y_test, lgb_model.predict_proba(X_test))
save_results(dataset, "LightGBM", lgb_accuracy, lgb_classification_report, lgb_log_loss)
print("\n--- LightGBM Accuracy ---")
print(lgb_accuracy)
print("\nDetailed LightGBM Report:")
print(lgb_classification_report)



#========= Comparatie performanta modelelor (F1-Score) pentru fiecare clasa ===================

# 1. Get metrics for all three models
rf_p, rf_r, rf_f1 = get_metrics(y_test, rf_preds)
xgb_p, xgb_r, xgb_f1 = get_metrics(y_test, xgb_preds)
lgb_p, lgb_r, lgb_f1 = get_metrics(y_test, lgb_preds) # FIXED: Get actual metrics, not feature_importances

class_labels = np.unique(y_test) + 1 
x = np.arange(len(class_labels)) 
width = 0.25 # Narrower width to fit 3 bars

fig, ax = plt.subplots(figsize=(14, 7))

# Plotting F1-Score for all three
rects1 = ax.bar(x - width, rf_f1, width, label='Random Forest', color='#1f77b4')
rects2 = ax.bar(x, xgb_f1, width, label='XGBoost', color='#ff7f0e')
rects3 = ax.bar(x + width, lgb_f1, width, label='LightGBM', color='#2ca02c')

ax.set_xlabel('Malware Class')
ax.set_ylabel('F1-Score')
ax.set_title(f'F1-Score Comparison by Class: {dataset}')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.1) 

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/Model_F1_Comparison_{dataset.split('.')[0]}.png")

# 2. Fixed Log Loss Comparison Plot
log_loss_values = [rf_log_loss, xgb_log_loss, lgb_log_loss]
models = ['Random Forest', 'XGBoost', 'LightGBM']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, log_loss_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Add labels on top of bars for clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, round(yval, 4), ha='center', va='bottom')

plt.ylabel('Log Loss (Lower is Better)')
plt.title(f'Log Loss Comparison: {dataset}')
plt.grid(axis='y', alpha=0.3)
plt.savefig(f"Organized_Modelling/vocabulary_testing/results/{dataset}/Model_Log_Loss_Comparison_{dataset.split('.')[0]}.png")

