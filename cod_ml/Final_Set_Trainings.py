import pandas as pd
import os
os.makedirs("final_results", exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


from sklearn.neighbors import KNeighborsClassifier

import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble as ensemble #For GPU processing of Random Forest and others

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, log_loss



def get_metrics(y_true, y_pred):
    # Returns precision, recall, and f1 for each class
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return p, r, f1

def save_results(dataset, model_name, accuracy, classification_report, log_loss_value):
    with open(f"final_results/{dataset}/Results_{dataset}.txt",encoding="utf-8", mode="a") as f:
        f.write("\n"+"="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Log Loss: {log_loss_value}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report)
    print(f"Results of {model_name} saved to final_results/{dataset}/Results_{dataset.split('.')[0]}.txt")




#=================================================================================================================================#
#================= Load Data ===================

df = pd.read_csv("csv/Full_Features_FINAL.csv")
dataset = "Full_Features_FINAL.csv"
print("Using dataset: Full_Features_FINAL.csv")

os.makedirs(f"final_results/{dataset}", exist_ok=True)

X = df.drop(columns = ["Id", "Class"])
y = df["Class"]

if y.min() == 1: #XGBoost foloseste clase DOAR incepand cu 0 deci modificam
    y = y - 1

#Data split with stratify for balanced class distribution in train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y) #Facem train test split, stratify face ca splitul de 80/20 sa fie EGAL pentru toate clasele


#=================================================================================================================================#
#==========Random Forest===========
print("Training Random Forest Classifier...")
rf_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42) #initializam
rf_model.fit(X_train, y_train) #antrenam
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
plt.savefig(f"results/{dataset}/Random_Forest_Feature_Importance_{dataset.split('.')[0]}.png")


cmrf = confusion_matrix(y_test, rf_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmrf, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class Random Forest {dataset}')
plt.savefig(f"results/{dataset}/Random_Forest_Confusion_Matrix_{dataset.split('.')[0]}.png")


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
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', device = 'cuda') #label_encoder -> ii spunem ca deja avem label-urile separate;  mlogloss -> Multi-class Logarithmic Loss
xgb_model.fit(X_train, y_train) #antrenam
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
plt.savefig(f"final_results/{dataset}/XGBoost_Feature_Importance_{dataset.split('.')[0]}.png")


cmxgb = confusion_matrix(y_test, xgb_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class XGBoost {dataset}')
plt.savefig(f"final_results/{dataset}/XGBoost_Confusion_Matrix_{dataset.split('.')[0]}.png")


xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_classification_report = classification_report(y_test, xgb_preds)
xgb_log_loss = log_loss(y_test, xgb_model.predict_proba(X_test)) #Calculam log loss pentru XGBoost

save_results(dataset, "XGBoost", xgb_accuracy, xgb_classification_report, xgb_log_loss)



print("\n--- XGBoost Accuracy ---")
print(xgb_accuracy)

print("\nDetailed XGBoost Report:")
print(xgb_classification_report)


#========== LightGBM ===========
print("Training LightGBM Classifier...")
# We use 'ovr' (one-vs-rest) for multi-class or 'multiclass' objective
lgbm_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, device="gpu") # Use device="cpu" if no GPU
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)

plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Bytes (LightGBM) {dataset}')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"final_results/{dataset}/LightGBM_Feature_Importance_{dataset.split('.')[0]}.png")


cmxgb = confusion_matrix(y_test, lgbm_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class LightGBM {dataset}')
plt.savefig(f"final_results/{dataset}/LightGBM_Confusion_Matrix_{dataset.split('.')[0]}.png")


lgbm_accuracy = accuracy_score(y_test, lgbm_preds)
lgbm_log_loss = log_loss(y_test, lgbm_model.predict_proba(X_test))
lgbm_report = classification_report(y_test, lgbm_preds)

save_results(dataset, "LightGBM", lgbm_accuracy, lgbm_report, lgbm_log_loss)
print(f"LightGBM Accuracy: {lgbm_accuracy}")




#========== SVM ===========
print("Training SVM Classifier (this might take a while)...")
# SVM needs probability=True to calculate log_loss
svm_model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Bytes (SVM) {dataset}')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"final_results/{dataset}/SVM_Feature_Importance_{dataset.split('.')[0]}.png")


cmxgb = confusion_matrix(y_test, svm_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class SVM {dataset}')
plt.savefig(f"final_results/{dataset}/SVM_Confusion_Matrix_{dataset.split('.')[0]}.png")


svm_accuracy = accuracy_score(y_test, svm_preds)
svm_log_loss = log_loss(y_test, svm_model.predict_proba(X_test))
svm_report = classification_report(y_test, svm_preds)

save_results(dataset, "SVM", svm_accuracy, svm_report, svm_log_loss)


#========== KNN ===========
print("Training KNN Classifier...")
knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Bytes (KNN) {dataset}')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"final_results/{dataset}/KNN_Feature_Importance_{dataset.split('.')[0]}.png")


cmxgb = confusion_matrix(y_test, knn_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class KNN {dataset}')
plt.savefig(f"final_results/{dataset}/KNN_Confusion_Matrix_{dataset.split('.')[0]}.png")


knn_accuracy = accuracy_score(y_test, knn_preds)
knn_log_loss = log_loss(y_test, knn_model.predict_proba(X_test))
knn_report = classification_report(y_test, knn_preds)

save_results(dataset, "KNN", knn_accuracy, knn_report, knn_log_loss)


#=================================================================================================================================#
#========= Comparatie performanta modelelor (F1-Score) pentru fiecare clasa ===================

class_labels = np.unique(y_test) + 1 #Luam lable-urile unice si adaugam 1 pentru a reveni la clasele originale (1-9)
rf_p, rf_r, rf_f1 = get_metrics(y_test, rf_preds) #Calculam precision, recall, f1 pentru fiecare clasa pentru Random Forest
xgb_p, xgb_r, xgb_f1 = get_metrics(y_test, xgb_preds) #Calculam precision, recall, f1 pentru fiecare clasa pentru XGBoost
lgbm_p, lgbm_r, lgbm_f1 = get_metrics(y_test, lgbm_preds) #Calculam precision, recall, f1 pentru fiecare clasa pentru LightGBM
svm_p, svm_r, svm_f1 = get_metrics(y_test, svm_preds) #Calculam precision, recall, f1 pentru fiecare clasa pentru SVM
knn_p, knn_r, knn_f1 = get_metrics(y_test, knn_preds) #Calculam precision, recall, f1 pentru fiecare clasa pentru KNN


x = np.arange(len(class_labels)) 
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting F1-Score (the best overall balance of precision/recall)
rects1 = ax.bar(x - width/2, rf_f1, width, label='Random Forest', color='#1f77b4')
rects2 = ax.bar(x + width/2, xgb_f1, width, label='XGBoost', color='#ff7f0e')
rects3 = ax.bar(x + width*1.5, lgbm_f1, width, label='LightGBM', color='#2ca02c')
rects4 = ax.bar(x + width*2.5, svm_f1, width, label='SVM', color='#d62728')
rects5 = ax.bar(x + width*3.5, knn_f1, width, label='KNN', color='#9467bd')
rects2 = ax.bar(x + width/2, xgb_f1, width, label='XGBoost', color='#ff7f0e')

ax.set_ylabel('F1-Score (0.0 to 1.0)')
ax.set_title(f'Model Performance Comparison by Malware Class {dataset}')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()
ax.set_ylim(0, 1.1) # Extra room for labels

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"final_results/{dataset}/Model_Performance_Comparison_{dataset.split('.')[0]}.png")


#We plot log loss in comparison for both classes
log_loss_values = [rf_log_loss, xgb_log_loss, lgbm_log_loss, svm_log_loss, knn_log_loss]
models = ['Random Forest', 'XGBoost', 'LightGBM', 'SVM', 'KNN']
plt.figure(figsize=(8, 6))
plt.bar(models, log_loss_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.ylabel('Log Loss (Lower is Better)')
plt.title(f'Model Log Loss Comparison {dataset}')
plt.savefig(f"final_results/{dataset}/Model_Log_Loss_Comparison_{dataset.split('.')[0]}.png")


acc_values = [rf_accuracy, xgb_accuracy, lgbm_accuracy, svm_accuracy, knn_accuracy]
loss_values = [rf_log_loss, xgb_log_loss, lgbm_log_loss, svm_log_loss, knn_log_loss]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy Plot
ax1.bar(models, acc_values, color='skyblue')
ax1.set_title('Accuracy Comparison')
ax1.set_ylim(0, 1.0)

# Log Loss Plot
ax2.bar(models, loss_values, color='salmon')
ax2.set_title('Log Loss Comparison (Lower is Better)')

plt.tight_layout()
plt.savefig(f"final_results/{dataset}/Final_Model_Comparison.png")