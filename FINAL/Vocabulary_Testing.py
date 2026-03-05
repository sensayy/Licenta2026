import pandas as pd
import os
os.makedirs("results", exist_ok=True)
import numpy as np


from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, log_loss
import matplotlib.pyplot as plt


def get_metrics(y_true, y_pred):
    # Returns precision, recall, and f1 for each class
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return p, r, f1

def save_results(dataset, model_name, accuracy, classification_report, log_loss_value):
    with open(f"results/{dataset}/Results_{dataset}.txt",encoding="utf-8", mode="a") as f:
        f.write("\n"+"="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Log Loss: {log_loss_value}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report)
    print(f"Results of {model_name} saved to results/{dataset}/Results_{dataset.split('.')[0]}.txt")




#=================================================================================================================================#
#================= Load Data ===================

'''
Set 1 (with file size features)
df = pd.read_csv(f"csv/ByteCount_Size_Labels.csv") Testam si fara asm
df = pd.read_csv(f"csv/Full_Features_v1.csv") Testam si cel initial
df = pd.read_csv(f"csv/Full_Features_vocab1.csv")
df = pd.read_csv(f"csv/Full_Features_vocab2.csv")
df = pd.read_csv(f"csv/Full_Features_vocab3.csv")
df = pd.read_csv(f"csv/Full_Features_vocab4.csv")

Set 2 (without file size features)

df = pd.read_csv(f"modified_csv/ByteCount_Labels.csv") Testam si fara asm
df = pd.read_csv(f"modified_csv/Full_Features_v1_modified.csv") Testam si cel initial
df = pd.read_csv(f"modified_csv/Full_Features_vocab1_modified.csv")
df = pd.read_csv(f"modified_csv/Full_Features_vocab2_modified.csv")
df = pd.read_csv(f"modified_csv/Full_Features_vocab3_modified.csv")
df = pd.read_csv(f"modified_csv/Full_Features_vocab4_modified.csv")
df = pd.read_csv(f"modified_csv/Full_Features_vocab5_modified.csv")

'''


original = False #Quick fix ca sa testez si cel doar pe bytes files si file size
Data = 1 # 1 pentru datasetul initial, 2 pentru cel modificat (fara file size features)
version = 2 #1-4 pentru fiecare tip de vocabular
vocab = True #True daca le folosim pe cele din setul cu mai multe tipuri de vocabular, False daca folosim versiuna initiala cu cateva op-codes

if original:
    if Data == 1:
        df = pd.read_csv("csv/ByteCount_Size_Labels.csv")
        dataset = "ByteCount_Size_Labels.csv"
        print("Using dataset: ByteCount_Size_Labels.csv")
    else:
        df = pd.read_csv("modified_csv/ByteCount_Labels.csv")
        dataset = "ByteCount_Labels.csv"
        print("Using dataset: ByteCount_Labels.csv")
else:
    if Data == 1:
        if vocab:
            df = pd.read_csv(f"csv/Full_Features_vocab{version}.csv")
            dataset = f"Full_Features_vocab{version}.csv"
            print(f"Using dataset: Full_Features_vocab{version}.csv")
        else:
            df = pd.read_csv(f"csv/Full_Features_v1.csv")
            dataset = "Full_Features_v1.csv"
            print("Using dataset: Full_Features_v1.csv")
    else:
        if vocab:
            df = pd.read_csv(f"modified_csv/Full_Features_vocab{version}_modified.csv")
            dataset = f"Full_Features_vocab{version}_modified.csv"
            print(f"Using dataset: Full_Features_vocab{version}_modified.csv")
        else:
            df = pd.read_csv(f"modified_csv/Full_Features_v1_modified.csv")
            dataset = "Full_Features_v1_modified.csv"
            print("Using dataset: Full_Features_v1_modified.csv")



            
os.makedirs(f"results/{dataset}", exist_ok=True)

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
plt.savefig(f"results/{dataset}/XGBoost_Feature_Importance_{dataset.split('.')[0]}.png")


cmxgb = confusion_matrix(y_test, xgb_preds)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cmxgb, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Confusion Matrix: Actual vs. Predicted Malware Class XGBoost {dataset}')
plt.savefig(f"results/{dataset}/XGBoost_Confusion_Matrix_{dataset.split('.')[0]}.png")


xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_classification_report = classification_report(y_test, xgb_preds)
xgb_log_loss = log_loss(y_test, xgb_model.predict_proba(X_test)) #Calculam log loss pentru XGBoost

save_results(dataset, "XGBoost", xgb_accuracy, xgb_classification_report, xgb_log_loss)



print("\n--- XGBoost Accuracy ---")
print(xgb_accuracy)

print("\nDetailed XGBoost Report:")
print(xgb_classification_report)



#=================================================================================================================================#
#========= Comparatie performanta modelelor (F1-Score) pentru fiecare clasa ===================

class_labels = np.unique(y_test) + 1 #Luam lable-urile unice si adaugam 1 pentru a reveni la clasele originale (1-9)
rf_p, rf_r, rf_f1 = get_metrics(y_test, rf_preds) #Calculam precision, recall, f1 pentru fiecare clasa pentru Random Forest
xgb_p, xgb_r, xgb_f1 = get_metrics(y_test, xgb_preds) #Calculam precision, recall, f1 pentru fiecare clasa pentru XGBoost

x = np.arange(len(class_labels)) 
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting F1-Score (the best overall balance of precision/recall)
rects1 = ax.bar(x - width/2, rf_f1, width, label='Random Forest', color='#1f77b4')
rects2 = ax.bar(x + width/2, xgb_f1, width, label='XGBoost', color='#ff7f0e')

ax.set_ylabel('F1-Score (0.0 to 1.0)')
ax.set_title(f'Model Performance Comparison by Malware Class {dataset}')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()
ax.set_ylim(0, 1.1) # Extra room for labels

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"results/{dataset}/Model_Performance_Comparison_{dataset.split('.')[0]}.png")


#We plot log loss in comparison for both classes
log_loss_values = [rf_log_loss, xgb_log_loss]
models = ['Random Forest', 'XGBoost']
plt.figure(figsize=(8, 6))
plt.bar(models, log_loss_values, color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Log Loss (Lower is Better)')
plt.title(f'Model Log Loss Comparison {dataset}')
plt.savefig(f"results/{dataset}/Model_Log_Loss_Comparison_{dataset.split('.')[0]}.png")

