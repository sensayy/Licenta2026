import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

import matplotlib.pyplot as plt
import numpy as np


#df = pd.read_csv("ByteCount_Size_Labels.csv")
df = pd.read_csv("Full_Features_v1.csv")

X = df.drop(columns=['Id', 'Class']) #Pregatim datele X fara ID si Clasa
y = df['Class'] #Label-urile

if y.min() == 1: #XGBoost foloseste clase DOAR incepand cu 0 deci modificam
    y = y - 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y) #Facem train test split, stratify face ca splitul de 80/20 sa fie EGAL pentru toate clasele

# --- RANDOM FOREST ---
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) #initializam
rf_model.fit(X_train, y_train) #antrenam
rf_preds = rf_model.predict(X_test) # testam

# --- XGBOOST ---
print("Training XGBoost...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') #label_encoder -> ii spunem ca deja avem label-urile separate;  mlogloss -> Multi-class Logarithmic Loss
xgb_model.fit(X_train, y_train) #antrenam
xgb_preds = xgb_model.predict(X_test) #testam



####################### PENTRU XGBOOST
importance = xgb_model.feature_importances_ #luam valorile cele mai importante pentru decizii
feature_names = X.columns #Luam numele coloanelor

indices = np.argsort(importance)[-20:] #sortam dupa importanta, primele 20 cele mai importante din cele 257. argsort sorteaza de la mic la mare deci luam cu - ca sa fie ultimele 20 cele mai importante
## np.sargsort practic scoate doar pozitiile nu si valorile, nu sorteaza in sine matricea ci doar scoate indicii in ordine. Sorting the arguments


plt.figure(figsize=(10, 8))
plt.title('Top 20 Most Important Bytes (XGBoost)')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
###########################


##########################Pentru Random Forest Classifier
importance = rf_model.feature_importances_ #luam valorile cele mai importante pentru decizii
feature_names = X.columns #Luam numele coloanelor

indices = np.argsort(importance)[-20:] #sortam dupa importanta, primele 20 cele mai importante din cele 257.


plt.figure(figsize=(10, 8))
plt.title('Top 20 Most Important Bytes (Random Forest)')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#############################

###########################Confusion Matrix XGBOOST
cm1 = confusion_matrix(y_test, xgb_preds) #se creeaza confusion matrix

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix: Actual vs. Predicted Malware Class XGBOOST')
plt.show()
#######################################


##########################Confusion Matrix Random Forest Classifier
cm2 = confusion_matrix(y_test, rf_preds) #se creeaza confusion matrix

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=range(1, 10)) 
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix: Actual vs. Predicted Malware Class Random Forest')
plt.show()

####################################

print("\n--- Random Forest Accuracy ---")
print(accuracy_score(y_test, rf_preds))

print("\n--- XGBoost Accuracy ---")
print(accuracy_score(y_test, xgb_preds))

print("\nDetailed XGBoost Report:")
print(classification_report(y_test, xgb_preds))

print("\nDetailed Random Forest Report:")
print(classification_report(y_test, rf_preds))


######################################### Grafice GEMINI
def get_metrics(y_true, y_pred):
    # Returns precision, recall, and f1 for each class
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return p, r, f1

# 1. Get stats for both models
# (Note: +1 to class_labels so graph shows 1-9 instead of 0-8)
class_labels = np.unique(y_test) + 1
rf_p, rf_r, rf_f1 = get_metrics(y_test, rf_preds)
xgb_p, xgb_r, xgb_f1 = get_metrics(y_test, xgb_preds)

# 2. Set up the bar chart
x = np.arange(len(class_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

# Plotting F1-Score (the best overall balance of precision/recall)
rects1 = ax.bar(x - width/2, rf_f1, width, label='Random Forest', color='#1f77b4')
rects2 = ax.bar(x + width/2, xgb_f1, width, label='XGBoost', color='#ff7f0e')

# 3. Labeling
ax.set_ylabel('F1-Score (0.0 to 1.0)')
ax.set_title('Model Performance Comparison by Malware Class')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()
ax.set_ylim(0, 1.1) # Extra room for labels

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 4. Print a quick Accuracy comparison
print(f"Overall RF Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"Overall XGB Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
##################### Grafice GEMINI