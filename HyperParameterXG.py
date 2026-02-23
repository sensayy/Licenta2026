import pandas as pd


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("ByteCount_Size_Labels.csv")


X = df.drop(columns=['Id', 'Class']) #Pregatim datele X fara ID si Clasa
y = df['Class'] #Label-urile

if y.min() == 1: #XGBoost foloseste clase DOAR incepand cu 0 deci modificam
    y = y - 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y) #Facem train test split, stratify face ca splitul de 80/20 sa fie EGAL pentru toate clasele

xgb = XGBClassifier(use_label_encoder = False, eval_metrics = 'mlogloss', random_state = 42, tree_method='hist', device='cuda')

param_grid={
    'learning_rate':[0.01,0.03,0.05,0.1,0.15,0.2],
     'n_estimators':[100,200,500,1000,2000],
     'max_depth':[3,5,10],
    'colsample_bytree':[0.1,0.3,0.5,1],
    'subsample':[0.1,0.3,0.5,1]
}

grid_search = GridSearchCV(estimator=xgb, 
                           param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=3, 
                           verbose=2,)

print("Starting Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("\nFinal Test Accuracy with Tuned Model:")
print(accuracy_score(y_test, predictions))