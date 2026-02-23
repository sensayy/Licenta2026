import pandas as pd
import cupy as cp

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV

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

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=50, 
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=1  # Keep at 1 when using GPU to avoid memory conflicts
)

print(f"Starting Randomized Search (Testing 50 of 1,440 combinations)...")
random_search.fit(X_train, y_train)

# 4. Results
print(f"\nBest Parameters found: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.4f}")


# 5. Evaluate the winner
best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test)

print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")