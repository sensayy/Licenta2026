import pandas as pd
import os
os.makedirs("tuning_results", exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import time

import pickle as pk
import json
import joblib

import optuna # For hyperparameter optimization, called Bayesian optimization, which is more efficient than Random Search for large parameter spaces
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, log_loss

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost") # Suppress warnings for cleaner output

start_time = time.time()


def get_metrics(y_true, y_pred):
    # Returns precision, recall, and f1 for each class
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return p, r, f1

def save_results(dataset, model_name, accuracy, classification_report, log_loss_value):
    with open(f"tuning_results/{dataset}/Results_{dataset}.txt",encoding="utf-8", mode="a") as f:
        f.write("\n"+"="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Log Loss: {log_loss_value}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report)
    print(f"Results of {model_name} saved to tuning_results/{dataset}/Results_{dataset.split('.')[0]}.txt")


df = pd.read_csv("csv/Full_Features_vocab2.csv")
dataset = "Full_Features_vocab2.csv"
print("Using dataset: Full_Features_vocab2.csv")

os.makedirs(f"tuning_results/{dataset}", exist_ok=True)

X = df.drop(columns = ["Id", "Class"]).astype(np.float32) #XGBoost functioneaza mai bine cu float32 decat cu float64, deci convertim toate valorile la float32 pentru a imbunatatii performanta
y = df["Class"]

if y.min() == 1: #XGBoost foloseste clase DOAR incepand cu 0 deci modificam
    y = y - 1

#Data split with stratify for balanced class distribution in train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y) #Facem train test split, stratify face ca splitul de 80/20 sa fie EGAL pentru toate clasele

def objective(trial): #In order to use Optuna, we need to define an objective function that takes a trial as input and returns the metric we want to optimize (in this case, accuracy)
    # 1. Define the parameter space
    param = {
        'device': 'cuda',
        'tree_method': 'hist',
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        # Tuning these:
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), #.suggest_int defines an integer parameter, with a given name and range. Optuna will sample values for this parameter from the specified range during optimization.
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), #.suggest_float defines a float parameter, with a given name and range. The log=True argument means that Optuna will sample values on a logarithmic scale, which is useful for parameters like learning_rate that can vary over several orders of magnitude.
        'max_depth': trial.suggest_int('max_depth', 3, 10), #.suggest_int defines an integer parameter that samples withing the range. Max_depth controls the maximum depth of the trees in the ensemble. Deeper trees can capture more complex patterns but are more prone to overfitting, while shallower trees are simpler and less likely to overfit but may underfit if too shallow.
        'subsample': trial.suggest_float('subsample', 0.1, 1.0), #subsample controls the fraction of the training data that is used to grow each tree. Setting subsample to a value less than 1.0 can help prevent overfitting by introducing randomness into the training process, similar to the concept of bagging in random forests.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0), #colsample_bytree controls the fraction of features that are randomly sampled for each tree. Setting colsample_bytree to a value less than 1.0 can help prevent overfitting by introducing randomness into the feature selection process, similar to the concept of feature bagging in random forests.
    }

    # 2. Train with current trial parameters
    model = XGBClassifier(**param)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"Trial {trial.number}: Accuracy = {cv_scores.mean():.4f} with params: {param}")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return np.mean(cv_scores) # Return the mean accuracy across the 3 folds as the metric to optimize



print("Starting Optuna Optimization...")
print(f"Time since init is: {time.time() - start_time:.2f} seconds")

study_name = "malware_xgb_optimization"
storage_name = f"sqlite:///tuning_results/optuna_study.db"

study = optuna.create_study(
    study_name=study_name, 
    storage=storage_name, 
    direction='maximize', 
    load_if_exists=True
)

# Now, if you ran 50 trials before, this will start at trial #51
study.optimize(objective, n_trials=1)


print("\nOptimization Finished!")
print(f"Best Accuracy: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")

best_params = study.best_params
best_params['device'] = 'cuda'
best_params['tree_method'] = 'hist'

final_model = XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

final_preds = final_model.predict(X_test)
final_probs = final_model.predict_proba(X_test)

acc = accuracy_score(y_test, final_preds)
l_loss = log_loss(y_test, final_probs)
report = classification_report(y_test, final_preds)
save_results(dataset, "XGBoost_Optuna_Tuned", acc, report, l_loss)
#We save the model


importance = final_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importance)[-20:]

plt.figure(figsize=(10, 8))
plt.title(f'Top 20 Most Important Features (Tuned XGBoost)')
plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"tuning_results/XGBoost_Feature_Importance.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, final_preds, 
    display_labels=range(1, 10), 
    cmap='Blues', 
    ax=ax
)
plt.title(f'Confusion Matrix: Tuned XGBoost')
plt.savefig(f"tuning_results/XGBoost_Confusion_Matrix.png")
plt.close()


#saving the model with pickle
pk.dump(final_model, open(f"tuning_results/xgb_malware_model.pkl", "wb"))

final_model.save_model(f"tuning_results/xgb_malware_model.json")
joblib.dump(final_model, f"tuning_results/xgb_model_joblib.pkl")

with open(f"tuning_results/best_params.json", 'w') as f:
    json.dump(study.best_params, f, indent=4)