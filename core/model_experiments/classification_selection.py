'''
Please run this file via the root package: core

Borrows hyperparameter settings from paper linked below:
https://arxiv.org/pdf/2204.06109.pdf
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocess import preprocess
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    '''
    test which model is optimal for predicting whether or not a claim 
    occurred for a policy
    '''
    data_f = "../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    X_preprocessed, Y, _, _ = preprocess(data, standardize=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y, 
                                                        test_size=0.2, random_state=42)
    print(Y_train.shape)
    cont = input("would you like to proceed? Enter 'no' if not: ")
    if cont == "no":
        return 
    
    models = {
        "LogisticRegression": LogisticRegression(), 
        "RandomForest": RandomForestClassifier(random_state=42), 
        "XGBoost": XGBClassifier(random_state=42)
    }
    # set hyperparameters
    param_grids = {
        "LogisticRegression": {
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], 
            "class_weight": ["balanced", None]
        }, 
        "RandomForest": {
            "criterion": ["gini", "entropy"], 
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 15, 25, None], 
            "max_features": ["auto", "log2", None], 
            "class_weight": ["balanced", "balanced_subsample", None]
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300], 
            "max_depth": [6, 12, 18, 20], 
            "learning_rate": [0.1, 0.5, 0.9, 0.95], 
            "subsample": [0.6, 1.0], 
            "scale_pos_weight": [1, 6.4095]
        }
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # train and tune the models: use accuracy metric for binary classification
    grids = {}
    for model_name, model in models.items():
        grids[model_name] = GridSearchCV(estimator=model, param_grid=param_grids[model_name], 
                                         cv=cv, scoring="accuracy", verbose=2)
        grids[model_name].fit(X_train, Y_train)
        best_params = grids[model_name].best_params_
        best_score = grids[model_name].best_score_

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best accuracy for {model_name}: {best_score}:\n")

    return None

if __name__ == "__main__":
    main()