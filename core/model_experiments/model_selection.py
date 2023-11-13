'''
Please run this file via the root package: core
`python3 -m model_experiments.model_selection`
'''

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from preprocess import preprocess
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    '''
    Functionality 
    - I only use the `claimcst0` variable as a response for now
    - the train test split utilizes 20% test, 80% training data splits
    - we test three different models here!
    - I use hyperparameter tuning laid out by Ken Jee in his video
    - then, I utilize 3 fold cross validation like Ken did
    '''
    data_f = "../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    X_preprocessed, _, _, Y = preprocess(data, standardize=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y, 
                                                        test_size=0.2, random_state=42)

    # set models to test / train
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    # set hyperparameters
    param_grids = {
        "LinearRegression": {},
        "RandomForest": {
            "n_estimators": [100, 200, 500], 
            "max_depth": [None, 10, 30],
            "min_samples_split": [2,5,10],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 500], 
            "learning_rate": [0.01, 0.1, 0.3], 
            "max_depth": [3, 6, 10],
        }
    }

    # 3-fold cross-validation
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # Train and tune the models
    grids = {}
    for model_name, model in models.items():
        grids[model_name] = GridSearchCV(estimator=model, param_grid=param_grids[model_name], 
                                         cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, 
                                         verbose=2)
        grids[model_name].fit(X_train, Y_train)
        best_params = grids[model_name].best_params_
        best_score = np.sqrt(-1 * grids[model_name].best_score_)

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best RMSE for {model_name}: {best_score}:\n")


if __name__ == "__main__":
    main()

# Model selection results: for standardized data
#       linear regression: RMSE = 1240.18411
#       Random Forest: RMSE = 1257.934
#       XGBoost: RMSE: 1242.335

# Model selection results: for non-standardized data
#       linear regression: RMSE = 1239.41
#       Random Forest: RMSE = 1257.906
#       XGBoost: RMSE = 1242.3497

# we are only concerned with the predicted claim cost, not the 
# claim