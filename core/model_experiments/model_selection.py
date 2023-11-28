'''
Please run this file via the root package: core
`python3 -m model_experiments.model_selection`
'''

from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from preprocess import preprocess, Gini, Log, pca_transform
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

# def gini(actual, pred, cmpcol = 0, sortcol = 1):
#     assert( len(actual) == len(pred) )
#     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
#     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
#     totalLosses = all[:,0].sum()
#     giniSum = all[:,0].cumsum().sum() / totalLosses
    
#     giniSum -= (len(actual) + 1) / 2.
#     return giniSum / len(actual)
 
# def Gini(a, p):
#     return gini(a, p) / gini(a, a)

def main():
    '''
    Functionality 
    - I only use the `claimcst0` variable as a response for now
    - the train test split utilizes 20% test, 80% training data splits
    - we test three different models here!
    - I use hyperparameter tuning laid out by Ken Jee in his video
    - then, I utilize 3 fold cross validation like Ken did
    '''
    train_data_f = "../data/InsNova_data_2023_train.csv"
    train_data = pd.read_csv(train_data_f)
    X_preprocessed, _, _, Y = preprocess(train_data, standardize=False)
    X_pca, _ = pca_transform(X_preprocessed, train_data)

    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_preprocessed, Y, 
                                                        test_size=0.2, random_state=42)
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_pca, Y, 
                                                           test_size=0.2, random_state=42)
    Y_train1.apply(Log)
    Y_train2.apply(Log)
    # set models to test / train
    model = {
        "XGBoost": XGBRegressor(random_state=42)
    }
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    param_gridb = {
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500], 
            "max_depth": [3, 6, 10, 12, 18, 20], 
            "learning_rate": [0.001, 0.01, 0.1, 0.5, 0.9, 0.95], 
            "subsample": [0.6, 1.0], 
            "scale_pos_weight": [1, 6.4095]
        }
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

    score = make_scorer(Gini, greater_is_better=True)
    # 3-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # print(40*"-")
    # print("fit for standard data")
    # # Train and tune the models
    # grids = {}
    # for model_name, model in model.items():
    #     grids[model_name] = GridSearchCV(estimator=model, param_grid=param_gridb[model_name], 
    #                                      cv=cv, scoring=score, n_jobs=-1, 
    #                                      verbose=2)
    #     grids[model_name].fit(X_train1, Y_train1)
    #     best_params = grids[model_name].best_params_
    #     best_score = grids[model_name].best_score_


    #     print(f"Best parameters for {model_name}: {best_params}")
    #     print(f"Best Gini for {model_name}: {best_score}:\n")

    pca_grids = {}
    print(40*"-")
    print("fit for pca transformed data")
    for model_name, model in model.items():
        pca_grids[model_name] = GridSearchCV(estimator=model, param_grid=param_gridb[model_name], 
                                         cv=cv, scoring=score, n_jobs=-1, 
                                         verbose=2)
        pca_grids[model_name].fit(X_train2, Y_train2)
        best_params = pca_grids[model_name].best_params_
        best_score = pca_grids[model_name].best_score_


        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best Gini for {model_name}: {best_score}:\n")


if __name__ == "__main__":
    main()

# Model selection results: for standardized data
#       linear regression: RMSE = 1240.18411
#       Random Forest: RMSE = 1257.934
#       XGBoost: RMSE: 1242.335
# linear regression will perform poorly if you feed in a ton of variables, 
# and many of them are very highly correlated to each other

# Model selection results: for non-standardized data
#       linear regression: Gini = 0.11
#       Random Forest: Gini = 0.13
#           best parameters: 
#               max_depth=10
#               min_samples_split=5
#               n_estimators=500
#           gini: 0.1342431
#       XGBoost: Gini: 0.17
#           best parameters:
#               max_depth = 6
#               n_estimators=200
#               learning_rate=0.01

# we are only concerned with the predicted claim cost, not the 
# claim

# when we apply the log transform to the training data, we get an improved 
# gini score metric for each of the models:
# for LinearRegression:
#       Gini = 0.256323
#       params = None
# for RandomForestRegression:
#       Gini = 0.26544
#       params
#           max_depth = 10
#           min_samples_split = 2
#           n_estimators = 500
# for XGBoost:
#       Gini = 0.308667
#       params
#           max_depth = 3
#           learning_rate = 0.01
#           n_estimators =