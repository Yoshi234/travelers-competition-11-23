'''
Credit: Borrowed heavily from Ken Jee's Housing 
Price Prediction walkthrough - Kaggle 

Please run from root package: core
`python3 -m model_experiments.mlp_reg`
'''

from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from preprocess import preprocess, Gini
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV

def main():
    data_f = "../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    X_preprocessed, _, _, Y = preprocess(data)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y,
                                                        test_size=0.2, random_state=42)
    mlp = MLPRegressor(random_state=42, max_iter=10000, n_iter_no_change=3, learning_rate=0.001)
 
    # define the prameter grid for tuning
    param_grid = {
        "hidden_layer_sizes": [(10,), (10,10), (10,10,10),
                               (25)],
        "activation": ["relu", "tanh"], 
        "alpha": [0.0001, 0.001, 0.01, .1, 1], 
        "learning_rate": ["constant", "invscaling", "adaptive"]
    }

    score = make_scorer(Gini, greater_is_better=True)
    # create the GridSearchCV object
    grid_search_mlp = GridSearchCV(mlp, param_grid, scoring=score, 
                                   cv=3, n_jobs=-1, verbose=1)
    
    # Fit the model on the training data
    grid_search_mlp.fit(X_train, Y_train)

    print("Best parameters found: ", grid_search_mlp.best_params_)

    # Evaluate the model on the test data
    best_score = grid_search_mlp.best_score_
    print("Test score: ", best_score)

    
if __name__ == "__main__":
    main()

    # tested an MLP regressor model
    # performance: RMSE = 1240.2290 (worse than linear regression)
    # parameters
    #       activation: "tanh"
    #       alpha: 0.001
    #       hidden_layer_sizes: (10, 10, 10) -> three layers of size 10
    #       learning_rate: "constant"