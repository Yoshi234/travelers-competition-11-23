'''
run: `python3 -m predictions.naeve_regression.predict_model
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from preprocess import preprocess, preprocess_predict, Log, pca_transform, pca_transform_predict
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def Gini(a, p):
    return gini(a, p) / gini(a, a)

def run_predictions(model, num_components):
    '''
    run predictions for the given model

    model --- trained python ml model
    '''
    data_f = "../data/InsNova_data_2023_vh.csv"
    data = pd.read_csv(data_f)
    ids = data["id"]
    # X_predict = preprocess_predict(data, standardize=False)
    X_predict_pca = pca_transform_predict(data, num_components)
    # make predictions and apply correct transformations
    Y_predict = np.power(10, model.predict(X_predict_pca))-1
    # Y_predict = np.power(10, Y_predict)
    
    run_number = "_xgboost_with_pca"
    predictions_txt = "predictions/naeve_regression/prediction_results/predictions{}.txt".format(run_number)
    predictions_csv = "predictions/naeve_regression/prediction_results/predictions{}.csv".format(run_number)
    
    with open(predictions_txt, "w") as f:
        for i in range(len(X_predict_pca)):
            f.write("id: {} | prediction: {}\n".format(ids[i], Y_predict[i]))

    col1 = "id"
    col2 = "Predict"
    with open(predictions_csv, "w") as f:
        f.write("{},{}\n".format(col1, col2))
        for i in range(len(X_predict_pca)):
            f.write("{},{}\n".format(int(ids[i]), Y_predict[i]))

def main():
    '''
    run linear regression for the model
    '''
    data_f = "../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    X_preprocessed, _, _, Y = preprocess(data, standardize=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y, 
                                                        test_size=0.2, random_state=42)
    # model1 = RandomForestRegressor(max_depth=10, min_samples_split=5, n_estimators=500)
    # model1.fit(X_train, Y_train.apply(Log))
    # gini_score1 = Gini(model1.predict(X_test), Y_test.apply(Log))
    # print("Log regressor performance: {}".format(gini_score1))

    # model2 = RandomForestRegressor(max_depth=10, min_samples_split=5, n_estimators=500)
    # model2.fit(X_train, Y_train)
    # gini_score2 = Gini(model2.predict(X_test), Y_test)
    # print("Normal Regressor performance: {}".format(gini_score2))
    # fit log10 transform to the response variable

    # learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 100, 'scale_pos_weight': 1, 'subsample': 1.0 - alternate parameters
    model1 = XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.01, random_state=42)
    model1.fit(X_train, Y_train.apply(Log))
    gini_score1 = Gini(np.power(10, model1.predict(X_test))-1, Y_test)
    print("Log regressor performance: {}".format(gini_score1))
    # test performance: 0.06140 = Gini

    model2 = XGBRegressor(max_depth=6, n_estimators=200, learning_rate=0.01, random_state=42)
    model2.fit(X_train, Y_train)
    gini_score2 = Gini(model2.predict(X_test), Y_test)
    print("Normal regressor performance: {}".format(gini_score2))
    # test performance: 0.10990 = Gini

    # learning_rate': 0.001, 'max_depth': 6, 'n_estimators': 200, 'scale_pos_weight': 1, 'subsample': 0.6}

    model3 = XGBRegressor(learning_rate=0.001, max_depth=6, n_estimators=200, scale_pos_weight=1, subsample=0.6)
    X_preprocessed_pca, num_components = pca_transform(X_preprocessed, data)
    X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_preprocessed_pca, Y, 
                                                                        test_size=0.2, random_state=42)
    model3.fit(X_train_pca, Y_train_pca.apply(Log))
    gini_score3 = Gini(np.power(10, model3.predict(X_test_pca))-1, Y_test_pca)
    print("PCA Regressor performance: {}".format(gini_score3))
    # model = XGBRegressor(max_depth=6, n_estimators=200, learning_rate=0.01)
    # model.fit(X_preprocessed, Y)
    # model.fit(X_train, Y_train)
    # score = model.score(X_test, Y_test)
    run_predictions(model3, num_components=num_components)
    
    # print("R^2 score:", score)
    # gini_score = Gini(model.predict(X_test), Y_test)
    # run_predictions(model)
    # print("Gini Score:", gini_score)

if __name__ == "__main__":
    main()