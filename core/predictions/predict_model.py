from sklearn.linear_model import LinearRegression
from preprocess import preprocess, preprocess_predict
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np

def run_predictions(model):
    '''
    run predictions for the given model

    model --- trained python ml model
    '''
    data_f = "../data/InsNova_data_2023_vh.csv"
    data = pd.read_csv(data_f)
    X_predict = preprocess_predict(data, standardize=False)
    Y_predict = model.predict(X_predict)
    
    run_number = 0
    predictions_txt = "predictions/predictions{}.txt".format(run_number)
    predictions_csv = "predictions/predictions{}.csv".format(run_number)
    
    with open(predictions_txt, "w") as f:
        for i in range(len(X_predict)):
            f.write("id: {} | prediction: {}\n".format(X_predict[i][0], Y_predict[i]))

    col1 = "id"
    col2 = "Predict"
    with open(predictions_csv, "w") as f:
        f.write("{},{}\n".format(col1, col2))
        for i in range(len(X_predict)):
            f.write("{},{}\n".format(int(X_predict[i][0]), Y_predict[i]))
    
        


def main():
    '''
    run linear regression for the model
    '''
    data_f = "../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    X_preprocessed, _, _, Y = preprocess(data, standardize=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y, 
                                                        test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    
    print("R^2 score:", score)
    run_predictions(model)


if __name__ == "__main__":
    main()
    