'''
run `python3 -m predictions.multi_stage_reg.predict_model`
'''
from preprocess import Log, invLog
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from preprocess import preprocess, preprocess_predict
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score

def run_regression(model):
    return

def run_classification(model):
    return

def test_regression():
    train_data_f = "../data/InsNova_data_2023_train.csv"
    train_data = pd.read_csv(train_data_f)

    # the log10 transformation is used on the claimscst response for training and prediction
    X_preprocessed, _, _, Y = preprocess(train_data, standardize=False, include=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y,
                                                        test_size=0.2, random_state=42)
    Y_train = Y_train.apply(Log)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, Y_test)
    print("r2 score1 = {}".format(score))
    predictions = np.power(10, predictions)
    score = r2_score(Y_test, predictions)
    print("r2 score2={}".format(score))



def main():
    train_data_f = "../data/InsNova_data_2023_train.csv"
    train_data = pd.read_csv(train_data_f)
    
    # data for training the classifier
    X_preprocessed1, Y1, _, _ = preprocess(train_data, standardize=False, include=False)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_preprocessed1, Y1, 
                                                            test_size=0.2, random_state=42)
    classifier = LogisticRegression(class_weight=None, solver="newton-cg")
    classifier.fit(X_train1, Y_train1)
    Y_test_pred1 = classifier.predict(X_test1)
    print(confusion_matrix(Y_test1, Y_test_pred1))

    score1 = classifier.score(X_test1, Y_test1)
    print("Classifier Accuracy: ", score1)

    # test the random forest classifier
    forest_classifier = RandomForestClassifier(n_estimators=300, 
                                               max_depth=None,
                                               max_features="log2", 
                                               criterion="gini")
    forest_classifier.fit(X_train1, Y_train1)
    Y_test_pred1b = forest_classifier.predict(X_test1)
    print(confusion_matrix(Y_test1, Y_test_pred1b))

    # test the decision tree classifier
    tree_classifier = DecisionTreeClassifier(criterion="entropy", 
                                             max_depth=None, 
                                             max_features=None, 
                                             min_samples_split=2,
                                             min_samples_leaf=1)
    tree_classifier.fit(X_train1, Y_train1)
    Y_test_pred1c = tree_classifier.predict(X_test1)
    print(confusion_matrix(Y_test1, Y_test_pred1c))
    

    # data for training the regressor
    X_preprocessed2, _, _, Y2 = preprocess(train_data, standardize=True, include=True)
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_preprocessed2, Y2, 
                                                            test_size=0.2, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train2, Y_train2)
    score2 = regressor.score(X_test2, Y_test2)
    print("R^2 Value: ", score2)

    test_regression()

    return 

if __name__ == "__main__":
    # main()
    test_regression()