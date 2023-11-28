'''
Much of this code was borrowed from Ken Jee's kaggle notebook "Housing Prices Example (With 
Video Walkthrough)"

https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ 
Borrow an oversampling method to improve the model - SMOTE
'''
from sklearn.decomposition import PCA
import math
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

def preprocess_help(data, standardize=True):
    '''
    helper function for preprocessing data
    '''
    # we may or may not want to use the standard scaler!! - try both options
    standard_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
    ])
    norm_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])
    # if a value is null, then we just treat it as a new category of data
    # the onehot encoder just treats each category as a separate column, with 
    # n-1 columns (where there are n distinct categories)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

    # Remove target variable(s) from numerical and categorical columns
    # numerical_columns = numerical_columns.drop(["clm", "numclaims", "claimcst0"])
    # Combine transformers using ColumnTransformer
    num_trans = None
    if standardize: num_trans = norm_numerical_transformer
    else: num_trans = standard_numerical_transformer

    # allows you to choose which columns you adjust!
    # passthrough means 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_trans, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ], remainder = 'passthrough'
        # setting remainder="passthrough" automatically passes all columns not specified by 
        # the transformers through the transformation
    )
    
    return preprocessor

def invLog(x):
    return 10**x - 1

# use a flexible transformation on the data
def Log(x):
    return math.log10(x + 1)
# the inverse transformation should raise 10 to 
# the power of the elements in the array


def preprocess(data: pd.DataFrame, standardize=True, include=False):
    '''
    Functionality
    - run through all numerical valued columns + replace missing values in the
    data with a mean value
    - It may or may not make sense to use a standardization scheme, but I include
    it in the initial transformer
    '''
    # print(data.head())
    # apply the pipeline to the dataset of interest, axis=1 specifies columns should be dropped
    if include == False:
        X = data.drop(["id", "clm", "numclaims", "claimcst0"], axis=1)
        # print(X.head())
    elif include == True:
        X = data.drop(["id", "claimcst0", "numclaims"], axis=1)
        # print(X.head())
    Y1 = data["clm"]
    Y2 = data["numclaims"]
    Y3 = data["claimcst0"]
    # assert Y3.apply(Log)
    # transform the response variable using a log10 transformation
    # fit the preprocessor to the predictor variables
    preprocessor = preprocess_help(X, standardize)
    pipeline1 = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
    X_preprocessed = pipeline1.fit_transform(X)

    return X_preprocessed, Y1, Y2, Y3

def preprocess_predict(data: pd.DataFrame, standardize=True):
    # apply the pipeline to the dataset of interest, axis=1 specifies columns should be dropped
    preprocessor = preprocess_help(data, standardize)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
    # fit the preprocessor to the predictor variables
    X_preprocessed = pipeline.fit_transform(data)

    return X_preprocessed

def pca_transform(preprocessed_data, standard_data):
    '''
    Take as input preprocessed data, and feed back out the 
    numerical components with dimensionality reduced

    takes a preprocessor object used 
    '''
    pca = PCA()
    X_pca_pre = pca.fit_transform(preprocessed_data)

    # calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

    pca = PCA(n_components=n_components)
    preprocessor = preprocess_help(standard_data, True)
    pipeline_pca = Pipeline(steps=
                            [('preprocessor', preprocessor),
                             ('pca', pca)])
    X_pca = pipeline_pca.fit_transform(standard_data)
    return X_pca, n_components

def pca_transform_predict(standard_data, n_components):
    pca = PCA(n_components = n_components)
    preprocessor = preprocess_help(standard_data, True)
    pipeline_pca = Pipeline(steps=
                            [('preprocessor', preprocessor), 
                             ('pca', pca)])
    X_pca_predict = pipeline_pca.fit_transform(standard_data)
    return X_pca_predict

# try a hierarchical learning approach - hierarchical modeling!
def main():
    data_f = "../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    # call the main preprocessing function and that's it - should be imported 
    # from the main run file
    _, _, _, _ = preprocess(data)
    return None

if __name__ == "__main__":
    main()