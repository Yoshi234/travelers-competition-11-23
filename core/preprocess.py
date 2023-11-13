'''
Much of this code was borrowed from Ken Jee's kaggle notebook "Housing Prices Example (With 
Video Walkthrough)"
'''
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess(data: pd.DataFrame, standardize=True):
    '''
    Functionality
    - run through all numerical valued columns + replace missing values in the
    data with a mean value
    - It may or may not make sense to use a standardization scheme, but I include
    it in the initial transformer
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
    numerical_columns = numerical_columns.drop(["clm", "numclaims", "claimcst0"])

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

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    # apply the pipeline to the dataset of interest, axis=1 specifies columns should be dropped
    X = data.drop(["clm", "numclaims", "claimcst0"], axis=1)
    Y1 = data["clm"]
    Y2 = data["numclaims"]
    Y3 = data["claimcst0"]
    # fit the preprocessor to the predictor variables
    X_preprocessed = pipeline.fit_transform(X)

    return X_preprocessed, Y1, Y2, Y3

def preprocess_predict(data: pd.DataFrame, standardize=True):
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

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    # apply the pipeline to the dataset of interest, axis=1 specifies columns should be dropped
    X = data
    # fit the preprocessor to the predictor variables
    X_preprocessed = pipeline.fit_transform(X)

    return X_preprocessed

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