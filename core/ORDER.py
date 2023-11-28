#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[5]:


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import math


# # Functions

# In[35]:


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


# In[12]:


def Log(x):
    if x == 0:
        return 0
    else:
        return math.log10(x)


# In[16]:


def preprocess_includes_numclaims(data: pd.DataFrame, standardize=True):
    
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
    numerical_columns = numerical_columns.drop(["claimcst0"])

    
    
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
    X = data.drop(["claimcst0"], axis=1)
    
    # NAT LOG OF THIS BELOW
    
    
    Y = data["claimcst0"].apply(Log)
    # fit the preprocessor to the predictor variables
    X_preprocessed = pipeline.fit_transform(X)

    return X_preprocessed, Y


# In[17]:


def preprocess_not_includes_numclaims(data: pd.DataFrame, standardize=True):
    
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
    numerical_columns = numerical_columns.drop(["numclaims", "claimcst0"])

    
    
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
    X = data.drop(["numclaims", "claimcst0"], axis=1)
    
    # SAME HERE
    
    Y = data["claimcst0"].apply(Log)
    # fit the preprocessor to the predictor variables
    X_preprocessed = pipeline.fit_transform(X)

    return X_preprocessed, Y


# In[18]:


# try a hierarchical learning approach - hierarchical modeling!
def main():
    data_f = "InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    # call the main preprocessing function and that's it - should be imported 
    # from the main run file
    x_includes, y = preprocess_includes_numclaims(data)
    x_not_in, y_n = preprocess_not_includes_numclaims(data)
    return x_includes, x_not_in, y
    
if __name__ == "__main__":
    main()


# # Datasets, Training and Testing

# In[19]:


x_includes, x_not_in, y = main()


# In[20]:


x_i_train, x_i_test, y_i_train, y_i_test = train_test_split(x_includes, y,
                                                        random_state=104,
                                                        test_size = 0.3,
                                                        shuffle = True)


# In[21]:


x_n_train, x_n_test, y_n_train, y_n_test = train_test_split(x_not_in, y,
                                                        random_state=104,
                                                        test_size = 0.3,
                                                        shuffle = True)


# # ML Regression (to check GINI)

# In[22]:


mlr_includes = LinearRegression()

mlr_includes.fit(x_i_train, y_i_train)


# In[23]:


mlr_not_in = LinearRegression()

mlr_not_in.fit(x_n_train, y_n_train)


# In[24]:


y_i_p_mlr = mlr_includes.predict(x_i_test)
y_i_p_mlr


# In[25]:


y_n_p_mlr = mlr_not_in.predict(x_n_test)
y_n_p_mlr


# In[33]:


mse(y_i_p_mlr, y_i_test)


# In[34]:


mse(y_n_p_mlr, y_n_test)


# In[36]:


Gini(y_i_p_mlr, y_i_test)


# In[37]:


Gini(y_n_p_mlr, y_n_test)

