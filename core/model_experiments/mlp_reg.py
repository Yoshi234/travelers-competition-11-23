'''
Please run from root package: core
`python3 -m model_experiments.mlp_reg`
'''

from sklearn.neural_network import MLPRegressor
from preprocess import preprocess
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    data_f = "../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_f)
    X_preprocessed, _, _, Y = preprocess(data)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_preprocessed, Y,
                                                        test_size=0.2, random_state=42)
if __name__ == "__main__":
    main()