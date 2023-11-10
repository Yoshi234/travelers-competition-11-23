"""
PCA for InsNova dataset

Follows the geeks for geeks guide linked here:
https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/#
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def setup(data):
    '''
    reformat the dataframe to include only continuous variables
    '''
    data = data.loc[:, ["veh_value",
                        "exposure",
                        "veh_age",
                        "max_power",
                        "driving_history_score",
                        "e_bill",
                        "credit_score"]]
    return data

def visualize2D(data, p_comps, fig_name):
    '''
    visualize the principal components in 2 dimensions
    
    Arguments;
    - data --- pandas dataframe with the response variable included
    - p_comps --- the principal components (ndarray object)
    '''
    plt.figure(figsize=(10,10))
    plt.scatter(p_comps[:,0], p_comps[:,1], c=data["clm"],cmap="plasma")
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.savefig("figures/{}".format(fig_name))
    
def visualize3D(data, p_comps, fig_name):
    '''
    visualize the principal components in 3 dimensions

    Arguments:
    - data --- pandas dataframe with response variable included
    - p_comps --- principal components (ndarray object)
    '''
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(p_comps[:,0], p_comps[:,1], p_comps[:,2], c=data["clm"], cmap="plasma")
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)
    plt.savefig("figures/{}".format(fig_name))

def main():
    data_path = "../../data/InsNova_data_2023_train.csv"
    data = pd.read_csv(data_path)

    # select only continous valued variables / features
    cont_data = setup(data)

    print(cont_data.shape)

    scaling=StandardScaler()
    scaling.fit(cont_data)
    Scaled_data = scaling.transform(cont_data)

    principal = PCA(n_components=3)
    principal.fit(Scaled_data)
    x=principal.transform(Scaled_data)

    print(x.shape)

    # visualize2D(data, x, "2d-principal-components.png")
    visualize3D(data, x, "3d-principal-components.png")

if __name__ == "__main__":
    main()