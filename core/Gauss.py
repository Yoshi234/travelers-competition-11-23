import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import f

T_set = pd.read_csv(r"C:\Users\jason\PYTHON\2023 Travelers Analytics Case Competition\InsNova_data_2023_train.csv")

T_numerical = T_set.loc[:, ["veh_value", "exposure", "veh_age",
                    "agecat", "max_power", "driving_history_score",
                   "e_bill", "trm_len", "credit_score", "high_education_ind",
                   "clm", "numclaims", "claimcst0"]]

T_non_zero = T_numerical.iloc[(T_numerical["claimcst0"] != 0).values]

response_variable = ["veh_value", "exposure", "veh_age", "agecat", "max_power",
                     "driving_history_score", "e_bill", "trm_len", "credit_score",
                     "high_education_ind"]

T_x = T_numerical[["veh_value", "exposure", "veh_age",
                    "agecat", "max_power", "driving_history_score",
                   "e_bill", "trm_len", "credit_score", "high_education_ind"]]
T_y = T_numerical["claimcst0"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(T_x, T_y, test_size=0.3)


from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))

gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp.fit(X_train, y_train)

y_pred, sigma = gp.predict(X_test, return_std=True)