# Initial Packages

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import f



# Datasets

T_set = pd.read_csv(r"C:\Users\jason\PYTHON\2023 Travelers Analytics Case Competition\InsNova_data_2023_train.csv")

T_n = T_set.loc[:, ["veh_value", "exposure", "veh_age",
                    "agecat", "max_power", "driving_history_score",
                   "e_bill", "trm_len", "credit_score", "high_education_ind",
                   "clm", "numclaims", "claimcst0"]]

T_n = T_n.iloc[(T_n["claimcst0"] != 0).values]



# Lists of Response Variables

response_variable = ["veh_value", "exposure", "veh_age", "agecat", "max_power",
                     "driving_history_score", "e_bill", "trm_len", "credit_score",
                     "high_education_ind"]

from itertools import combinations
 
poss_reg = []

for n in range(1,11):
    r_c = list(combinations(response_variable, n))
    
    for r in r_c:
        poss_reg.append(list(r))



# Finding the Best Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

best_p_val = 1
best_p_r = []
T_y = T_n["claimcst0"]
n = len(T_y)

for p_r in poss_reg:
    T_x = T_n[p_r]
        
    the_reg_object = LinearRegression().fit(T_x, T_y)
    
    Rsq = the_reg_object.score(T_x, T_y)
    k = len(p_r)
    
    F_score = (Rsq / k) / ((1 - Rsq) / (n - k - 1))
    
    p_val = 1 - f.cdf(F_score, k, n-k-1)
    
    if p_val <= best_p_val:
        best_p_val = p_val
        best_p_r = p_r
        


# Returning and Graphing the Best Regression

print(f"BEST Regression: {best_p_r}")
print(f"\tP-Value: {best_p_val}")

reg = LinearRegression().fit(T_n[best_p_r], T_y)

plt.scatter(reg.predict(T_n[best_p_r]), reg.predict(T_n[best_p_r]) - T_y, s=10)
plt.title("Residual errors")
plt.show()