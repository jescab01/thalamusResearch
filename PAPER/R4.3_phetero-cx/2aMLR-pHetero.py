'''

Using 30% of the sample to adjust a Multiple Regression model so
we can predict heterogeneous p based on coupling factor (g) and ROI indegree (relative to average).

The check with the 70% of computed sample

https://www.w3schools.com/python/python_ml_multiple_regression.asp
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

import plotly.graph_objects as go

data_folder = "E:\LCCN_Local\PycharmProjects\\thalamusResearch\PAPER\R4.3_phetero-cx\data\\"
sim_tag = "pHetero-SUBJECTs_m12d08y2022-t11h.19m.24s"

df = pd.read_pickle(data_folder + sim_tag + "/.1pHeteroTABLE-SUBJECTS.pkl")

df["degree_rel"] = df["g"] * (df["degree"] - df["degree_avg"])
df["degree_fromth_rel"] = df["g"] * (df["degree_fromth"] - df["degree_fromth_avg"])

# Split the sample
subj_ids = [35, 49, 50]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

# Use just 30% of the sample to fit the model: training sample
df_train = df.loc[(df["subject"].isin(subjects))]
df_test = df.loc[-(df["subject"].isin(subjects))]


## TRAINING the model
X = df_train[["degree_rel", "degree_fromth_rel"]]
y = df_train[["p_adjusted"]]

reg = LinearRegression()
reg.fit(X, y)

print("The linear model is: Y = %0.5f + g * %0.15f degree + %0.15f degree_fromth" %
      (reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]))


# Assess the relevance of factors
X = np.column_stack((df_train["degree_rel"], df_train["degree_fromth_rel"]))
y = df_train[["p_adjusted"]]

X2 = sm.add_constant(X)
estimation = sm.OLS(y, X2)
est_results = estimation.fit()
print(est_results.summary())


df_train["predicted_p"] = reg.predict(df_train[["degree_rel", "degree_fromth_rel"]])[:, 0]

# Plot fit between predictions and actual values in train
fig = go.Figure(go.Scatter(x=df_train["degree_rel"], y=df_train["predicted_p"], mode="markers", opacity=0.8))
fig.add_trace(go.Scatter(x=df_train["degree_rel"], y=df_train["p_adjusted"], mode="markers", opacity=0.7))
fig.show("browser")


## TESTING
df_test["predicted_p"] = reg.predict(df_test[["degree_rel", "degree_fromth_rel"]])[:, 0]

# Plot predictions and real: check fit for test group
fig = go.Figure(go.Scatter(x=df_test["degree"], y=df_test["predicted_p"], mode="markers", opacity=0.8))
fig.add_trace(go.Scatter(x=df_test["degree"], y=df_test["p_adjusted"], mode="markers", opacity=0.7))
fig.show("browser")

"""
Models:
(all) Y = 0.08981 + -0.000489696250515 degree_rel + -0.004589638421834 degree_fromth_rel
(g==2) Y = 0.08974 + -0.000655695291370 degree_rel + -0.006121249955016 degree_fromth_rel
(g==1) Y = 0.08987 + -0.000320391770089 degree_rel + -0.003053545016936 degree_fromth_rel
(g==0.5) Y = 0.08993 + -0.000158383940004 degree_rel + -0.001520751228527 degree_fromth_rel

Final model:
Y = 0.08981 + g * (-0.000332632750063 degree_rel + -0.003015538178012 degree_fromth_rel)

"""

