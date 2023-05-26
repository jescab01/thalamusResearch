
import scipy.io
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import pingouin as pg


data_folder = "E:\LCCN_Local\PycharmProjects\\thalamusResearch\CHECKS\SCmaxlength\\"


## 0. Prepare dataframe
df = pd.DataFrame()
for measure in ["WEIGHTS", "LENGTHS"]:
    for maxlength in ["180", "250", "300"]:
        matrix = scipy.io.loadmat(data_folder + measure + "_whole_brain_AAL2_"+ maxlength + "mm.mat")["connectivity"]
        matrix = matrix[np.triu_indices(len(matrix), 1)]
        # matrix = matrix[matrix != 0]

        temp = pd.DataFrame([[measure] * len(matrix), [maxlength] * len(matrix), matrix]).transpose()
        df = pd.concat((df, temp))

df.columns = ["measure", "maxlength", "value"]
df = df.astype({"measure":str, "maxlength":str, "value":float})

## 1. WEIGHTS
# 1.1 Violin plot
subset = df.loc[df["measure"] == "WEIGHTS"]

fig = px.violin(subset, x="maxlength", y="value", color="maxlength", points="all", color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_traces(meanline_visible=True)
fig.update_layout(template="plotly_white", yaxis1=dict(title="Number of streamlines"), xaxis=dict(title="Max. length (mm)"), height=700, width=600)
pio.write_html(fig, data_folder + "Weights_maxlength_comparison.html", auto_open=True)

# 1.2 Stats - [F(2, 21417)=0.0387, p=0.96, eta2p=0.000004]
aov = pg.anova(data=subset, dv="value", between="maxlength")
aov


## 2. Tract Lengths
# 2.1 Violin plot for lengths
subset = df.loc[(df["measure"] == "LENGTHS") & (df["value"] != 0)]

fig = px.violin(subset, x="maxlength", y="value", color="maxlength", points="all", color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_traces(meanline_visible=True)
fig.update_layout(template="plotly_white", yaxis1=dict(title="Tract length average (mm)"), xaxis=dict(title="Max. length (mm)"), height=700, width=600)
pio.write_html(fig, data_folder + "Lengths_maxlength_comparison.html", auto_open=True)

# 2.2 STATS - [F(2, 8316)=24.93, p=1.6e-11, n2p=0.00596]
aov = pg.anova(data=subset, dv="value", between="maxlength")
aov

