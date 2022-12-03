
'''
dos cosas se deben decir de este plot:

- cuanto mas noise, no hay necesariamente mejor correlacion
- hay un signal to noise optimo, en el cual alta rPLV y todavía se tiene en 10Hz

'''

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio


simulations_tag = "PSEmpi_adjustingrange_Noise_v2-m10d05y2022-t22h.17m.20s"
folder = 'PAPER\\R4.2_sigmaadjust\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

df_avg_ = df.loc[df["sigma"] != 0].groupby(["subject", "model", "th", "cer", "g", "sigma"]).mean().reset_index()
df_base = df.loc[(df["sigma"] == 0)].groupby(["subject", "model", "th", "cer", "g", "sigma"]).mean().reset_index()

# Calculate signal/noise ratio
df_avg_["s2n"] = np.nan  # Preallocating

df_avg = df_avg_.copy()
for g in set(df_avg["g"].values):
    for th in ["Th", "pTh"]:
        # For th -
        df_avg["s2n"].loc[(df_avg["th"]==th) & (df_avg["g"]==g)] = \
            df_base["std_th"].loc[(df_base["th"]==th) & (df_base["g"]==g)].values[0]/\
            df_avg["sigma"].loc[(df_avg["th"]==th) & (df_avg["g"]==g)].values

df_avg["log_s2n"] = np.log10(df_avg["s2n"].values)

# df_avg["s2n_"] = np.log(df_avg["std_th"]/df_avg["sigma"])


structure_th = ["woTh", "Th",  "pTh"]

x_type = "log"

for subject in list(set(df_avg.subject)):

    fig = make_subplots(rows=4, cols=3, column_titles=["woTh", "Th", "pTh"],
                        row_titles=["$r_{PLV}$", "bifurcations (cx)", "thalamus log(s/n)", "FFT peak"],
                        x_title="Gaussian noise into thalamus (std)", y_title="Coupling factor (g)",
                        shared_yaxes=True, shared_xaxes=True)

    for j, th in enumerate(structure_th):
        subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == th)]

        sl = True if j == 0 else False

        fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.sigma, y=subset.g, colorscale='RdBu', reversescale=True,
                       zmin=-0.5, zmax=0.5, showscale=sl, colorbar=dict(thickness=7, len=0.2, y=0.9, x=1.05)), row=1, col=(1 + j))

        fig.add_trace(go.Heatmap(z=subset.max_cx - subset.min_cx, x=subset.sigma, y=subset.g, colorscale='Viridis',
                       showscale=sl, colorbar=dict(thickness=7, len=0.2, y=0.64, x=1.05, title="mV")), row=2, col=(1 + j))

        fig.add_trace(go.Heatmap(z=subset.log_s2n, x=subset.sigma, y=subset.g, colorscale='Geyser', reversescale=True,
                       showscale=sl, colorbar=dict(thickness=7, len=0.2, y=0.37, x=1.05, title="log(s/n)"), zmin=-4, zmax=4), row=3, col=(1 + j))

        fig.add_trace(go.Heatmap(z=subset.IAF, x=subset.sigma, y=subset.g, colorscale='Turbo',
                       showscale=sl, colorbar=dict(thickness=7, len=0.2, y=0.09, x=1.05, title="Hz")), row=4, col=(1 + j))


    fig.add_vline(x=0.05, col=[2, 3], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
    fig.add_vline(x=0.3, col=[2, 3], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)

    fig.update_layout(xaxis1=dict(type=x_type), xaxis2=dict(type=x_type), xaxis3=dict(type=x_type), xaxis4=dict(type=x_type),
                      xaxis5=dict(type=x_type), xaxis6=dict(type=x_type), xaxis7=dict(type=x_type), xaxis8=dict(type=x_type), xaxis9=dict(type=x_type),
                      xaxis10=dict(type=x_type, tickangle=45), xaxis11=dict(type=x_type, tickangle=45), xaxis12=dict(type=x_type, tickangle=45),
                      template="plotly_white", width=500, height=900)

    pio.write_html(fig, file=folder + "/PAPER8sm-sigmaadjust.html", auto_open=True, include_mathjax="cdn")
    pio.write_image(fig, file=folder + "/PAPER8sm-sigmaadjust.svg", engine="kaleido")




