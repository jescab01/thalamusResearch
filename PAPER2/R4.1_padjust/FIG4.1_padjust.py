
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import numpy as np

simulations_tag = "PSEmpi_adjustrange_P-m12d20y2022-t02h.17m.01s"
folder = 'PAPER2\\R4.1_padjust\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

df_avg = df.groupby(["subject", "model", "th", "cer", "g", "pth", "sigmath"]).mean().reset_index()

## Calculate snr
df_avg["snr"] = np.log((df_avg.max_th - df_avg.min_th)/df_avg.sigmath)

subject = "NEMOS_035"
# for i, subject in enumerate(list(set(df_avg.subject))):

## FIG 4.1a - main
fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.075, subplot_titles=[r"$r_{PLV}$", "FFT peak", "SNR (th)", "Bifurcations (cx)"],
                    x_title=r'$\text{Mean input to thalamus } (p_{th})$', y_title=r"$\text{Coupling factor (g)}$",
                    shared_yaxes=True, shared_xaxes=True)

subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == "pTh")]

fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.pth, y=subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
                         showscale=True, colorbar=dict(title="r", thickness=4,  x=0.19)), row=1, col=1)

fig.add_trace(go.Heatmap(z=subset.IAF, x=subset.pth, y=subset.g, colorscale='Turbo',
               showscale=True, colorbar=dict(title="Hz", thickness=4, x=0.46)), row=1, col=2)


fig.add_trace(go.Heatmap(z=subset.snr, x=subset.pth, y=subset.g, colorscale='Geyser',
               showscale=True, colorbar=dict(title="log<br>(snr)", thickness=4,  x=0.73)), row=1, col=3)

fig.add_trace(go.Heatmap(z=subset.max_cx - subset.min_cx, x=subset.pth, y=subset.g, colorscale='Viridis',
               showscale=True, zmin=0, zmax=0.14, colorbar=dict(title="mV", thickness=4,  x=1)), row=1, col=4)

fig.update_layout(width=800, height=400)

pio.write_html(fig, file=folder + "/PAPER4sm-padjust.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER4sm-padjust.svg", engine="kaleido")

