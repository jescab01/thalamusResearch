
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import numpy as np

simulations_tag = "PSEmpi_adjustrange_P-m12d20y2022-t02h.17m.01s"
folder = 'PAPER2\\R4.1_padjust\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

df_avg_p = df.groupby(["subject", "model", "th", "cer", "g", "pth", "sigmath"]).mean().reset_index()


## Load simulations from sigma adjust
simulations_tag = "PSEmpi_adjustrange_sigma-m12d20y2022-t04h.08m.02s"
folder = 'PAPER2\\R4.2_sigmaadjust\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

df_avg_s = df.groupby(["subject", "model", "th", "cer", "g", "pth", "sigmath"]).mean().reset_index()

subject="NEMOS_035"
folder = 'PAPER2\\R4.3_p&sigma\\'

## FIG 4.1b - FC specifics
fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.07, subplot_titles=["PLV mean", "PLV std", "PLV mean", "PLV std"],
                    x_title=r'$\text{Mean input to thalamus } (p_{th} \text{)                           Gaussian std. of thalamic input } (\eta_{th})$',
                    y_title=r"$\text{Coupling factor (g)}$",
                    shared_yaxes=True, shared_xaxes=True)

subset = df_avg_p.loc[(df_avg_p["subject"] == subject) & (df_avg_p["th"] == "pTh")]

fig.add_trace(go.Heatmap(z=subset.plv_m, x=subset.pth, y=subset.g, colorscale='Turbo',
                         showscale=True, colorbar=dict(title="", thickness=4, x=0.19)), row=1, col=1)

fig.add_trace(go.Heatmap(z=subset.plv_sd, x=subset.pth, y=subset.g, colorscale='Turbo',
                         showscale=True, colorbar=dict(title="", thickness=4, x=0.46)), row=1, col=2)

subset = df_avg_s.loc[(df_avg_s["subject"] == subject) & (df_avg_s["th"] == "pTh")]
fig.add_trace(go.Heatmap(z=subset.plv_m, x=subset.sigmath, y=subset.g, colorscale='Turbo',
                         showscale=True, colorbar=dict(title="", thickness=4, x=0.73)), row=1, col=3)

fig.add_trace(go.Heatmap(z=subset.plv_sd, x=subset.sigmath, y=subset.g, colorscale='Turbo',
                         showscale=True, colorbar=dict(title="", thickness=4, x=1)), row=1, col=4)
fig.add_vline(x=0.05, col=[3,4], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)
fig.add_vline(x=0.15, col=[3,4], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)

fig.update_layout(height=400, width=800, xaxis3=dict(type="log"), xaxis4=dict(type="log"))

pio.write_html(fig, file=folder + "/PAPER4bsm-padjust.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER4bsm-padjust.svg", engine="kaleido")


