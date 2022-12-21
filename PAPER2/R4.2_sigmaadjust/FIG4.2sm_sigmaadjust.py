
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import numpy as np

simulations_tag = "PSEmpi_adjustrange_sigma-m12d20y2022-t04h.08m.02s"
folder = 'PAPER2\\R4.2_sigmaadjust\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

df_avg = df.groupby(["subject", "model", "th", "cer", "g", "pth", "sigmath"]).mean().reset_index()

## Calculate snr
df_avg["snr"] = np.log((df_avg.max_th - df_avg.min_th)/df_avg.sigmath)

for i, subject in enumerate(list(set(df_avg.subject))):

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1, subplot_titles=[r"$r_{PLV}$", "FFT peak", "SNR (th)", "Bifurcations (cx)"],
                        x_title=r'$\text{Gaussian std. of thalamic input } (\eta_{th})$', y_title=r"$\text{Coupling factor (g)}$",
                        shared_yaxes=True, shared_xaxes=True)

    subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == "pTh")
                        & (df_avg["sigmath"] > 0.0005) & (df_avg["sigmath"] < 5)]

    fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.sigmath, y=subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
                             showscale=True, colorbar=dict(title="r", thickness=4,  x=0.265)), row=1, col=1)

    fig.add_trace(go.Heatmap(z=subset.IAF, x=subset.sigmath, y=subset.g, colorscale='Turbo',
                   showscale=True, colorbar=dict(title="Hz", thickness=4, x=0.63)), row=1, col=2)


    fig.add_trace(go.Heatmap(z=subset.snr, x=subset.sigmath, y=subset.g, colorscale='Geyser',
                   showscale=True, zmin=-(np.max(subset.snr)), zmax=np.max(subset.snr), colorbar=dict(title="log<br>(snr)", thickness=4,  x=1)), row=1, col=3)

    # fig.add_trace(go.Heatmap(z=subset.max_cx - subset.min_cx, x=subset.sigmath, y=subset.g, colorscale='Viridis',
    #                showscale=True, zmin=0, zmax=0.14, colorbar=dict(title="mV", thickness=4,  x=1)), row=1, col=4)

fig.add_vline(x=0.05, col=[1, 2], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
fig.add_vline(x=0.15, col=[1, 2], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
fig.add_vline(x=0.05, col=[3], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)
fig.add_vline(x=0.15, col=[3], line_width=1, line_dash="dot", line_color="gray", opacity=0.6)

fig.update_layout(width=650, height=400, xaxis1=dict(type="log"), xaxis2=dict(type="log"),
                  xaxis3=dict(type="log"), xaxis4=dict(type="log"))

pio.write_html(fig, file=folder + "/PAPER5sm-sigmaadjust.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER5sm-sigmaadjust.svg", engine="kaleido")


## FIG 4.2b - FC specifics
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15, subplot_titles=["PLV mean", "PLV std", "SNR (th)", "Bifurcations (cx)"],
                    x_title=r'$\text{Intrinsic mean input to thalamic nodes } (p_{th})$', y_title=r"$\text{Coupling factor (g)}$",
                    shared_yaxes=True, shared_xaxes=True)

subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == "pTh")]

fig.add_trace(go.Heatmap(z=subset.plv_m, x=subset.sigmath, y=subset.g, colorscale='Turbo',
                         showscale=True, colorbar=dict(title="", thickness=4,  x=0.425)), row=1, col=1)

fig.add_trace(go.Heatmap(z=subset.plv_sd, x=subset.sigmath, y=subset.g, colorscale='Turbo',
               showscale=True, reversescale=True, colorbar=dict(title="", thickness=4,  x=1)), row=1, col=2)


# fig.add_trace(go.Heatmap(z=subset.dfc_m, x=subset.pth, y=subset.g, colorscale='Geyser',
#                showscale=True, colorbar=dict(title="log<br>(snr)", thickness=4,  x=0.73)), row=1, col=3)
#
# fig.add_trace(go.Heatmap(z=subset.dfc_sd, x=subset.pth, y=subset.g, colorscale='Viridis',
#                showscale=True, zmin=0, zmax=0.14, colorbar=dict(title="mV", thickness=4,  x=1)), row=1, col=4)

fig.update_layout(width=500, height=400, xaxis1=dict(type="log"), xaxis2=dict(type="log"))
fig.add_vline(x=0.05, col=[1, 2], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
fig.add_vline(x=0.15, col=[1, 2], line_width=1, line_dash="dot", line_color="lightgray", opacity=0.6)
pio.write_html(fig, file=folder + "/PAPER4.2bsm-sigmaadjust.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER4.2bsm-sigmaadjust.svg", engine="kaleido")
