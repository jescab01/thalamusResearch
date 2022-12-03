
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio


simulations_tag = "PSEmpi_adjustrange_P-m10d05y2022-t17h.57m.56s"
folder = 'PAPER\\R4.1_padjust\\' + simulations_tag + '\\'
df = pd.read_csv(folder + "results.csv")

df_avg = df.groupby(["subject", "model", "th", "cer", "g", "p", "sigma"]).mean().reset_index()

structure_th = ["woTh", "Th",  "pTh"]

for i, subject in enumerate(list(set(df_avg.subject))):

    title = subject + "_paramSpace"
    auto_open = True if i < 1 else False

    fig = make_subplots(rows=3, cols=3,
                        row_titles=("$r_{PLV}$", "bifurcations (cx)", "bifurcations (th)"),
                        x_title=r'$\text{Intrinsic mean input to thalamic nodes } (p_{th})$', y_title="Coupling factor (g)",
                        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]],
                        column_titles=["woTh", "Th", "pTh"],
                        shared_yaxes=True, shared_xaxes=True)

    for j, th in enumerate(structure_th):
        subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == th)]

        sl = True if j == 0 else False

        fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.p, y=subset.g, colorscale='RdBu', reversescale=True, zmin=-0.5, zmax=0.5,
                                 showscale=True, colorbar=dict(thickness=7, len=0.25, y=0.85, x=1.05)), row=1, col=(1 + j))

        fig.add_trace(go.Heatmap(z=subset.max_cx - subset.min_cx, x=subset.p, y=subset.g, colorscale='Viridis',
                       showscale=sl, colorbar=dict(thickness=7, len=0.65, y=0.3, x=1.05, title="mV"), zmin=0, zmax=0.14), row=2, col=(1 + j))

        # if th is not "woTh":
        fig.add_trace(go.Heatmap(z=subset.max_th - subset.min_th, x=subset.p, y=subset.g, colorscale='Viridis',
                       showscale=False, zmin=0, zmax=0.14), row=3, col=(1 + j))

    fig.update_layout(template="plotly_white", width=500, height=700)

    pio.write_html(fig, file=folder + "/PAPER7sm-padjust.html", auto_open=auto_open, include_mathjax="cdn")
    pio.write_image(fig, file=folder + "/PAPER7sm-padjust.svg", engine="kaleido")

