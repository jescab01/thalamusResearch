
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from tvb.simulator.lab import connectivity


ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
main_folder = "E:\LCCN_Local\PycharmProjects\\thalamusResearch\PAPER2\Rx_Review\\"


## 1. Plotting PSE with g&s :: INITIAL PSE
# Load data
simtag = "PSEmpi_JR_v2-m05d20y2023-t02h.12m.34s"

results = pd.read_csv(main_folder + simtag + "\\results.csv")
results = results.groupby(["th", "g", "s"]).mean().reset_index()

th_modes = ["woTh", "Th", "pTh"]

# Plot
fig = make_subplots(rows=3, cols=3, column_titles=th_modes, row_titles=["rPLV", "IAF", "Power"])

for j, th in enumerate(th_modes):
    ss = True if j==0 else False
    subset = results.loc[results["th"] == th]
    fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.s, y=subset.g,
                             colorscale="RdBu", reversescale=True, zmin=-0.5, zmax=0.5, showscale=ss,
                             colorbar=dict(len=0.28, thickness=10, y=0.85, title="r")), row=1, col=1+j)
    fig.add_trace(go.Heatmap(z=subset.IAF, x=subset.s, y=subset.g, colorscale="Turbo", showscale=ss,
                             colorbar=dict(len=0.28, thickness=10, y=0.5, title="Hz")), row=2, col=1+j)
    fig.add_trace(go.Heatmap(z=subset.bModule, x=subset.s, y=subset.g, colorscale="Viridis", showscale=ss,
                             colorbar=dict(len=0.28, thickness=10, y=0.125, title="dB")), row=3, col=1+j)

fig.update_layout(template="plotly_white", height=750, width=600)
pio.write_html(fig, main_folder + "PSE_g&s_review_pre.html", auto_open=True)




## 2. Plotting PSE with g&s :: INITIAL PSE
# Load data
simtag = "PSEmpi_JR_v2-m05d22y2023-t20h.03m.15s"

results = pd.read_csv(main_folder + simtag + "\\results.csv")
results = results.groupby(["th", "g", "s"]).mean().reset_index()

th_modes = ["woTh", "Th", "pTh"]

# Plot
fig = make_subplots(rows=3, cols=3, column_titles=th_modes, row_titles=["rPLV", "IAF", "Power"])

for j, th in enumerate(th_modes):
    ss = True if j==0 else False
    subset = results.loc[results["th"] == th]
    fig.add_trace(go.Heatmap(z=subset.rPLV, x=subset.s, y=subset.g,
                             colorscale="RdBu", reversescale=True, zmin=-0.5, zmax=0.5, showscale=ss,
                             colorbar=dict(len=0.28, thickness=10, y=0.85, title="r")), row=1, col=1+j)
    fig.add_trace(go.Heatmap(z=subset.IAF, x=subset.s, y=subset.g, colorscale="Turbo", showscale=ss,
                             colorbar=dict(len=0.28, thickness=10, y=0.5, title="Hz")), row=2, col=1+j)
    fig.add_trace(go.Heatmap(z=subset.bModule, x=subset.s, y=subset.g, colorscale="Viridis", showscale=ss,
                             colorbar=dict(len=0.28, thickness=10, y=0.125, title="dB")), row=3, col=1+j)

fig.update_layout(template="plotly_white", height=750, width=600)
pio.write_html(fig, main_folder + "PSE_g&s_review_post.html", auto_open=True)





## 3. Plot dynamical spectra

# Load data
simtag = "briefPSE_spectralExploration_m05d19y2023-t20h.26m.48s.pkl"
dynData = pd.read_pickle(main_folder + simtag)



# Select rois to plot
th = "pTh"
rois_ids = [1, 15, 22, 58]
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2" + th + "_pass.zip")
rois = [conn.region_labels[id] for id in rois_ids] + ["avg"]
rois_ids += ["avg"]

freqs = dynData.loc[0, "freqs"]

rtitles = ["g == " + str(g) for g in sorted(set(dynData.g))]

fig = make_subplots(rows=2, cols=len(rois), column_titles=rois, x_title="Frequency (Hz)", row_titles=rtitles)
## Add initial traces
frames = []
for s in sorted(set(dynData.s)):
    data = []
    for i, g in enumerate(sorted(set(dynData.g))):
        subset = dynData.loc[(dynData["th"] == th) & (dynData["g"] == g) & (dynData["s"] == s)]

        for j, id in enumerate(rois_ids):
            spectrum = subset.power.values[0][id, :] if id != "avg" else np.average(subset.power.values[0], axis=0)

            if s == 0.5:
                fig.add_trace(go.Scatter(x=freqs, y=spectrum), row=1+i, col=j+1)

            data.append(go.Scatter(y=spectrum))

    frames.append(go.Frame(data=data, traces=list(range(len(rois)*2)), name=str(s)))

fig.update(frames=frames)



# CONTROLS : Add sliders and buttons
fig.update_layout(

    template="plotly_white", legend=dict(tracegroupgap=2),
    sliders=[dict(
        steps=[dict(method='animate',
                    args=[[str(s)], dict(mode="immediate",
                                         frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                         transition=dict(duration=0))],
                    label=str(s)) for i, s in enumerate(sorted(set(dynData.s)))],
        transition=dict(duration=0), x=0.15, xanchor="left", y=-0.15,
        currentvalue=dict(font=dict(size=15), prefix="Conduction speed" + " - ", visible=True, xanchor="right"),
        len=0.8, tickcolor="white")],

    updatemenus=[dict(type="buttons", showactive=False, y=-0.2, x=0, xanchor="left",
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None, dict(frame=dict(duration=250, redraw=True, easing="cubic-in-out"),
                                                     transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                               dict(label="Pause", method="animate",
                                    args=[[None],
                                          dict(frame=dict(duration=250, redraw=False, easing="cubic-in-out"),
                                               transition=dict(duration=0), mode="immediate")])])])

pio.write_html(fig, file=main_folder + "/animated_spectra.html", auto_open=True, auto_play=False)





