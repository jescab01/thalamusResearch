'''
Creating figures and calculating statistics for ThalamusResearch
A priori using already computed gexplore_data: to get an idea of the plotting.

TODO after the idea is ready recompute calculus

  -  WITH OR WITHOUT THALAMUS  -
'''

import pandas as pd
import pingouin as pg
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import pickle


## DATA for g_explore
simulations_tag = "g_explore-FIG4_N35-initialConditions-m10d05y2022-t15h.18m.30s.pkl"
folder = 'PAPER\\R3_gexplore\\gexplore_data\\'

# Load and plot already computed simulations.
with open(folder + simulations_tag, "rb") as input_file:
    g_sel, output = pickle.load(input_file)

param = "g"
plot_ms = 2000



## DATA for reference rPLV & KSD
simulations_tag = "PSEmpi_JRstd0.022TH-m10d04y2022-t21h.16m.56s"
folder = 'PAPER\\R1_TH-type&noise\\' + simulations_tag + '\\'

df = pd.read_csv(folder + "results.csv")
# Average out repetitions and subjects
df_groupavg = df.groupby(["model", "th", "cer", "g", "sigma"]).mean().reset_index()

# Colours
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2
c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.9


##       FIGURE     ##################
n_g = len(g_sel)
add_cols = 2

specs = [[{}, {"colspan": n_g, "secondary_y": True}] + [{}]*(n_g)] + [[{} for g in range(n_g+add_cols)]]*4
id_emp = (n_g + add_cols) * 3
[""] * (n_g + add_cols) + [""] + [param + "==" + str(g) for g in g_sel]

sp_titles_r1 = [param + "==" + str(g) for g in g_sel]
sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g+add_cols)*5)]
# sp_titles[6:6+n_g-1] = sp_titles_r1
# sp_titles[6] = "Cortex(" + sp_titles[6]
# sp_titles[6+n_g-1] = sp_titles[n_g-1] + ")"


fig = make_subplots(rows=5, cols=n_g+add_cols, specs=specs,  # row_titles=["", "signals", "FFT", "FC", "dFC"],
                    shared_yaxes=True, subplot_titles=sp_titles, horizontal_spacing=0.05)

# Plot reference rPLV + KSD in pTh active contition
df_sub_avg = df_groupavg.loc[(df_groupavg["th"] == "pTh") & (df_groupavg["sigma"] == 0.022)]
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, mode="lines", opacity=opacity,
               line=dict(width=4, color=c3), showlegend=False), row=1, col=2)

df_sub_avg = df_groupavg.loc[(df_groupavg["th"] == "pTh") & (df_groupavg["sigma"] == 0.022)]
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, mode="lines", opacity=opacity,
                         line=dict(dash='solid', color=c3, width=2), showlegend=False), secondary_y=True, row=1, col=2)

sl_groups = list()
for i, g in enumerate(g_sel):

    fig.add_vline(x=g, line=dict(dash="dash", color="gray"), row=1, col=2)

    sl = True if i < 1 else False

    # Unpack output
    signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = output[i]

    freqs = np.arange(len(signals[0]) / 2)
    freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

    cmap = px.colors.qualitative.Plotly
    for ii, signal in enumerate(signals):

        regionGroup = regionLabels[ii].split("_")[0]
        if regionGroup in ["Thal", "Frontal", "Temporal", "Parietal", "Occipital"]:

            # controling showlabels
            sl1 = False
            if regionGroup not in sl_groups:
                sl_groups.append(regionGroup)
                sl1 = True

            # If thalamic roi and first g_sel - then plot in apart.
            if regionGroup == "Thal" and i == 0:

                # Timeseries
                fig.add_trace(go.Scatter(x=timepoints[0:plot_ms] / 1000, y=signal[0:plot_ms], name=regionGroup,
                                         legendgroup=regionGroup,
                                         showlegend=sl1, marker_color=cmap[ii % len(cmap)]), row=2, col=n_g + 2)
                # Spectra
                freqRange = [2, 40]
                fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                fft = np.asarray(
                    fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                         marker_color=cmap[ii % len(cmap)], legendgroup=regionGroup,
                                         showlegend=False), row=3, col=n_g + 2)

            elif regionGroup in ["Frontal", "Temporal", "Parietal", "Occipital"]:
                # Timeseries
                fig.add_trace(go.Scatter(x=timepoints[0:plot_ms]/1000, y=signal[0:plot_ms], name=regionGroup,
                                         legendgroup=regionGroup,
                                         showlegend=sl1, marker_color=cmap[ii % len(cmap)]), row=2, col=i+2)
                # Spectra
                freqRange = [2, 40]
                fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                         marker_color=cmap[ii % len(cmap)], legendgroup=regionGroup,
                                         showlegend=False), row=3, col=i+2)

    # Functional Connectivity
    fig.add_trace(go.Heatmap(z=plv, x=regionLabels[SC_cortex_idx], y=regionLabels[SC_cortex_idx], colorbar=dict(thickness=4),
                             colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=4, col=i+2)

    # dynamical Fuctional Connectivity
    step = 2
    fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient/1000, len(dplv) * step, step),
                             y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                             colorbar=dict(thickness=8, len=0.4, y=0, yanchor="bottom"),
                             showscale=sl, zmin=0, zmax=1), row=5, col=i+2)

# empirical FC matrices
fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels[SC_cortex_idx], y=regionLabels[SC_cortex_idx], colorbar=dict(thickness=4), legendgroup="",
                         colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=4, col=1)

# dynamical Fuctional Connectivity
dFC_emp = dFC_emp[:len(dplv)][:, :len(dplv)]
fig.add_trace(go.Heatmap(z=dFC_emp, x=np.arange(transient/1000, len(dFC_emp) * step, step),
                         y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                         showscale=False, zmin=0, zmax=1), row=5, col=1)


# Update layout
w_ = 800 if n_g < 3 else 1000
fig.update_layout(legend=dict(yanchor="top", y=0.75, xanchor="left", x=-0.05, tracegroupgap=2),
                  template="plotly_white", height=1100, width=w_,
                  yaxis2=dict(title="<b>rPLV<b>", showticklabels=True), yaxis3=dict(title="KSD"),
                  xaxis2=dict(title="Coupling factor (g)"))

for col in range(n_g+add_cols):  # +1 empirical column
    # second row
    idx = 1 * (n_g+add_cols) + (col+1)  # +1 to avoid 0 indexing in python
    fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
    if idx == 1 * (n_g+add_cols) + (1+2):
        fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}
        fig["layout"]["yaxis" + str(idx)]["showticklabels"] = True

    # third row
    idx = 2 * (n_g+add_cols) + (col+1)  # +1 to avoid 0 indexing in python
    fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
    if idx == 2 * (n_g+add_cols) + (1+2):
        fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}
        fig["layout"]["yaxis" + str(idx)]["showticklabels"] = True

    # fourth row
    # idx = 3 * n_g+1 + (col+1)  # +1 to avoid 0 indexing in python
    # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
    # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}

    # fifth row
    idx = 4 * (n_g+add_cols) + (col+1)  # +1 to avoid 0 indexing in python
    fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
    if idx == (4 * (n_g+add_cols) + 2):
        fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'Time (s)'}


pio.write_html(fig, file=folder + "/PAPER4_g_explore.html", auto_open=True)
# pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")

