'''
Creating figures and calculating statistics for ThalamusResearch
A priori using already computed gexplore_data: to get an idea of the plotting.

TODO after the idea is ready recompute calculus

  -  WITH OR WITHOUT THALAMUS  -
'''

import pandas as pd
import pingouin as pg
import numpy as np
from collections import Counter

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import pickle


## DATA for reference rPLV & KSD
simulations_tag = "PSEmpi_JRstd0.022TH-m10d04y2022-t21h.16m.56s"
folder = 'PAPER\\R1_TH-type&noise\\' + simulations_tag + '\\'

df = pd.read_csv(folder + "results.csv")
# Average out repetitions and subjects
df_groupavg = df.groupby(["model", "th", "cer", "g", "sigma"]).mean().reset_index()


## DATA for g_explore
simulations_tag = "g_explore-FIG4_N35-initialConditions-m10d05y2022-t15h.18m.30s.pkl"
folder = 'PAPER\\R3_gexplore\\data\\'

# Load and plot already computed simulations.
with open(folder + simulations_tag, "rb") as input_file:
    g_sel, output = pickle.load(input_file)

param = "g"


# Colours
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2
c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.9
c4, c5 = "gray", "dimgray" #cmap_s2[-1], cmap_p2[-1]

##       FIGURE     ##################
n_g = len(g_sel)
specs = [[{}, {"colspan": n_g, "secondary_y": True}] + [{}]*(n_g-1)] +\
        [[{}, {"colspan": n_g                     }] + [{}]*(n_g-1)] +\
        [[{} for g in range(n_g+1)]]*4

id_emp = (n_g + 1) * 4 - 1
sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g+1)*6)]
sp_titles_r1 = [param + "==" + str(g) for g in g_sel]
sp_titles[11:11+n_g-1] = sp_titles_r1


fig = make_subplots(rows=6, cols=n_g+1, specs=specs, row_titles=["", "bifurcations", "signals", "FFT", "FC", "dFC"],
                    shared_yaxes=False, subplot_titles=sp_titles, horizontal_spacing=0.05)

# Plot reference rPLV + KSD in pTh active contition
df_sub_avg = df_groupavg.loc[(df_groupavg["th"] == "pTh") & (df_groupavg["sigma"] == 0.022)]
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, mode="lines", opacity=opacity,
               line=dict(width=4, color=c3), showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, mode="lines", opacity=opacity,
                         line=dict(dash='solid', color=c3, width=2), showlegend=False), secondary_y=True, row=1, col=2)

# Plot CX bifurcation
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_cx, name="cortical ROIs",
                         legendgroup="cortical ROIs", mode="lines",
                         line=dict(width=4, color=c4), showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_cx, name="cortical ROIs",
                         legendgroup="cortical ROIs", mode="lines",
                         line=dict(width=4, color=c4), showlegend=False), row=2, col=2)

# Plot TH bifurcation
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_th, name="thalamic ROIs",
                         legendgroup="thalamic ROIs", mode="lines",
                         line=dict(width=2, dash='dot', color=c5),
                         showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_th, name="thalamic ROIs",
                         legendgroup="thalamic ROIs", mode="lines",
                         line=dict(width=2, dash='dot', color=c5),
                         showlegend=False), row=2, col=2)


for i, g in enumerate(g_sel):

    fig.add_vline(x=g, line=dict(dash="dash", color="lightgray"), row=1, col=2)
    fig.add_vline(x=g, line=dict(dash="dash", color="lightgray"), row=2, col=2)

    sl = True if i < 1 else False

    # Unpack output
    signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = output[i]

    freqs = np.arange(len(signals[0]) / 2)
    freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

    cmap = px.colors.qualitative.Plotly
    sl_groups = list()
    for ii, signal in enumerate(signals):

        regionGroup = regionLabels[ii].split("_")[0]
        if regionGroup in ["Thal", "Frontal", "Temporal", "Parietal", "Occipital"]:

            if regionGroup not in sl_groups:
                sl_groups.append(regionGroup)

            if Counter(sl_groups)[regionGroup] < 3:
                sl1 = True if i == 0 else False
                sl_groups.append(regionGroup)
                # Timeseries
                fig.add_trace(go.Scatter(x=timepoints[2900:5000]/1000, y=signal[2900:5000], name=regionLabels[ii],
                                         legendgroup=regionGroup,
                                         showlegend=sl1, marker_color=cmap[ii % len(cmap)]), row=3, col=i+2)
                # Spectra
                freqRange = [2, 40]
                fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                         marker_color=cmap[ii % len(cmap)], legendgroup=regionGroup, name=regionLabels[ii],
                                         showlegend=False), row=4, col=i+2)

    # Functional Connectivity
    fig.add_trace(go.Heatmap(z=plv, x=regionLabels[SC_cortex_idx], y=regionLabels[SC_cortex_idx], colorbar=dict(thickness=4),
                             colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=5, col=i+2)

    # dynamical Fuctional Connectivity
    step = 2
    fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient/1000, len(dplv) * step, step),
                             y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                             colorbar=dict(thickness=8, len=0.25, y=0, yanchor="bottom"),
                             showscale=sl, zmin=0, zmax=1), row=6, col=i+2)

# empirical FC matrices
fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels[SC_cortex_idx], y=regionLabels[SC_cortex_idx], colorbar=dict(thickness=4), legendgroup="",
                         colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=5, col=1)

# dynamical Fuctional Connectivity
dFC_emp = dFC_emp[:len(dplv)][:, :len(dplv)]
fig.add_trace(go.Heatmap(z=dFC_emp, x=np.arange(transient/1000, len(dFC_emp) * step, step),
                         y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
                         showscale=False, zmin=0, zmax=1), row=6, col=1)


# Update layout
w_ = 900 if n_g < 3 else 1000
fig.update_layout(legend=dict(yanchor="top", y=0.6, xanchor="left", x=1, tracegroupgap=10),
                  template="plotly_white", height=1200, width=w_,
                  yaxis2=dict(title="$r_{PLV}$", showticklabels=True), yaxis3=dict(title="KSD"),
                  yaxis8=dict(title="min-max<br>Voltage (mV)<br><b>cx</b> | th"), xaxis7=dict(title="Coupling factor (g)"))

for col in range(n_g+1):  # +1 empirical column
    # Third row
    idx = 2 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
    fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
    fig["layout"]["yaxis" + str(idx)]["showticklabels"] = True
    if idx == 2 * (n_g+1) + (1+2):
        fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}


    # fourth row
    idx = 3 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
    fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
    fig["layout"]["yaxis" + str(idx)]["showticklabels"] = True
    if idx == 3 * (n_g+1) + (1+2):
        fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}


    # fifth row
    idx = 4 * (n_g+1) + (col+1) + 2 # +1 to avoid 0 indexing in python
    fig["layout"]["yaxis" + str(idx)]["showticklabels"] = False


    # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
    # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}

    # sixth row
    idx = 5 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
    fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
    if idx == (5 * (n_g+1) + 2):
        fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'Time (s)'}


pio.write_html(fig, file=folder + "/PAPER4_g_explore.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "/PAPER4_g_explore.svg", engine="kaleido")


## Calculate signals amplitude vs dc-offset
# i = 3
# signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = output[i]
#
# # DC-offset
# sig_avg = np.average(signals[SC_cortex_idx, :], axis=1)
# df_offset = np.average([np.abs(s - np.average(sig_avg)) for s in sig_avg])
#
# # signals amplitude
# amplitudes = np.average(np.array([s.max() - s.min() for s in signals[SC_cortex_idx]]))
#
# dc_offset/amplitudes
#
# a=np.average(np.std(signals, axis=1))
