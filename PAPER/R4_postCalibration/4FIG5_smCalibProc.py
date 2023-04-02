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
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import pickle



## DATA for reference rPLV & KSD: PRE-POST && DATA for g_explore
simulations_tag = "PSEmpi_CalibProc_prepost-m12d09y2022-t20h.07m.01s"

if "PAPER" in os.getcwd():
    folder = 'data\\' + simulations_tag + '\\'
else:
    folder = 'PAPER\\R4_postCalibration\\data\\' + simulations_tag + '\\'

df_prepost = pd.read_csv(folder + "results.csv")
# Average out repetitions and subjects
df_prepost_avg = df_prepost.groupby(["subject", "model", "th", "cer", "g", "pth", "sigma", "pcx"]).mean().reset_index()


# Colours
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2
c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.9

c4, c5 = "gray", "dimgray" #cmap_s2[-1], cmap_p2[-1]


## g-explore data
subj_ids = [58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

# for each of the simulated gexplores (subjects):
for subj in subjects:

    # Load and plot already computed simulations.
    with open(folder + "\\g_explore-" + subj + "inTestingGroup-PrePost.pkl", "rb") as input_file:
        output = pickle.load(input_file)

    pn_vals = [("0", 0.09, 0.022, 0.09), ("A", 0.15, 0.022, 0.09), ("B", 0.15, 0.15, 0.09), ("C", 0.15, 0.15, "MLR")]
    sp_t = [[r"$%s.   p_{th}=%0.2f;  \eta_{th}=%0.2f;  p_{\neq th}=%s$" % (id, pth, sigma, pcx), ""] for i, (id, pth, sigma, pcx) in enumerate(pn_vals)]
    sp_t = [elem for sp in sp_t for elem in sp]

    ##       FIGURE     ##################
    n_sim = len(output)
    specs = [[{"secondary_y": True, "colspan": 2}, {}] * n_sim,
             [{"colspan": 2}, {}] * n_sim,
             [{"secondary_y": True, "colspan": 2}, {}] * n_sim,
             [{"secondary_y": True, "colspan": 2}, {}] * n_sim,
             [{"r": -0.03}, {"l": -0.03}] * n_sim]

    fig = make_subplots(rows=5, cols=n_sim*2, shared_yaxes=False, specs=specs, vertical_spacing=0.11,
                        horizontal_spacing=0.08, column_titles=sp_t, row_heights=[0.18, 0.18, 0.18, 0.18, 0.28])

    for i, (set, sim) in enumerate(output):

        # Unpack output
        subj, g, pth, sigma, pcx = set
        signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = sim[0]

        df_sub_avg = df_prepost_avg.loc[(df_prepost_avg["subject"] == subj) & (df_prepost_avg["th"] == "pTh") &
                                        (df_prepost_avg["pth"] == pth) & (df_prepost_avg["sigma"] == sigma) &
                                        (df_prepost_avg["pcx"] == str(pcx))]

        col = i * 2 + 1

        # Plot rPLV
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, mode="lines", opacity=opacity,
                       line=dict(width=4, color=cmap_p2[2]), showlegend=False), row=1, col=col)

        # Plot KSD
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, mode="lines", opacity=opacity,
                                 line=dict(dash='solid', color=c3, width=2), showlegend=False),
                      secondary_y=True, row=1, col=col)

        # Plot CX bifurcation
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_cx, mode="lines",
                                 line=dict(width=4, color=c4), showlegend=False), row=2, col=col)

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_cx,  mode="lines",
                                 line=dict(width=4, color=c4), showlegend=False), row=2, col=col)

        # Plot TH bifurcation
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_th, legendgroup="Thal", mode="lines",
                                 line=dict(width=2, dash='dot', color=c5), showlegend=False),
                      row=2, col=col)

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_th, legendgroup="Thal", mode="lines",
                                 line=dict(width=2, dash='dot', color=c5), showlegend=False),
                      row=2, col=col)

        fig.add_vline(x=2, line=dict(dash="dash", color="lightgray"), row=1, col=col)
        fig.add_vline(x=2, line=dict(dash="dash", color="lightgray"), row=2, col=col)

        ## SIGNAls and SPECTRA: PRE-
        freqs = np.arange(len(signals[0]) / 2)
        freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

        cmap = px.colors.qualitative.Plotly
        sl_groups = list()
        for ii, signal in enumerate(signals):

            regionGroup = regionLabels[ii].split("_")[0]
            if regionGroup in ["Thal", "Frontal", "Temporal", "Parietal", "Occipital"]:

                if regionGroup not in sl_groups:
                    sl_groups.append(regionGroup)

                sy = True if regionGroup == "Thal" else False
                sl = True if i==0 else False

                if Counter(sl_groups)[regionGroup] < 3:
                    sl_groups.append(regionGroup)
                    # Timeseries
                    fig.add_trace(go.Scatter(x=timepoints[2900:5000] / 1000, y=signal[2900:5000], name=regionLabels[ii],
                                             legendgroup=regionGroup, line=dict(width=1), opacity=0.7,
                                             showlegend=sl, marker_color=cmap[ii % len(cmap)]), secondary_y=sy, row=3, col=col)
                    # Spectra
                    freqRange = [2, 40]
                    fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                    fft = np.asarray(
                        fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                    fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                    fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
                                             line=dict(width=1), opacity=0.7,
                                             marker_color=cmap[ii % len(cmap)], legendgroup=regionGroup, name=regionLabels[ii],
                                             showlegend=False), secondary_y=sy, row=4, col=col)

        # Functional Connectivity
        fig.add_trace(
            go.Heatmap(z=plv, x=regionLabels[SC_cortex_idx], y=regionLabels[SC_cortex_idx], colorbar=dict(thickness=4),
                       colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=5, col=i*2+1)

        # dynamical Fuctional Connectivity
        step = 2
        fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient / 1000, len(dplv) * step, step),
                                 y=np.arange(transient / 1000, len(dplv) * step, step), colorscale='Viridis',
                                 colorbar=dict(thickness=8, len=0.17, y=-0.025, x=0.98, yanchor="bottom"),
                                 showscale=True, zmin=0, zmax=1), row=5, col=i*2+2)

    fig.update_layout(template="plotly_white", legend=dict(yanchor="top", y=0.68, x=0.97), width=1500, height=760,
                      yaxis1=dict(title="$r_{PLV}$", range=[0, 0.6]), yaxis2=dict(title="KSD", range=[0, 1]),
                      yaxis4=dict(title="$r_{PLV}$", range=[0, 0.6]), yaxis5=dict(title="KSD", range=[0, 1]),
                      yaxis7=dict(title="$r_{PLV}$", range=[0, 0.6]), yaxis8=dict(title="KSD", range=[0, 1]),
                      yaxis10=dict(title="$r_{PLV}$", range=[0, 0.6]), yaxis11=dict(title="KSD", range=[0, 1]),
                      yaxis13=dict(title="min-max<br>Voltage (mV)<br><b>cx</b> | th"),
                      yaxis21=dict(title="Voltage (mV)<br>cx | th"),
                      yaxis33=dict(title="Power (dB)<br>cx | th"),
                      yaxis47=dict(showticklabels=False), yaxis49=dict(showticklabels=False),
                      yaxis51=dict(showticklabels=False),

                      xaxis1=dict(title="Coupling factor (g)"), xaxis3=dict(title="Coupling factor (g)"),
                      xaxis5=dict(title="Coupling factor (g)"), xaxis7=dict(title="Coupling factor (g)"),

                      xaxis9=dict(title="Coupling factor (g)"), xaxis11=dict(title="Coupling factor (g)"),
                      xaxis13=dict(title="Coupling factor (g)"), xaxis15=dict(title="Coupling factor (g)"),

                      xaxis17=dict(title="Time (ms)"), xaxis19=dict(title="Time (ms)"),
                      xaxis21=dict(title="Time (ms)"), xaxis23=dict(title="Time (ms)"),
                      xaxis25=dict(title="Frequency (Hz)"), xaxis27=dict(title="Frequency (Hz)"),
                      xaxis29=dict(title="Frequency (Hz)"), xaxis31=dict(title="Frequency (Hz)"),

                      xaxis34=dict(title="Time (ms)"), xaxis36=dict(title="Time (ms)"),
                      xaxis38=dict(title="Time (ms)"), xaxis40=dict(title="Time (ms)"),
                      )

    pio.write_html(fig, file=folder + "/PAPER5_CalibProc_"+subj+".html", auto_open=False, include_mathjax="cdn")
    pio.write_image(fig, file=folder + "/PAPER5_CalibProc_"+subj+".svg", engine="kaleido")



