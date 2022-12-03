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


## DATA for reference rPLV & KSD: PRE-
simulations_tag = "PSEmpi_JRstd0.022TH-m10d04y2022-t21h.16m.56s"
folder = 'PAPER\\R1_TH-type&noise\\' + simulations_tag + '\\'

df_pre = pd.read_csv(folder + "results.csv")
# Average out repetitions and subjects
df_pre_avg = df_pre.groupby(["subject", "model", "th", "cer", "g", "sigma"]).mean().reset_index()


## DATA for reference rPLV & KSD: POST-
simulations_tag = "PSEmpi_JRstd0.022TH-post-m10d06y2022-t18h.55m.23s"
folder = 'PAPER\\R4_postCalibration\\data\\' + simulations_tag + '\\'

df_post = pd.read_csv(folder + "results.csv")
# Average out repetitions and subjects
df_post_avg = df_post.groupby(["subject", "model", "th", "cer", "g", "sigma"]).mean().reset_index()


# Colours
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2
c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.9

c4, c5 = "gray", "dimgray" #cmap_s2[-1], cmap_p2[-1]

for id in [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]:

    subj = "NEMOS_0" + str(id)

    ## DATA for g_explore
    simulations_tag = "g_explore-FIG5_N"+str(id)+"-PrePost.pkl"
    folder = 'PAPER\\R4_postCalibration\\data\\'

    # Load and plot already computed simulations.
    with open(folder + simulations_tag, "rb") as input_file:
        ref, output = pickle.load(input_file)

    ##       FIGURE     ##################
    specs = [[{"secondary_y": True, "colspan": 2}, {}, {"secondary_y": True, "colspan": 2}, {}],
             [{"colspan": 2}, {}, {"colspan": 2}, {}],
             [{"secondary_y": True, "colspan": 2}, {}, {"secondary_y": True, "colspan": 2}, {}],
             [{"secondary_y": True, "colspan": 2}, {}, {"secondary_y": True, "colspan": 2}, {}],
             [{"r": -0.08}, {"l": -0.08}, {"r": -0.08}, {"l": -0.08}]]

    fig = make_subplots(rows=5, cols=4, shared_yaxes=False, specs=specs, horizontal_spacing=0.2, vertical_spacing=0.11,
                        row_heights=[0.18, 0.18, 0.18, 0.18, 0.28], column_titles=["<i>Pre-calibration", "", "<i>Post-calibration"])

    for i, phase in enumerate(["pre", "post"]):

        # Load PRE - data
        # Unpack output
        signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = output[i]

        # Plot reference rPLV + KSD in pTh active contition: PRE
        sigma = 0.022 if phase == "pre" else 0.15

        col = i*2+1

        if phase == "pre":
            df_sub_avg = df_pre_avg.loc[(df_pre_avg["th"] == "pTh") & (df_pre_avg["sigma"] == sigma) & (df_pre_avg["subject"] == subj)]

        elif phase == "post":
            df_sub_avg = df_post_avg.loc[(df_post_avg["th"] == "pTh") & (df_post_avg["sigma"] == 0.15) & (df_post_avg["subject"] == subj)]

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, mode="lines", opacity=opacity,
                       line=dict(width=4, color=c3), showlegend=False), row=1, col=col)

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, mode="lines", opacity=opacity,
                                 line=dict(dash='solid', color=c3, width=2), showlegend=False), secondary_y=True, row=1, col=col)

        # Plot CX bifurcation
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_cx, mode="lines",
                                 line=dict(width=4, color=c4), showlegend=False), row=2, col=col)

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_cx,  mode="lines",
                                 line=dict(width=4, color=c4), showlegend=False), row=2, col=col)

        # Plot TH bifurcation
        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_th, legendgroup="Thal", mode="lines",
                                 line=dict(width=2, dash='dot', color=c5), showlegend=False), row=2, col=col)

        fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_th, legendgroup="Thal", mode="lines",
                                 line=dict(width=2, dash='dot', color=c5), showlegend=False), row=2, col=col)

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
                                             legendgroup=regionGroup,
                                             showlegend=sl, marker_color=cmap[ii % len(cmap)]), secondary_y=sy, row=3, col=col)
                    # Spectra
                    freqRange = [2, 40]
                    fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
                    fft = np.asarray(
                        fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
                    fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
                    fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
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
                                 colorbar=dict(thickness=8, len=0.17, y=0, x=0.98, yanchor="bottom", title="Pearson's r"),
                                 showscale=True, zmin=0, zmax=1), row=5, col=i*2+2)

    fig.update_layout(template="plotly_white", width=1000, height=900,
                      legend=dict(yanchor="top", y=0.64),
                      yaxis1=dict(title="<b>rPLV<b>"), yaxis2=dict(title="KSD"),
                      #yaxis4=dict(title="<b>rPLV<b>"), yaxis5=dict(title="KSD"),

                      yaxis7=dict(title="min-max<br>Voltage (mV)<br><b>cx</b> | th"),
                      # yaxis9=dict(title="min-max<br>Voltage (mV)<br><b>cx</b> | th"),

                      yaxis11=dict(title="Voltage (mV)<br>-cx-"), yaxis12=dict(title="-th-"),
                      yaxis17=dict(title="Power (dB)<br>-cx-"), yaxis18=dict(title="-th-"),

                      yaxis25=dict(showticklabels=False), #yaxis24=dict(title="Time (ms)"), #yaxis26=dict(title="Time (ms)"),

                      xaxis1=dict(title="Coupling factor (g)"), xaxis3=dict(title="Coupling factor (g)"),
                      xaxis5=dict(title="Coupling factor (g)"), xaxis7=dict(title="Coupling factor (g)"),

                      xaxis9=dict(title="Time (ms)"), xaxis11=dict(title="Time (ms)"),
                      xaxis13=dict(title="Frequency (Hz)"), xaxis15=dict(title="Frequency (Hz)"),
                      xaxis18=dict(title="Time (ms)"), xaxis20=dict(title="Time (ms)"),
                      )
    pio.write_html(fig, file=folder + "/PAPER9sm_PrePost_"+subj+".html", auto_open=True)
    # pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")





    # Update layout
    # w_ = 900 if n_g < 3 else 1000
    # fig.update_layout(legend=dict(yanchor="top", y=0.75, xanchor="left", x=1, tracegroupgap=10),
    #                   template="plotly_white", height=1100, width=w_,
    #                   yaxis2=dict(title="<b>rPLV<b>", showticklabels=True), yaxis3=dict(title="KSD"),
    #                   xaxis2=dict(title="Coupling factor (g)"))

    # for col in range(n_g+1):  # +1 empirical column
    #     # second row
    #     idx = 1 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
    #     fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
    #     fig["layout"]["yaxis" + str(idx)]["showticklabels"] = True
    #

