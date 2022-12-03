'''

Final model is:

Y = 0.08981 + g * (-0.000332632750063 degree_rel + -0.003015538178012 degree_fromth_rel)
'''

import time
import os
import numpy as np
import pandas as pd

import pickle
from report.functions import g_explore, simulate
from tvb.simulator.lab import connectivity

data_folder = "E:\LCCN_Local\PycharmProjects\\thalamusResearch\PAPER\R4.3_phetero-cx\data\\"
sim_tag = "pHetero-SUBJECTs_m11d23y2022-t19h.18m.27s"

table = pd.read_pickle(data_folder + sim_tag + "/.1pHeteroGD_TABLE-SUBJECTS.pkl")

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"

subj_ids = [64, 65, 71, 75, 77]  # Validate just with testing subjects
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

plot_subj = ["NEMOS_077"]  ## Define a subject to plot

th = "pTh"

plot, dc_offsets = [], []
for subject in sorted(set(subjects)):
    for g in np.arange(0.5, 3, 0.5):
        print("subject%s - g%0.2f" % (subject, g))
        table_sub = table.loc[(table["subject"] == subject) & (table["g"] == g)]

        # Simulate ::
        conn = connectivity.Connectivity.from_file(ctb_folder + subject + "_AAL2pTh_pass.zip")

        ## 1. With pth and sigmath calibrated model
        raw_data, raw_time, _, _, _, _, regionLabels, _, _ = \
            simulate(subject, "jr", g=g, p_th=0.15, sigma=0.22, mode="sim")

        # computa las medias de cada señal
        signals_avg = np.average(raw_data, axis=1)
        glob_avg = np.average(signals_avg)

        glob_diff = np.sum(np.abs(signals_avg - glob_avg))
        dc_offsets.append(["base", subject, g, glob_diff])

        if (subject == plot_subj[0]) & (g == plot_subj[1]):
            plot.append(["base", subject, g, raw_data, raw_time, regionLabels])

        ## 2. (1) + MLR pcx
        p_predicted = [0.09 + g * (-0.0003 * (table_sub["degree"].loc[table_sub["roi"]==roi].values[0] - table_sub["degree_avg"].loc[table_sub["roi"]==roi].values[0]) +
                                   -0.003 * (table_sub["degree_fromth"].loc[table_sub["roi"]==roi].values[0] - table_sub["degree_fromth_avg"].loc[table_sub["roi"]==roi].values[0]))
                       if roi in table_sub["roi"].values else 0.15 for roi in conn.region_labels]
        raw_data, raw_time, _, _, _, _, regionLabels, _, _ = \
            simulate(subject, "jr", g=g, sigma=0.22, p_array=p_predicted)

        # computa DC-offsets
        signals_avg = np.average(raw_data, axis=1)
        glob_avg = np.average(signals_avg)

        glob_diff = np.sum(np.abs(signals_avg - glob_avg))
        dc_offsets.append(["MLR", subject, g, glob_diff])

        if (subject == plot_subj[0]):
            plot.append(["MLR", subject, g, raw_data, raw_time, regionLabels])

        ## 3. (1) + GD pcx
        p_adjusted = [table_sub["p_adjusted"].loc[table_sub["roi"]==roi].values[0] if roi in table_sub["roi"].values else 0.15 for roi in conn.region_labels]
        raw_data, raw_time, _, _, _, _, regionLabels, _, _ = \
            simulate(subject, "jr", g=g, sigma=0.22, p_array=p_adjusted)

        # computa DC-offsets
        signals_avg = np.average(raw_data, axis=1)
        glob_avg = np.average(signals_avg)

        glob_diff = np.sum(np.abs(signals_avg - glob_avg))
        dc_offsets.append(["GD", subject, g, glob_diff])

        if (subject == plot_subj[0]) :
            plot.append(["GD", subject, g, raw_data, raw_time, regionLabels])


dc_offsets = pd.DataFrame(dc_offsets, columns=["mode", "subj", "g", "offset"])


## Save simulations results using pickle
file = open(data_folder + sim_tag + "/.2pHeteroMLR-Validation.pkl", "wb")
pickle.dump([dc_offsets, plot], file)
file.close()

import plotly.express as px
fig = px.scatter(dc_offsets, x="mode", y="offset", color="g")
fig.show("browser")


#         dcoffset_reducval(output, [0.09, "MLR", "GD"], param="p(cx)", mode="html", folder=folder)
#
#
# def dcoffset_reducval(output, g_sel, param="g", mode="html", folder="figures"):
#
#     if len(output[0]) == 9:
#
#         n_g = len(g_sel)
#         col_titles = [""] + [param + "==" + str(g) for g in g_sel]
#         specs = [[{} for g in range(n_g+1)]]*4
#         id_emp = (n_g + 1) * 2
#         sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g+1)*4)]
#         fig = make_subplots(rows=4, cols=n_g+1, specs=specs, row_titles=["signals", "FFT", "FC", "dFC"],
#                             column_titles=col_titles, shared_yaxes=True, subplot_titles=sp_titles)
#
#         for i, g in enumerate(g_sel):
#
#             sl = True if i < 1 else False
#
#             # Unpack output
#             signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient = output[i]
#
#             freqs = np.arange(len(signals[0]) / 2)
#             freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs
#
#             cmap = px.colors.qualitative.Plotly
#             for ii, signal in enumerate(signals):
#
#                 # Timeseries
#                 fig.add_trace(go.Scatter(x=timepoints[:5000]/1000, y=signal[:5000], name=regionLabels[ii],
#                                          legendgroup=regionLabels[ii],
#                                          showlegend=sl, marker_color=cmap[ii % len(cmap)]), row=1, col=i+2)
#                 # Spectra
#                 freqRange = [2, 40]
#                 fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
#                 fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
#                 fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
#                 fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
#                                          marker_color=cmap[ii % len(cmap)], name=regionLabels[ii],
#                                          legendgroup=regionLabels[ii], showlegend=False), row=2, col=i+2)
#
#             # Functional Connectivity
#             fig.add_trace(go.Heatmap(z=plv, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
#                                      colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=i+2)
#
#             # dynamical Fuctional Connectivity
#             step = 2
#             fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient/1000, len(dplv) * step, step),
#                                      y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
#                                      colorbar=dict(thickness=8, len=0.4, y=0, yanchor="bottom"),
#                                      showscale=sl, zmin=0, zmax=1), row=4, col=i+2)
#
#         # empirical FC matrices
#         fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4), legendgroup="",
#                                  colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=1)
#
#         # dynamical Fuctional Connectivity
#         dFC_emp=dFC_emp[:len(dplv)][:, :len(dplv)]
#         fig.add_trace(go.Heatmap(z=dFC_emp, x=np.arange(transient/1000, len(dFC_emp) * step, step),
#                                  y=np.arange(transient/1000, len(dplv) * step, step), colorscale='Viridis',
#                                  showscale=False, zmin=0, zmax=1), row=4, col=1)
#
#         w_ = 800 if n_g < 3 else 1000
#         fig.update_layout(legend=dict(yanchor="top", y=1.05, tracegroupgap=1),
#                           template="plotly_white", height=900, width=w_)
#
#         # Update layout
#         for col in range(n_g+1):  # +1 empirical column
#             # first row
#             idx = col + 1  # +1 to avoid 0 indexing in python
#             if idx > 1:
#                 fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
#                 if idx == 2:
#                     fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}
#
#             # second row
#             idx = 1 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
#             if idx > 1 + n_g:
#                 fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
#                 if idx == 3 + n_g:
#                     fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}
#
#             # third row
#             # idx = 2 * n_g+1 + (col+1)  # +1 to avoid 0 indexing in python
#             # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
#             # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
#
#             # fourth row
#             idx = 3 * (n_g+1) + (col+1)  # +1 to avoid 0 indexing in python
#             fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
#             if idx == (3 * (n_g+1) + 1):
#                 fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
#
#         if mode == "html":
#             pio.write_html(fig, file=folder + "/PAPER3_g_explore.html", auto_open=True)
#         elif mode == "png":
#             pio.write_image(fig, file=folder + "/g_explore" + str(time.time()) + ".png", engine="kaleido")
#         elif mode == "svg":
#             pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")
#
#         elif mode == "inline":
#             plotly.offline.iplot(fig)
#
#     elif len(output[0]) == 10:
#
#         n_g = len(g_sel)
#         col_titles = [""] + [param + "==" + str(g) for g in g_sel]
#         specs = [[{} for g in range(n_g+1)]]*5
#         id_emp = (n_g + 1) * 2
#         sp_titles = ["Empirical" if i == id_emp else "" for i in range((n_g+1)*4)]
#         fig = make_subplots(rows=5, cols=n_g+1, specs=specs, row_titles=["signals", "FFT", "FC", "dFC", "TH-inputs"],
#                             column_titles=col_titles, shared_yaxes=True, subplot_titles=sp_titles)
#
#         bar_stim = np.max([np.max(np.abs(set_[-1])) for set_ in output])
#
#         for i, g in enumerate(g_sel):
#
#             sl = True if i < 1 else False
#
#             # Unpack output
#             signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, stimulus = output[i]
#
#             freqs = np.arange(len(signals[0]) / 2)
#             freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs
#
#             cmap = px.colors.qualitative.Plotly
#             for ii, signal in enumerate(signals):
#                 # Timeseries
#                 fig.add_trace(go.Scatter(x=timepoints[:5000] / 1000, y=signal[:5000], name=regionLabels[ii],
#                                          legendgroup=regionLabels[ii],
#                                          showlegend=sl, marker_color=cmap[ii % len(cmap)]), row=1, col=i + 2)
#                 # Spectra
#                 freqRange = [2, 40]
#                 fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
#                 fft = np.asarray(
#                     fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
#                 fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies
#                 fig.add_trace(go.Scatter(x=freqs[(freqs > freqRange[0]) & (freqs < freqRange[1])], y=fft,
#                                          marker_color=cmap[ii % len(cmap)], name=regionLabels[ii],
#                                          legendgroup=regionLabels[ii], showlegend=False), row=2, col=i + 2)
#
#             # Functional Connectivity
#             fig.add_trace(go.Heatmap(z=plv, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
#                                      colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=i + 2)
#
#             # dynamical Fuctional Connectivity
#             step = 2
#             fig.add_trace(go.Heatmap(z=dplv, x=np.arange(transient / 1000, len(dplv) * step, step),
#                                      y=np.arange(transient / 1000, len(dplv) * step, step), colorscale='Viridis',
#                                      colorbar=dict(thickness=8, len=0.25, y=0.2, yanchor="bottom"),
#                                      showscale=sl, zmin=0, zmax=1), row=4, col=i + 2)
#
#             # stimulation pattern
#             fig.add_trace(go.Heatmap(z=stimulus, x=timepoints/1000, y=list(range(len(stimulus))),
#                                      colorbar=dict(thickness=8, len=0.15, y=-0.02, yanchor="bottom"),
#                                      colorscale='IceFire', reversescale=True, zmin=-bar_stim, zmax=bar_stim), row=5, col=i+2)
#
#         # empirical FC matrices
#         fig.add_trace(go.Heatmap(z=plv_emp, x=regionLabels, y=regionLabels, colorbar=dict(thickness=4),
#                                  colorscale='Viridis', showscale=False, zmin=0, zmax=1), row=3, col=1)
#
#         # dynamical Fuctional Connectivity
#         dFC_emp = dFC_emp[:len(dplv)][:, :len(dplv)]
#         fig.add_trace(go.Heatmap(z=dFC_emp, x=np.arange(transient / 1000, len(dFC_emp) * step, step),
#                                  y=np.arange(transient / 1000, len(dplv) * step, step), colorscale='Viridis',
#                                  showscale=False, zmin=0, zmax=1), row=4, col=1)
#
#         w_ = 800 if n_g < 3 else 1000
#         fig.update_layout(legend=dict(yanchor="top", y=1.05, tracegroupgap=1),
#                           template="plotly_white", height=1100, width=w_)
#
#         # Update layout
#         for col in range(n_g + 1):  # +1 empirical column
#             # first row
#             idx = col + 1  # +1 to avoid 0 indexing in python
#             if idx > 1:
#                 fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Time (s)"}
#                 if idx == 2:
#                     fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Voltage (mV)"}
#
#             # second row
#             idx = 1 * (n_g + 1) + (col + 1)  # +1 to avoid 0 indexing in python
#             if idx > 1 + n_g:
#                 fig["layout"]["xaxis" + str(idx)]["title"] = {'text': "Frequency (Hz)"}
#                 if idx == 3 + n_g:
#                     fig["layout"]["yaxis" + str(idx)]["title"] = {'text': "Power (dB)"}
#
#             # third row
#             # idx = 2 * n_g+1 + (col+1)  # +1 to avoid 0 indexing in python
#             # fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
#             # fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'masdfasde (mV)'}
#
#             # fourth row
#             idx = 3 * (n_g + 1) + (col + 1)  # +1 to avoid 0 indexing in python
#             fig["layout"]["xaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
#             if idx == (3 * (n_g + 1) + 1):
#                 fig["layout"]["yaxis" + str(idx)]["title"] = {'text': 'Time (s)'}
#
#         if mode == "html":
#             pio.write_html(fig, file=folder + "/PAPER3_g_explore.html", auto_open=True)
#         elif mode == "png":
#             pio.write_image(fig, file=folder + "/g_explore" + str(time.time()) + ".png", engine="kaleido")
#         elif mode == "svg":
#             pio.write_image(fig, file=folder + "/g_explore.svg", engine="kaleido")
#
#         elif mode == "inline":
#             plotly.offline.iplot(fig)
#
