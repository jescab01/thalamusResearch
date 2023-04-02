'''

Final model is:

Y = 0.08981 + g * (-0.000332632750063 degree_rel + -0.003015538178012 degree_fromth_rel)
'''

import time
import os
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

import pickle
from report.functions import g_explore, simulate
from tvb.simulator.lab import connectivity


data_folder = "E:\LCCN_Local\PycharmProjects\\thalamusResearch\PAPER\R4.3_phetero-cx\data\\"
sim_tag = "pHetero-SUBJECTs_m12d08y2022-t11h.19m.24s"
table = pd.read_pickle(data_folder + sim_tag + "/.1pHeteroTABLE-SUBJECTS.pkl")


subj_ids = [58, 59, 64, 65, 71, 75, 77]  # Validate just with testing subjects
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

plot_subj = ["NEMOS_077", 2]  ## Define a subject to plot

th = "pTh"


def DCoffset_validation(subject, g, table, plot_data=False):
    dc_offsets, signals = [], []

    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"

    table_sub = table.loc[(table["subject"] == subject) & (table["g"] == g)]

    # Simulate ::
    conn = connectivity.Connectivity.from_file(ctb_folder + subject + "_AAL2pTh_pass.zip")

    ## 1. With pth and sigmath calibrated model
    raw_data, raw_time, _, _, _, _, regionLabels, _, _ = \
        simulate(subject, "jr", g=g, p_th=0.15, sigma=0.15, mode="sim")

    # computa las medias de cada señal
    signals_avg = np.average(raw_data, axis=1)
    glob_avg = np.average(signals_avg)

    glob_diff = np.sum(np.abs(signals_avg - glob_avg))
    dc_offsets.append(["base", subject, g, glob_diff])

    if plot_data:
        signals.append(["base", subject, g, raw_data, raw_time, regionLabels])

    ## 2. (1) + MLR pcx
    p_predicted = [0.09 + g * (-0.0003 * (table_sub["degree"].loc[table_sub["roi"] == roi].values[0] -
                                          table_sub["degree_avg"].loc[table_sub["roi"] == roi].values[0]) +
                               -0.003 * (table_sub["degree_fromth"].loc[table_sub["roi"] == roi].values[0] -
                                         table_sub["degree_fromth_avg"].loc[table_sub["roi"] == roi].values[0]))
                   if roi in table_sub["roi"].values else 0.15 for roi in conn.region_labels]
    raw_data, raw_time, _, _, _, _, regionLabels, _, _ = \
        simulate(subject, "jr", g=g, sigma=0.15, p_array=p_predicted)

    # computa DC-offsets
    signals_avg = np.average(raw_data, axis=1)
    glob_avg = np.average(signals_avg)

    glob_diff = np.sum(np.abs(signals_avg - glob_avg))
    dc_offsets.append(["MLR", subject, g, glob_diff])

    if plot_data:
        signals.append(["MLR", subject, g, raw_data, raw_time, regionLabels])

    ## 3. (1) + GD pcx
    p_adjusted = [
        table_sub["p_adjusted"].loc[table_sub["roi"] == roi].values[0] if roi in table_sub["roi"].values else 0.15 for
        roi in conn.region_labels]
    raw_data, raw_time, _, _, _, _, regionLabels, _, _ = \
        simulate(subject, "jr", g=g, sigma=0.15, p_array=p_adjusted)

    # computa DC-offsets
    signals_avg = np.average(raw_data, axis=1)
    glob_avg = np.average(signals_avg)

    glob_diff = np.sum(np.abs(signals_avg - glob_avg))
    dc_offsets.append(["GD", subject, g, glob_diff])

    if plot_data:
        signals.append(["GD", subject, g, raw_data, raw_time, regionLabels])

    if plot_data:
        return dc_offsets, signals
    else:
        return dc_offsets


# # 1. Get data for plotting DC-offset improvements
# plot, dc_offsets = [], pd.DataFrame
# for subject in sorted(set(subjects)):
#     for g in np.arange(0.5, 3, 0.5):
#         print("subject%s - g%0.2f" % (subject, g))
#         dc_offsets += DCoffset_validation(subject, g, table, plot_data=False)
#
# ## Save simulations results
# dc_offsets = pd.DataFrame(dc_offsets, columns=["mode", "subj", "g", "offset"])
# dc_offsets.to_csv(data_folder + sim_tag + "/.2GD-MLR-Base_validation_OFFSETs.csv")

dc_offsets = pd.read_csv(data_folder + sim_tag + "/.2GD-MLR-Base_validation_OFFSETs.csv")

# 2. PLOTs

# subject, g = "NEMOS_077", 2
# # for subject in sorted(set(subjects)):
# print("subject%s - g%i" % (subject, g))
# dcoff_temp, plot_data = DCoffset_validation(subject, g, table, plot_data=True)
#
# with open(data_folder + sim_tag +"/.2GD-MLR-Base_validation_DATA.pkl", "wb") as file:
#     pickle.dump(plot_data, file)

with open(data_folder + sim_tag +"/.2GD-MLR-Base_validation_DATA.pkl", "rb") as file:
    plot_data = pickle.load(file)

fig = make_subplots(rows=3, cols=2, column_widths=[0.2, 0.8], horizontal_spacing=0.2,
                    specs=[[{"rowspan": 3}, {}], [{}, {}], [{}, {}]], row_titles=["Base", "MLR", "GD"])

# Add box plot
fig.add_trace(go.Box(x=dc_offsets["mode"].values, y=dc_offsets.offset.values, showlegend=False,
                     marker_color="lightgray", opacity=0.9), row=1, col=1)

# Add signals
cmap = px.colors.qualitative.Plotly
for r, out in enumerate(plot_data):
    sl = True if r == 0 else False
    signals, timepoints, regionLabels = out[3:6]

    c = 0
    for ii, signal in enumerate(signals):
        if ii in np.arange(0, len(signals), 10):
            fig.add_trace(go.Scatter(x=timepoints[:5000]/1000, y=signal[:5000], name=regionLabels[ii],
                                     legendgroup=regionLabels[ii], showlegend=sl, marker=dict(color=cmap[c % len(cmap)]), opacity=0.7, line=dict(width=1)),
                          row=1+r, col=2)
            c+=1

fig.update_layout(template="plotly_white", legend=dict(x=1.05, y=0.5), height=500, width=900,
                  yaxis1=dict(title="DC-offset (mV)"),
                  yaxis2=dict(title="Voltage (mV)"),
                  yaxis4=dict(title="Voltage (mV)"),
                  yaxis6=dict(title="Voltage (mV)"),  xaxis6=dict(title="Time (s)"))

pio.write_html(fig, file=data_folder + sim_tag + "/.2GD-MLR-Base_validation_OFFSETs.html", auto_open=True)
pio.write_image(fig, file=data_folder + sim_tag + "/.2GD-MLR-Base_validation_OFFSETs.svg")

