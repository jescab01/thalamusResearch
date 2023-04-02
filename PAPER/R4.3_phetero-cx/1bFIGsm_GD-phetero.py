
import os
import pickle
import numpy as np
from report.functions import p_adjust
from tvb.simulator.lab import connectivity
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# Load and plot already computed simulations.
data_folder = "E:\LCCN_Local\PycharmProjects\\thalamusResearch\PAPER\R4.3_phetero-cx\data\\"
simulation_tag = "pHetero-SUBJECTs_m12d08y2022-t11h.19m.24s\\"
with open(data_folder + simulation_tag + ".1pHeteroFULL-SUBJECTS.pkl", "rb") as input_file:
    full_results = pickle.load(input_file)

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"


for emp_subj, th, g, output in full_results:
    # STRUCTURAL CONNECTIVITY      #########################################
    # Use "pass" for subcortical (thalamus) while "end" for cortex
    # based on [https://groups.google.com/g/dsi-studio/c/-naReaw7T9E/m/7a-Y1hxdCAAJ]
    n2i_indexes = []  # not to include indexes

    # Thalamus structure
    if th == 'pTh':
        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh_pass.zip")
    else:
        conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
        if th == 'woTh':
            n2i_indexes = n2i_indexes + [i for i, roi in enumerate(conn.region_labels) if 'Thal' in roi]

    indexes = [i for i, roi in enumerate(conn.region_labels) if i not in n2i_indexes]
    conn.region_labels = conn.region_labels[indexes]

    conn.weights = conn.scaled_weights(mode="tract")

    # load text with FC rois; check if match SC
    SClabs = list(conn.region_labels)
    SC_notTh_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" not in roi]

    p_array, signals, p_array_init, signals_init, timepoints, regionLabels, degree, degree_avg, degree_fromth, degree_fromth_avg, results = output

    cmap = px.colors.qualitative.Plotly
    fig = make_subplots(rows=3, cols=2, specs=[[{}, {}], [{"colspan": 2}, {}], [{}, {}]], horizontal_spacing=0.15,
                        subplot_titles=["Signals - Pre", "Signals - Post", "Gradient descent error", "", "Corr w/ indegree", "Corr w/ th inputs"])

    # plot all signals
    for c, ii in enumerate(SC_notTh_idx[::10]):
        # Timeseries
        fig.add_trace(go.Scatter(x=timepoints / 1000, y=signals_init[ii, :], name=regionLabels[ii], opacity=0.8,
                                 legendgroup=regionLabels[ii], mode="lines", line=dict(color=cmap[c%len(cmap)], width=1)), row=1, col=1)
        # Timeseries
        fig.add_trace(go.Scatter(x=timepoints / 1000, y=signals[ii, :], name=regionLabels[ii], opacity=0.8,
                                 legendgroup=regionLabels[ii], mode="lines", line=dict(color=cmap[c%len(cmap)], width=1), showlegend=False), row=1, col=2)

    # PLOT Gradient descent error
    fig.add_trace(go.Scatter(x=results[:, 1], y=results[:, 0], showlegend=False, line=dict(color="black")), row=2, col=1)

    # PLOT connections vs p_array
    fig.add_trace(go.Scatter(x=np.sum(conn.weights[SC_notTh_idx], axis=1), y=p_array[SC_notTh_idx],
                             mode="markers", marker=dict(color="darkgray", opacity=0.6), showlegend=False), row=3, col=1)

    # PLOT connections to thalamus vs p_array
    SC_Th_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" in roi]
    fig.add_trace(go.Scatter(x=np.sum(conn.weights[SC_notTh_idx, :][:, SC_Th_idx], axis=1), y=p_array[SC_notTh_idx],
                             mode="markers", marker=dict(color="darkgray", opacity=0.6), showlegend=False), row=3, col=2)

    fig.update_layout(template="plotly_white", legend=dict(tracegroupgap=1, y=1.05), width=1000, height=900,
                      xaxis1=dict(title="Time (ms)"), yaxis1=dict(title="Voltage (mV)"),
                      xaxis2=dict(title="Time (ms)"), yaxis2=dict(title="Voltage (mV)"),
                      xaxis3=dict(title="Iteration"), yaxis3=dict(title=r"$\sum_{i=1}^{N} |\overline{s_i} - \overline{S}|$"),
                      xaxis5=dict(title="Node indegree"), yaxis5=dict(title=r"$p_{\neq th} \text{ adjusted}$"),
                      xaxis6=dict(title="Node indegree (from thalamus)"), yaxis6=dict(title=r"$p_{\neq th} \text{ adjusted}$"))

    pio.write_image(fig, file=data_folder + simulation_tag + "prepostGD_" + emp_subj + "_" + th + "_g" + str(g) + ".svg")
    pio.write_html(fig, file= data_folder + simulation_tag + "prepostGD_" + emp_subj + "_" + th + "_g" + str(g) + ".html",
                   auto_open=False, include_mathjax="cdn")

