
import time
import numpy as np
import scipy.signal
import scipy.stats
import pandas as pd

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
from mpi4py import MPI
import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

result = list()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    ctb_folderOLD = "E:\\LCCN_Local\PycharmProjects\CTB_dataOLD\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order

## Folder structure - CLUSTER
else:
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data3/"
    ctb_folderOLD = wd + "CTB_dataOLD/"

    import sys
    sys.path.append(wd)
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc, kuramoto_order



# Prepare simulation parameters
simLength = 10 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 2000  # ms

subj_ids = [35] #, 49, 50, 58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]
# subjects.append("NEMOS_AVG")

coupling_vals = np.arange(0, 30, 0.5)  # 0.5


### MODE 1: th-noisy
modes_dict = {"th_noisy": [subjects[0], "jr", "pTh", "pCer", 0.15, 0.22, 0.09, 0],
              "th_noisy_phetero": [subjects[0], "jr", "pTh", "pCer", 0.15, 0.22, "MLR", 0],
              "allnoisy": [subjects[0], "jr", "pTh", "pCer", 0.15, 0.22, 0.09, 0.22],
              "classical": [subjects[0], "jr", "pTh", "pCer", 0.09, 0, 0.09, 0],
              "allnoisy_prebif": [subjects[0], "jr", "pTh", "pCer", 0.09, 0.022, 0.09, 0.022]}

for mode, args in modes_dict.items():

    ### PLOTTING
    # Load data
    results_df = pd.read_pickle("data/Criticality_" + mode + ".pkl")
    results_df.columns=["subj", "model", "th", "cer", "g", "pth", "sigmath", "pcx", "sigmacx",
                                           "signals", "time", "regionLabels", "rPLV"]

    cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2
    c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
    opacity = 0.9

    ## PLOTTING
    fig = make_subplots(rows=3, cols=3, column_widths=[0.25,0.5,0.25], row_heights=[0.25,0.25,0.5],
                        specs=[[{},{},{}],[{},{},{}],[{"colspan":3},{},{}]], column_titles=["", "Signals by g  _MODE:" + mode, ""])
    # rPLV and reference
    fig.add_trace(go.Scatter(x=results_df.g, y=results_df.rPLV, mode="lines", opacity=opacity,
                             line=dict(width=4, color=c3), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 0], y=[-0.05, 0.5], mode="lines",
                   line=dict(width=1, color="black"), showlegend=False), row=1, col=2)

    # BIFURCATION
    # Calculate the bifurcation based on the data from all simulations
    bif = pd.DataFrame([[row.g, "max", np.max(s)] for i, row in results_df.iterrows() for s in row.signals] +
                     [[row.g, "min", np.min(s)] for i, row in results_df.iterrows() for s in row.signals],
                     columns=["g", "min-max", "value"])

    bif_avg = bif.groupby(["g", "min-max"]).mean().reset_index()

    # add static bifurcation
    sub = bif_avg.loc[bif_avg["min-max"] == "max"]
    fig.add_trace(go.Scatter(x=sub.g, y=sub.value, mode="lines", line=dict(width=3, color="gray"), showlegend=False), row=2, col=2)
    sub = bif_avg.loc[bif_avg["min-max"] == "min"]
    fig.add_trace(go.Scatter(x=sub.g, y=sub.value, mode="lines", line=dict(width=3, color="gray"), showlegend=False), row=2, col=2)

    # SIGNALS and BIFURACTION reference
    skip = 10
    regionLabels = results_df["regionLabels"].loc[results_df["g"]==0].values[0][::skip]
    time = results_df["time"].loc[results_df["g"]==0].values[0]
    temp_max, temp_min = bif_avg.loc[bif_avg["min-max"]=="max"].copy(), bif_avg.loc[bif_avg["min-max"] == "min"].copy()
    for ii, roi_signals in enumerate(results_df["signals"].loc[results_df["g"]==0].values[0][::skip]):

        # define the bif_avg(g) that better matches the signal
        temp_max["diff"] = temp_max.value - max(roi_signals)
        temp_min["diff"] = temp_min.value - min(roi_signals)

        g_ref = (temp_max["g"].loc[temp_max["diff"] == temp_max["diff"].min()].values[0] +
                 temp_min["g"].loc[temp_min["diff"] == temp_min["diff"].min()].values[0])/2

        fig.add_trace(
            go.Scatter(x=[g_ref, g_ref], y=[min(roi_signals)-0.01, max(roi_signals)+0.01], mode="lines",
                       name=regionLabels[ii], legendgroup=regionLabels[ii], line=dict(width=1, color=cmap_s2[ii%len(cmap_s2)]), showlegend=False), row=2, col=2)

        ## Add signals
        fig.add_trace(go.Scatter(x=time, y=roi_signals, name=regionLabels[ii], legendgroup=regionLabels[ii], marker_color=cmap_s2[ii%len(cmap_s2)]), row=3, col=1)

    ## Make it dynamical: ADD FRAMES
    frames = []
    traces = [1] + list(range(4, len(regionLabels)*2+4))
    for g in sorted(set(results_df.g)):

        data = [go.Scatter(x=[g, g])]
        for ii, roi_signals in enumerate(results_df["signals"].loc[results_df["g"]==g].values[0][::skip]):

            # define the bif_avg(g) that better matches the signal
            temp_max["diff"] = abs(temp_max.value - max(roi_signals))
            temp_min["diff"] = abs(temp_min.value - min(roi_signals))

            g_ref = (temp_max["g"].loc[temp_max["diff"] == temp_max["diff"].min()].values[0] +
                     temp_min["g"].loc[temp_min["diff"] == temp_min["diff"].min()].values[0])/2

            ## Add references on bifurcation
            data = data + [
                go.Scatter(x=[g_ref, g_ref], y=[min(roi_signals)-0.01, max(roi_signals)+0.01]),
                go.Scatter(y=roi_signals)]

        frames.append((go.Frame(data=data, traces=traces, name=str(g))))
    fig.update(frames=frames)

    # CONTROLS : Add sliders and buttons
    sliders = [dict(steps=[dict(method='animate',
                                args=[[str(g)], dict(mode="immediate", frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                               transition=dict(duration=300))], label=str(g)) for i, g in enumerate(sorted(set(results_df.g)))],
            transition=dict(duration=100), x=0.15, xanchor="left", y=1.4,
            currentvalue=dict(font=dict(size=15), prefix="Coupling factor (g) - ", visible=True, xanchor="right"),
            len=0.8, tickcolor="white")]
    updatemenus = [dict(type="buttons", showactive=False, y=1.35, x=0, xanchor="left",
                          buttons=[dict(label="Play", method="animate",
                                   args=[None, dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                                    transition=dict(duration=300), fromcurrent=True, mode='immediate')]),
                              dict(label="Pause", method="animate", args=[[None],
                                         dict(frame=dict(duration=100, redraw=True, easing="cubic-in-out"),
                                              transition=dict(duration=300), mode="immediate")])])]
    fig.update_layout(
        template="plotly_white", legend=dict(x=0.82, y=1, tracegroupgap=5, groupclick="toggleitem"),
        xaxis2=dict(title="Coupling factor (g)"), yaxis2=dict(title="rPLV"),
        xaxis5=dict(title="Coupling factor (g)"), yaxis5=dict(title="Bifurcation<br>(signals min-max)"),
        xaxis7=dict(title="Time (ms)"), yaxis7=dict(title="Voltage (mV)", range=[bif.value.min(), bif.value.max()]),
        sliders=sliders, updatemenus=updatemenus)

    pio.write_html(fig, "data/Criticality_"+mode+".html", auto_open=True)
