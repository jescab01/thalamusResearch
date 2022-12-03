
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.

from report.functions import simulate
import plotly.graph_objects as go
from tvb.simulator.lab import connectivity
import numpy as np

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2pTh_pass.zip")

# load text with FC rois; check if match SC
SClabs = list(conn.region_labels)
SC_notTh_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" not in roi]

# Initial p_array and init simulation
p_array = np.asarray([0.22 if 'Thal' in roi else 0.09 for roi in conn.region_labels])
signals, timepoints, regionLabels = simulate("NEMOS_035", "jr", g=3, p_array=p_array, sigma=0.15, th='pTh', t=6, mode="pHetero")


fig = go.Figure()
for ii in range(len(signals)):
    # Timeseries
    fig.add_trace(go.Scatter(x=timepoints / 1000, y=signals[ii, :], name=regionLabels[ii], legendgroup=regionLabels[ii]))
fig.show("browser")


## Loop to get heterogeneous p
results = []

for i in range(50):

    print("Iteration %i" % i)
    ps_cx = p_array[SC_notTh_idx]

    # computa las medias de cada señal
    signals_avg = np.average(signals[SC_notTh_idx, :], axis=1)

    # computa las medias de las medias que sera el punto que equivaldrá a 0.09
    glob_avg = np.average(signals_avg)

    # diff
    diffs = signals_avg - glob_avg
    glob_diff = np.sum(np.abs(diffs))

    # haz una regla de actualizacion de los ps que lleve a los que tienen media por encima
    p_array[SC_notTh_idx] = ps_cx - diffs

    signals, timepoints, regionLabels = simulate("NEMOS_035", "jr", g=3, p_array=p_array, sigma=0.15, th='pTh', t=6, mode="pHetero")

    print(p_array)
    print("Global difference %0.5f" % glob_diff)
    results.append([glob_diff, i])



### PLOTS
## Plot result: decay of difference
results_ = np.asarray(results)
fig = go.Figure(go.Scatter(x=results_[:, 1], y=results_[:, 0]))
fig.show("browser")

# plot all signals
fig = go.Figure()
for ii in range(len(signals)):
    # Timeseries
    fig.add_trace(go.Scatter(x=timepoints / 1000, y=signals[ii, :], name=regionLabels[ii], legendgroup=regionLabels[ii]))
fig.show("browser")

# plot not Th signals
fig = go.Figure()
for ii in SC_notTh_idx:
    # Timeseries
    fig.add_trace(go.Scatter(x=timepoints / 1000, y=signals[ii, :], name=regionLabels[ii], legendgroup=regionLabels[ii]))
fig.show("browser")


# compute a linear model between indegree and final p
import pingouin as pg
pg.corr(np.sum(conn.weights[SC_notTh_idx], axis=1), p_array[SC_notTh_idx])

# PLOT indegree vs p_array
fig = go.Figure()
fig.add_scatter(x=np.sum(conn.weights[SC_notTh_idx], axis=1), y=p_array[SC_notTh_idx], mode="markers")
fig.show("browser")

# PLOT connections to thalamus vs p_array
SC_Th_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" in roi]
fig = go.Figure()
fig.add_scatter(x=np.sum(conn.weights[SC_notTh_idx, :][:, SC_Th_idx], axis=1), y=p_array[SC_notTh_idx], mode="markers")
fig.show("browser")