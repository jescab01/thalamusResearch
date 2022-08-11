
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from report.functions import simulate, g_explore, configure_states
from tvb.simulator.lab import connectivity
import matplotlib.pyplot as plt

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2pTh_pass.zip")

stimulus = configure_states("sinusoid", 0.5, 5, 100, 0.25, conn, 20000, False)

pattern = stimulus()
plt.imshow(pattern, interpolation='none', aspect='auto')
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar()


import pandas as pd

a = pd.read_csv("E:\LCCN_Local\PycharmProjects\\thalamusResearch\mpi_jr-vMultiStim\PSE\PSEmpi_MultiStim-m08d10y2022-t16h.22m.52s\\results.csv")



a.to_pickle("asdfa.pkl")

b=pd.read_pickle("asdfa.pkl")

output = [simulate("NEMOS_035", "jr", g=3, p_th=0.22, sigma=0.15, th='pTh', t=10, stimulate=False)]
g_explore(output, [3])



stim = ["sinusoid", 1, 5, 0.25, [[0,10000], [10000,15000], [20000,23000], [28000,45000],[50000,51000]]]  # stim_type, gain, nstates, tstates, pinclusion, deterministic

output = [simulate("NEMOS_035", "jr", g=3, p_th=0.22, sigma=0.15, th='pTh', t=60, stimulate=stim)]

g_explore(output, [3])


stim = ["sinusoid", 0.5, 5, 100, 0.25, False]  # stim_type, gain, nstates, tstates(ms), pinclusion, deterministic

output = [simulate("NEMOS_035", "jr", g=3, p_th=0.22, sigma=0.15, th='pTh', t=30, stimulate=stim)]

g_explore(output, [3])



#
# # Simulate :: NEMOS_035 - may take a while
# output = []
# g_sel = [4, 9, 15, 37, 60]
# for g in g_sel:
#     output.append(simulate("NEMOS_035", "jr", g=g, th='pTh', t=60))
#
#
#
#
# # Plot FC and dFC matrices. Add empirical ones as reference.
# plv_dplv(plv, dFC, regionLabels, transient=transient/1000, step=2, mode="inline")  # Simulated
# plv_dplv(plv_emp, dFC_emp, regionLabels, step=2, mode="inline", title="Empirical matrices for NEMOS_035")  # Empirical ref.



