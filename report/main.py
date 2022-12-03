
import os

import numpy as np
import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from report.functions import simulate, g_explore, configure_states
from tvb.simulator.lab import connectivity
import matplotlib.pyplot as plt

# ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
# conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2pTh_pass.zip")
#
# stimulus = configure_states("sinusoid", 0.5, 5, 100, 0.25, conn, 20000, False)
#
# pattern = stimulus()
# plt.imshow(pattern, interpolation='none', aspect='auto')
# plt.xlabel('Time')
# plt.ylabel('Space')
# plt.colorbar()



## Not deterministic stimulation
# stim = ["sinusoid", 0.5, 5, 100, 0.25, False]  # stim_type, gain, nstates, tstates(ms), pinclusion, deterministic
# output = [simulate("NEMOS_035", "jr", g=3, p_th=0.22, sigma=0.15, th='pTh', t=30, stimulate=stim)]
# g_explore(output, [3])


# Deterministic stimulation

# You set up for each state what the timing.
# We are going to fill up the desired timerange with the number of states
# defined (without superposition), randomly assigned to the timeline.

# Leave out a part of the simulation to have the global minimum: 20 seconds.
# Then, randomly assign sequential states without overlapping. Random duration.

simLength = 60  # s
nstates = 50

stim_start = 0  # ms before start of stimulation: to get the global minimum.
times = sorted(np.random.randint(stim_start, simLength*1000, nstates))
timing = [[times[i], times[i+1]] for i in range(len(times)-1)]
timing.append([times[-1], simLength*1000])


stim = ["sinusoid", 0.3, nstates, None, 0.15, timing]  # stim_type, gain, nstates, tstates, pinclusion, deterministic
output = [simulate("NEMOS_035", "jr", g=2, p_th=0.22, sigma=0.15, th='pTh', t=simLength, stimulate=stim)]

g_explore(output, [2])





# Super imposed states
simLength = 60  # s
nstates = 50

stim_start = 0  # ms before start of stimulation: to get the global minimum.
times = sorted(np.random.randint(stim_start, simLength*1000, nstates))
timing = [[times[i], times[i+2]] for i in range(len(times)-2)]
timing.append([times[-2], simLength*1000])
timing.append([times[-1], simLength*1000])

stim = ["sinusoid", 0.5, nstates, None, 0.15, timing]  # stim_type, gain, nstates, tstates, pinclusion, deterministic
output = [simulate("NEMOS_035", "jr", g=2, p_th=0.22, sigma=0.15, th='pTh', t=simLength, stimulate=stim)]

g_explore(output, [2])
