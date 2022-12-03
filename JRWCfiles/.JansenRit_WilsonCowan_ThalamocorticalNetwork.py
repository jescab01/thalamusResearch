
import os
import time
import subprocess

import numpy as np
import scipy.signal
import pandas as pd
import scipy.stats

from tvb.simulator.lab import *
from mne import time_frequency, filter
import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio

from tvb.simulator.models.JansenRit_WilsonCowan import JansenRit_WilsonCowan

import sys
sys.path.append("E://LCCN_Local/PycharmProjects/")
from toolbox.signals import timeseriesPlot
from toolbox.fft import multitapper
# from toolbox import timeseriesPlot, FFTplot, FFTpeaks, AEC, PLV, PLI, epochingTool, paramSpace


# This simulation will generate FC for a virtual "Subject".
# Define identifier (i.e. could be 0,1,11,12,...)

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"

emp_subj = "NEMOS_035"


tic0 = time.time()

simLength = 5000  # ms - relatively long simulation to be able to check for power distribution
samplingFreq = 1000  # Hz
transient = 1000  # ms to exclude from timeseries due to initial transient

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh_pass.zip")
conn.weights = conn.scaled_weights(mode="tract")

conn.speed = np.array([12.5])

jrMask_wc = [[False] if 'Thal' in roi else [True] for roi in conn.region_labels]


## MODEL
m = JansenRit_WilsonCowan(

    # Jansen-Rit nodes parameters. From Stefanovski et al. (2019)
    He=np.array([3.25]), Hi=np.array([22]),
    tau_e=np.array([10]), tau_i=np.array([20]),
    c=np.array([135.0]), p=np.array([0.09]),
    c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
    c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
    v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),

    # Wilson-Cowan nodes parameters. From Abeysuriya et al. (2018)
    P=np.array([0.4]), sigma=np.array([0.15]), Q=np.array([0]),
    c_ee=np.array([3.25]), c_ei=np.array([2.5]),
    c_ie=np.array([3.75]), c_ii=np.array([0]),
    tau_e_wc=np.array([10]), tau_i_wc=np.array([20]),
    a_e=np.array([4]), a_i=np.array([4]),
    b_e=np.array([1]), b_i=np.array([1]),
    c_e=np.array([1]), c_i=np.array([1]),
    k_e=np.array([1]), k_i=np.array([1]),
    r_e=np.array([0]), r_i=np.array([0]),
    theta_e=np.array([0]), theta_i=np.array([0]),
    alpha_e=np.array([1]), alpha_i=np.array([1]),

    # JR mask | WC mask
    jrMask_wc=np.asarray(jrMask_wc)

)
m.He, m.Hi = np.array([32.5 / m.tau_e]), np.array([440 / m.tau_i])


## COUPLING
coup = coupling.SigmoidalJansenRit_Linear(

    # Jansen-Rit Sigmoidal coupling
    a=np.array([0]), e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]),

    # Wilson-Cowan Linear coupling
    a_linear=np.asarray([0.0]),

    # JR mask | WC mask
    jrMask_wc=np.asarray(jrMask_wc)

)

# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)

# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
sim.configure()

output = sim.run(simulation_length=simLength)
print("Simulation time: %0.2f sec" % (time.time() - tic0,))
# Extract gexplore_data cutting initial transient

# Wilson-Cowan outputs
firing_data = output[0][1][transient:, 3, :, 0].T
# Jansen-Rit outputs
sum_vExc_vInh = output[0][1][transient:, 1, :, 0].T - output[0][1][transient:, 2, :, 0].T


firing_data[np.squeeze(jrMask_wc)] = m.e0 / (1.0 + np.exp(m.r * (m.v0 - sum_vExc_vInh[np.squeeze(jrMask_wc), :])))
cortex_mVdata = sum_vExc_vInh[np.squeeze(jrMask_wc), :]

raw_time = output[0][0][transient:]
regionLabels = conn.region_labels

# Check initial transient and cut gexplore_data
timeseriesPlot(firing_data, raw_time, regionLabels, title="Thalamus_firing", folder="E:\LCCN_Local\PycharmProjects\\thalamusResearch\\figures")
multitapper(firing_data, samplingFreq, regionLabels, title="Thalamus_firing", plot=True, folder="E:\LCCN_Local\PycharmProjects\\thalamusResearch\\figures")

timeseriesPlot(cortex_mVdata, raw_time, regionLabels[np.squeeze(jrMask_wc)], title="cortical_mV", folder="E:\LCCN_Local\PycharmProjects\\thalamusResearch\\figures")
#
# # Fourier Analysis plot
multitapper(cortex_mVdata, samplingFreq, regionLabels[np.squeeze(jrMask_wc)], title="cortical_mV", plot=True, folder="E:\LCCN_Local\PycharmProjects\\thalamusResearch\\figures")
