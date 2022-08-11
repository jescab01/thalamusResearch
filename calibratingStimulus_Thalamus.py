
import time
import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt

from tvb.simulator.lab import *
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003
import datetime


## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool, timeseriesPlot
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc

## Folder structure - CLUSTER
else:
    from toolbox import multitapper, PLV, epochingTool
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data2/"


## Define working points per subject


# Prepare simulation parameters
simLength = 5 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 500  # ms

emp_subj, model, th, g, s = "NEMOS_035", "jr", "wpTh", 9, 9.5

tic = time.time()


# STRUCTURAL CONNECTIVITY      #########################################
n2i_indexes = []  # not to include indexes
# Thalamus structure
if 'pTh' in th:
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2pTh_pass.zip")
else:
    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")
    if 'woTh' in th:
        n2i_indexes = n2i_indexes + [i for i, roi in enumerate(conn.region_labels) if 'Thal' in roi]

indexes = [i for i, roi in enumerate(conn.region_labels) if i not in n2i_indexes]
conn.weights = conn.weights[:, indexes][indexes]
conn.tract_lengths = conn.tract_lengths[:, indexes][indexes]
conn.region_labels = conn.region_labels[indexes]
conn.weights = conn.scaled_weights(mode="tract")

# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                 'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                 'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                 'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                 'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Calcarine_L',
                 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                 'Temporal_Inf_R']
cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                 'Thalamus_L', 'Thalamus_R']

# load text with FC rois; check if match SC
FClabs = list(np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/roi_labels_rms.txt", dtype=str))
FC_cortex_idx = [FClabs.index(roi) for roi in
                 cortical_rois]  # find indexes in FClabs that matches cortical_rois
SClabs = list(conn.region_labels)
SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]


# NEURAL MASS MODEL    #########################################################
if model == "jrd":  # JANSEN-RIT-DAVID
    # if "_def" in mode:
    #     sigma_array = 0.022
    #     p_array = 0.22
    # else:  # for jrd_pTh and jrd modes
    sigma_array = np.asarray([0.022 if 'Thal' in roi else 0 for roi in conn.region_labels])
    p_array = np.asarray([0.22 if 'Thal' in roi else 0 for roi in conn.region_labels])

    # Parameters edited from David and Friston (2003).
    m = JansenRitDavid2003(He1=np.array([3.25]), Hi1=np.array([22]),  # SLOW population
                           tau_e1=np.array([10.8]), tau_i1=np.array([22.0]),
                           He2=np.array([3.25]), Hi2=np.array([22]),  # FAST population
                           tau_e2=np.array([4.6]), tau_i2=np.array([2.9]),

                           w=np.array([0.8]), c=np.array([135.0]),
                           c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
                           c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
                           v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),
                           p=np.array([p_array]), sigma=np.array([sigma_array]))

    # Remember to hold tau*H constant.
    m.He1, m.Hi1 = np.array([32.5 / m.tau_e1]), np.array([440 / m.tau_i1])
    m.He2, m.Hi2 = np.array([32.5 / m.tau_e2]), np.array([440 / m.tau_i2])

else:  # JANSEN-RIT
    # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
    m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                         a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                         a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                         mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]),
                         p_min=np.array([0]),
                         r=np.array([0.56]), v0=np.array([6]))

# COUPLING FUNCTION   #########################################
if model == "jrd":
    coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=m.w, e0=m.e0, v0=m.v0, r=m.r)
else:
    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))
conn.speed = np.array([s])

# OTHER PARAMETERS   ###
# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

mon = (monitors.Raw(),)


#   STIMULUS      #########################
# Multiple stimuli tutorial ::
# https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_documentation/demos/multiple_stimuli.ipynb

# what type of stimulus?
stimulus_type = 'sinusoid'

# Where
where = 'random_th'

state1_weighting = np.asarray([np.round(np.random.random()) if "Thal" in roi else 0 for roi in conn.region_labels])
state2_weighting = np.asarray([np.round(np.random.random()) if "Thal" in roi else 0 for roi in conn.region_labels])

# When
when = {"state1_start": 2000, "state1_end": 3000,
        "state2_start": 3500, "state2_end": 4000}

# What
what = {"dc_offset": 0.05, "sin_amp": 0.05, "noise_std": 0.15}


# stim_params = {"type": "dc",
#                "labels": ["state1", "state2", "state1"],
#                "when": [(1000, 1500), (1300, 2000), (2500, 3500)],
#                "weighting": [state1_weighting}


if stimulus_type == "dc":

    # State 1 weighting -
    eqn_t = equations.DC()
    eqn_t.parameters.update(dc_offset=what["dc_offset"], t_start=when["state1_start"], t_end=when["state1_end"])
    state1_stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=state1_weighting)

    # State 2 weighting -
    eqn_t = equations.DC()
    eqn_t.parameters.update(dc_offset=what["dc_offset"], t_start=when["state2_start"], t_end=when["state2_end"])
    state2_stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=state2_weighting)

elif stimulus_type == "sinusoid":

    # State 1 weighting -
    eqn_t = equations.Sinusoid()
    eqn_t.parameters.update(amp=what["sin_amp"], frequency=10, onset=when["state1_start"], offset=when["state1_end"])
    state1_stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=state1_weighting)

    # State 2 weighting -
    eqn_t = equations.Sinusoid()
    eqn_t.parameters.update(amp=what["sin_amp"], frequency=10, onset=when["state2_start"], offset=when["state2_end"])
    state2_stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=state2_weighting)

elif stimulus_type == "noise":

    # # RNS
    # eqn_t = equations.Noise()
    # eqn_t.parameters["mean"] = stim_params
    # eqn_t.parameters["std"] = (1 - eqn_t.parameters[
    #     "mean"]) / 3  # p(mean<x<mean+std) = 0.34 in gaussian distribution [max=1; min=-1]
    # eqn_t.parameters["onset"] = 0
    # eqn_t.parameters["offset"] = simLength

    # State 1 weighting -
    eqn_t = equations.Noise()
    eqn_t.parameters.update(mean=0, std=what["noise_std"], onset=when["state1_start"], offset=when["state1_end"])
    state1_stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=state1_weighting)

    # State 2 weighting -
    eqn_t = equations.Noise()
    eqn_t.parameters.update(mean=0, std=what["noise_std"], onset=when["state2_start"], offset=when["state2_end"])
    state2_stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=state2_weighting)

stimulus = patterns.MultiStimuliRegion(state1_stimulus, state2_stimulus)
stimulus.configure_space()
stimulus.configure_time(np.arange(0, simLength, 1))

pattern = stimulus()
plt.imshow(pattern, interpolation='none', aspect='auto')
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar()

print("Simulating %s (%is)  ||  PARAMS: g%i s%i" % (model, simLength / 1000, g, s))


# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon, stimulus=stimulus)
sim.configure()
output = sim.run(simulation_length=simLength)

# Extract data: "output[a][b][:,0,:,0].T" where:
# a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
if model == "jrd":
    raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
               (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
else:
    raw_data = output[0][1][transient:, 0, :, 0].T
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels


timeseriesPlot(raw_data[state1_weighting==1], raw_time, regionLabels[state1_weighting==1], folder="figures", title=None, mode="html", auto_open=True)

# Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
# _, _, IAF, module, band_module = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)
#
#
# print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))




