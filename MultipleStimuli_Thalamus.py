
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
simLength = 30 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 2000  # ms

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

# # Subset for Cingulum Bundle
# if "cb" in mode:
#     FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
#     SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in FClabs that matches cortical_rois
#     conn.weights = conn.weights[:, SC_cb_idx][SC_cb_idx]
#     conn.tract_lengths = conn.tract_lengths[:, SC_cb_idx][SC_cb_idx]
#     conn.region_labels = conn.region_labels[SC_cb_idx]

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
stimulus_type = 'noise'

# Where
where = 'random_th'

state1_weighting = np.asarray([np.round(np.random.random()) if "Thal" in roi else 0 for roi in conn.region_labels])
state2_weighting = np.asarray([np.round(np.random.random()) if "Thal" in roi else 0 for roi in conn.region_labels])

# When
when = {"state1_start": 5000, "state1_end": 10000,
        "state2_start": 15000, "state2_end": 20000}

# What
what = {"dc_offset": 0.05, "sin_amp": 0.05, "noise_std": 1}


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

# Extract gexplore_data: "output[a][b][:,0,:,0].T" where:
# a=monitorIndex, b=(gexplore_data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
if model == "jrd":
    raw_data = m.w * (output[0][1][transient:, 0, :, 0].T - output[0][1][transient:, 1, :, 0].T) + \
               (1 - m.w) * (output[0][1][transient:, 3, :, 0].T - output[0][1][transient:, 4, :, 0].T)
else:
    raw_data = output[0][1][transient:, 0, :, 0].T
raw_time = output[0][0][transient:]
regionLabels = conn.region_labels

# Extract signals of interest ## BE AWARE of the NOT
# if "cb" not in mode:
raw_data = raw_data[SC_cortex_idx, :]

# Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
_, _, IAF, module, band_module = multitapper(raw_data, samplingFreq, regionLabels, peaks=True, plot=True, folder="E:\LCCN_Local\PycharmProjects\\thalamusResearch\\figures\\")


bands = [["3-alpha"], [(8, 12)]]
# bands = [["1-delta", "2-theta", "3-alpha", "4-beta", "5-gamma"], [(2, 4), (5, 7), (8, 12), (15, 29), (30, 59)]]

for b in range(len(bands[0])):
    (lowcut, highcut) = bands[1][b]

    # Band-pass filtering
    filterSignals = filter.filter_data(raw_data, samplingFreq, lowcut, highcut)

    # EPOCHING timeseries into x seconds windows epochingTool(signals, windowlength(s), samplingFrequency(Hz))
    efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals")

    # Obtain Analytical signal
    efPhase = list()
    efEnvelope = list()
    for i in range(len(efSignals)):
        analyticalSignal = scipy.signal.hilbert(efSignals[i])
        # Get instantaneous phase and amplitude envelope by channel
        efPhase.append(np.angle(analyticalSignal))
        efEnvelope.append(np.abs(analyticalSignal))

    # Check point
    # from toolbox import timeseriesPlot, plotConversions
    # regionLabels = conn.region_labels
    timeseriesPlot(raw_data, raw_time, regionLabels, folder="E:\LCCN_Local\PycharmProjects\\thalamusResearch\\figures")
    # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

    # CONNECTIVITY MEASURES
    ## PLV
    plv = PLV(efPhase)

    plv_emp = \
        np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
        FC_cortex_idx][
            FC_cortex_idx]

    # Comparisons
    t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
    t1[0, :] = plv[np.triu_indices(len(plv), 1)]
    t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
    plv_r = np.corrcoef(t1)[0, 1]

    print("rPLV = %0.2f" % plv_r)

    ## dynamical Functional Connectivity
    # Sliding window parameters
    window, step = 1, 0.5  # seconds

    ## dFC
    dFC = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV", verbose=False,
                     folder="E:\LCCN_Local\PycharmProjects\\thalamusResearch\\figures", plot="ON", auto_open=True)

    # dFC_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_dPLV4s.txt")
    #
    # # Compare dFC vs dFC_emp
    # t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
    # t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
    # t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
    # dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

    # ## Metastability: Kuramoto Order Parameter
    # ko = kuramoto_order(raw_data[sc_rois_cortex, :], samplingFreq)
    # ko_emp = np.loadtxt(ctb_folder + "FC_" + emp_subj + "/" + bands[0][b] + "_KO.txt")

    print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))




