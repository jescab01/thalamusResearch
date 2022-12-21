
import time
import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt

from tvb.simulator.lab import *
import tvb.datatypes.projections as projections
from mne import filter
from tvb.simulator.models.jansen_rit_david_mine import JansenRitDavid2003, JansenRit1995
import datetime


## Folder structure - Local
if "LCCN_Local" in os.getcwd():
    ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"
    import sys
    sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
    from toolbox.fft import multitapper
    from toolbox.signals import epochingTool, timeseriesPlot
    from toolbox.fc import PLV
    from toolbox.dynamics import dynamic_fc
    from toolbox.mixes import timeseries_spectra

## Folder structure - CLUSTER
else:
    from toolbox import multitapper, PLV, epochingTool
    wd = "/home/t192/t192950/mpi/"
    ctb_folder = wd + "CTB_data3/"


## Define working points per subject


# Prepare simulation parameters
simLength = 10 * 1000  # ms
samplingFreq = 1000  # Hz
transient = 2000  # ms

emp_subj, model, th, g, s = "NEMOS_035", "jr", "pTh", 15, 15

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


# load text with FC rois; check if match SC
FClabs = list(np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/roi_labels.txt", dtype=str))
FC_cortex_idx = [FClabs.index(roi) for roi in
                 cortical_rois]  # find indexes in FClabs that matches cortical_rois
SClabs = list(conn.region_labels)
SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]


# NEURAL MASS MODEL    #########################################################
sigma_array = np.asarray([100 if 'Thal' in roi else 0 for roi in conn.region_labels])
p_array = np.asarray([0.15 if 'Thal' in roi else 0.09 for roi in conn.region_labels])
# taui_sel = 10
# taui_array = np.asarray([taui_sel if 'Precentral' in roi else 20 for roi in conn.region_labels])

if model == "jrd":  # JANSEN-RIT-DAVID

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

    coup = coupling.SigmoidalJansenRitDavid(a=np.array([g]), w=np.array([0.8]), e0=np.array([0.005]),
                                            v0=np.array([6.0]), r=np.array([0.56]))

else:  # JANSEN-RIT
    # Parameters from Stefanovski 2019. Good working point at g=33, s=15.5 on AAL2red connectome.
    m = JansenRit1995(He=np.array([3.25]), Hi=np.array([22]),
                      tau_e=np.array([10]), tau_i=np.array([20]),
                      c=np.array([1]), c_pyr2exc=np.array([135]), c_exc2pyr=np.array([108]),
                      c_pyr2inh=np.array([33.75]), c_inh2pyr=np.array([33.75]),
                      p=np.array([p_array]), sigma=np.array([sigma_array]),
                      e0=np.array([0.005]), r=np.array([0.56]), v0=np.array([6]))

    coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                       r=np.array([0.56]))

conn.speed = np.array([s])

# OTHER PARAMETERS   ###
# integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
# integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)



# Region mapping indices come from Brainstorm; change them to match SC indices.
# rm_pre = region_mapping.RegionMapping.from_file(ctb_folder + "tvbMonitor-EEG65_ICBM152_AAL2pth\\regionmapping-AAL2pth_toICBM152.txt")
# rm_post = region_mapping.RegionMapping.from_file(ctb_folder + "tvbMonitor-EEG65_ICBM152_AAL2pth\\regionmapping-AAL2pth_toICBM152.txt")
# # load text with FC rois; check if match SC
# FClabs = list(np.loadtxt(ctb_folder + "tvbMonitor-EEG65_ICBM152_AAL2pth\\roi_labels.txt", dtype=str))
# SC_fromFC_idx = [list(conn.region_labels).index(roi) if roi in conn.region_labels else "nan" for roi in FClabs]  # find indexes in FClabs that matches cortical_rois

# for i, sc_idx in enumerate(SC_fromFC_idx):
#     rm_post.array_data[rm_pre.array_data == i] = sc_idx
#
# ## update conn
# conn.region_labels=conn.region_labels[SC_fromFC_idx]
# conn.weights=conn.weights[:, SC_fromFC_idx][SC_fromFC_idx]
# conn.tract_lengths = conn.tract_lengths[:, SC_fromFC_idx][SC_fromFC_idx]
# conn.centres = conn.centres[SC_fromFC_idx]
# conn.cortical = conn.cortical[SC_fromFC_idx]
#
# pr = projections.ProjectionSurfaceEEG.from_file(ctb_folder + "tvbMonitor-EEG65_ICBM152_AAL2pth\\headmodel-ICBM152_toEEG65.mat")
# ss = sensors.SensorsEEG.from_file(source_file=ctb_folder + "tvbMonitor-EEG65_ICBM152_AAL2pth\\sensors-EEG65.txt")
#
# mon = (monitors.Raw(), monitors.EEG(projection=pr, sensors=ss, region_mapping=rm_post))
mon = (monitors.Raw(),)

print("Simulating %s (%is)  ||  PARAMS: g%i s%i" % (model, simLength / 1000, g, s))


# Run simulation
sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
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


## Plot
timeseries_spectra(raw_data, simLength, transient, regionLabels, mode="html", folder="figures",
                       freqRange=[1, 35], opacity=0.8, title="sigma10", auto_open=True)


