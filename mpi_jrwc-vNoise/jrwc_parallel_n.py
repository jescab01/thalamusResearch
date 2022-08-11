
import time
import numpy as np
import scipy.signal
import scipy.stats

from tvb.simulator.lab import *
from tvb.simulator.models.JansenRit_WilsonCowan import JansenRit_WilsonCowan
from mne import filter
from mpi4py import MPI
import datetime


def jrwc_parallel(params_):
    result = list()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print("Hello world from rank", str(rank), "of", str(size), '__', datetime.datetime.now().strftime("%Hh:%Mm:%Ss"))

    ## Folder structure - Local
    if "LCCN_Local" in os.getcwd():
        ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
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
        ctb_folder = wd + "CTB_data2/"
        ctb_folderOLD = wd + "CTB_dataOLD/"

        import sys
        sys.path.append(wd)
        from toolbox.fft import multitapper
        from toolbox.signals import epochingTool
        from toolbox.fc import PLV
        from toolbox.dynamics import dynamic_fc, kuramoto_order

    # Prepare simulation parameters
    simLength = 40 * 1000  # ms
    samplingFreq = 1000  # Hz
    transient = 4000  # ms

    for ii, set in enumerate(params_):

        tic = time.time()
        print("Rank %i out of %i  ::  %i/%i " % (rank, size, ii + 1, len(params_)))

        print(set)
        emp_subj, th, cer, g_jr, g_wc, std_n, r = set

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

        # Cerebellum structre
        if cer == "Cer":
            cer_indexes = [i for i, roi in enumerate(conn.region_labels) if ('Cer' in roi) or ('Ver' in roi)]
            # update weight matrix: summing up cerebellum weights and averaging tract lengths
            # weights right hem
            weights_sum = np.sum(conn.weights[cer_indexes[0::2], :], axis=0)
            conn.weights[cer_indexes[0], :] = weights_sum
            conn.weights[:, cer_indexes[0]] = weights_sum
            # weights left hem
            weights_sum = np.sum(conn.weights[cer_indexes[1::2], :], axis=0)
            conn.weights[cer_indexes[1], :] = weights_sum
            conn.weights[:, cer_indexes[1]] = weights_sum

            # tract lengths right hem
            tracts_avg = np.average(conn.tract_lengths[cer_indexes[0::2], :], axis=0)
            conn.tract_lengths[cer_indexes[0], :] = tracts_avg
            conn.tract_lengths[:, cer_indexes[0]] = tracts_avg
            # tract lengths left hem
            tracts_avg = np.average(conn.tract_lengths[cer_indexes[1::2], :], axis=0)
            conn.tract_lengths[cer_indexes[1], :] = tracts_avg
            conn.tract_lengths[:, cer_indexes[1]] = tracts_avg

            n2i_indexes = n2i_indexes + cer_indexes[2:]

        elif cer == "woCer":
            n2i_indexes = n2i_indexes + [i for i, roi in enumerate(conn.region_labels) if
                                         ('Cer' in roi) or ('Ver' in roi)]

        indexes = [i for i, roi in enumerate(conn.region_labels) if i not in n2i_indexes]
        conn.weights = conn.weights[:, indexes][indexes]
        conn.tract_lengths = conn.tract_lengths[:, indexes][indexes]
        conn.region_labels = conn.region_labels[indexes]
        conn.weights = conn.scaled_weights(mode="tract")

        conn.speed = np.array([15])

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

        jrMask_wc = [[False] if 'Thal' in roi else [True] for roi in conn.region_labels]


        # NEURAL MASS MODEL    #########################################################
        m = JansenRit_WilsonCowan(

            # Jansen-Rit nodes parameters. From Stefanovski et al. (2019)
            He=np.array([3.25]), Hi=np.array([22]),
            tau_e=np.array([10]), tau_i=np.array([20]),
            c=np.array([135.0]), p=np.array([0.09]),
            c_pyr2exc=np.array([1.0]), c_exc2pyr=np.array([0.8]),
            c_pyr2inh=np.array([0.25]), c_inh2pyr=np.array([0.25]),
            v0=np.array([6.0]), e0=np.array([0.005]), r=np.array([0.56]),

            # Wilson-Cowan nodes parameters. From Abeysuriya et al. (2018)
            P=np.array([0.4]), sigma=np.array([std_n]), Q=np.array([0]),
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

        # COUPLING FUNCTION   #########################################

        coup = coupling.SigmoidalJansenRit_Linear(

            # Jansen-Rit Sigmoidal coupling
            a=np.array([g_jr]), e0=np.array([0.005]), v0=np.array([6]), r=np.array([0.56]),

            # Wilson-Cowan Linear coupling
            a_linear=np.asarray([g_wc]),

            # JR mask | WC mask
            jrMask_wc=np.asarray(jrMask_wc)

        )


        # OTHER PARAMETERS   ###
        # integrator: dt=T(ms)=1000/samplingFreq(kHz)=1/samplingFreq(HZ)
        # integrator = integrators.HeunStochastic(dt=1000/samplingFreq, noise=noise.Additive(nsig=np.array([5e-6])))
        integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)

        mon = (monitors.Raw(),)

        print("Simulating %s (%is)  ||  PARAMS: g_jr%i - g_wc%0.2f std_n%0.2f" %
              (emp_subj, simLength / 1000, g_jr, g_wc, std_n))

        # Run simulation
        sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon)
        sim.configure()
        output = sim.run(simulation_length=simLength)

        # Extract data: "output[a][b][:,0,:,0].T" where:
        # a=monitorIndex, b=(data:1,time:0) and [200:,0,:,0].T arranges channel x timepoints and to remove initial transient.
        raw_data = output[0][1][transient:, 0, :, 0].T
        regionLabels = conn.region_labels

        max_cx = np.average(np.array([max(signal) for i, signal in enumerate(raw_data) if "Thal" not in regionLabels[i]]))
        min_cx = np.average(np.array([min(signal) for i, signal in enumerate(raw_data) if "Thal" not in regionLabels[i]]))
        max_th = np.average(np.array([max(signal) for i, signal in enumerate(raw_data) if "Thal" in regionLabels[i]]))
        min_th = np.average(np.array([min(signal) for i, signal in enumerate(raw_data) if "Thal" in regionLabels[i]]))

        # Cortical output from Jansen-Rit masses
        raw_data = raw_data[SC_cortex_idx, :]
        regionLabels = conn.region_labels[SC_cortex_idx]



        # Saving in FFT results: coupling value, conduction speed, mean signal freq peak (Hz; module), all signals info.
        _, _, IAF, module, band_module = multitapper(raw_data, samplingFreq, regionLabels, peaks=True)

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
            # timeseriesPlot(raw_data, raw_time, regionLabels)
            # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)

            # CONNECTIVITY MEASURES
            ## PLV
            plv = PLV(efPhase)

            # ## PLE - Phase Lag Entropy
            # ## PLE parameters - Phase Lag Entropy
            # tau_ = 25  # ms
            # m_ = 3  # pattern size
            # ple, patts = PLE(efPhase, tau_, m_, samplingFreq, subsampling=20)

            # Load empirical data to make simple comparisons
            plv_emp = \
                np.loadtxt(ctb_folder + "FCrms_" + emp_subj + "/" + bands[0][b] + "_plv_rms.txt", delimiter=',')[:,
                FC_cortex_idx][
                    FC_cortex_idx]

            # Comparisons
            t1 = np.zeros(shape=(2, len(plv) ** 2 // 2 - len(plv) // 2))
            t1[0, :] = plv[np.triu_indices(len(plv), 1)]
            t1[1, :] = plv_emp[np.triu_indices(len(plv), 1)]
            plv_r = np.corrcoef(t1)[0, 1]

            ## dynamical Functional Connectivity
            # Sliding window parameters
            window, step = 4, 2  # seconds

            ## dFC
            dFC = dynamic_fc(raw_data, samplingFreq, transient, window, step, "PLV")

            dFC_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_dPLV4s.txt")

            # Compare dFC vs dFC_emp
            t2 = np.zeros(shape=(2, len(dFC) ** 2 // 2 - len(dFC) // 2))
            t2[0, :] = dFC[np.triu_indices(len(dFC), 1)]
            t2[1, :] = dFC_emp[np.triu_indices(len(dFC), 1)]
            dFC_ksd = scipy.stats.kstest(dFC[np.triu_indices(len(dFC), 1)], dFC_emp[np.triu_indices(len(dFC), 1)])[0]

            ## Metastability: Kuramoto Order Parameter
            ko_std, ko_mean = kuramoto_order(raw_data, samplingFreq)
            ko_emp = np.loadtxt(ctb_folderOLD + "FC_" + emp_subj + "/" + bands[0][b] + "_sdKO.txt")

            ## Gather results
            result.append(
                (emp_subj, th, cer, g_jr, g_wc, std_n, r,
                 min_cx, max_cx, min_th, max_th,
                 IAF[0], module[0], band_module[0], bands[0][b],
                 plv_r, dFC_ksd, ko_std, ko_emp))

        print("LOOP ROUND REQUIRED %0.3f seconds.\n\n" % (time.time() - tic,))

    return np.asarray(result, dtype=object)
