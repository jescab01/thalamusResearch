

import numpy as np
import pickle



## DATA for g_explore
simulations_tag = "g_explore-FIG3_N35-initialConditions_v2-m12d19y2022-t19h.29m.26s.pkl"
folder = 'PAPER2\\R3_scenariosCharact\\data\\'

# Load and plot already computed simulations.
with open(folder + simulations_tag, "rb") as input_file:
    g_sel, output = pickle.load(input_file)


## SIGNAL To noise ratio in thalamus
s2n = []
for i, (g, sigma) in enumerate(g_sel[:2]):

    # Unpack output
    signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = output[i]

    mask = [True if "Thal" in roi else False for roi in regionLabels]

    s2n_temp = np.average(((np.max(signals, axis=1) - np.min(signals, axis=1))/sigma)[mask])
    s2n.append([g, sigma, s2n_temp])


## RELATIVE POWER between THALAMUS and CORTEX
cx2th_power, spectra = [], []
for i, (g, sigma) in enumerate(g_sel):
    # Unpack output
    signals, timepoints, plv, dplv, plv_emp, dFC_emp, regionLabels, simLength, transient, SC_cortex_idx = output[i]

    freqs = np.arange(len(signals[0]) / 2)
    freqs = freqs / ((simLength - transient) / 1000)  # simLength (ms) / 1000 -> segs

    temp_spectra = []
    for ii, signal in enumerate(signals):

        # Spectra
        freqRange = [2, 40]
        fft_temp = abs(np.fft.fft(signal))  # FFT for each channel signal
        fft = np.asarray(fft_temp[range(int(len(signal) / 2))])  # Select just positive side of the symmetric FFT
        fft = fft[(freqs > freqRange[0]) & (freqs < freqRange[1])]  # remove undesired frequencies

        temp_spectra.append(fft)
    spectra.append(temp_spectra)


mask = [True if "Thal" in roi else False for roi in regionLabels]
for i, (g, sigma) in enumerate(g_sel):

    # calcula integral de thalamo y cortex
    fft_integral = np.sum(np.array(spectra[i]), axis=1)

    cx2th_temp = np.average(fft_integral[np.invert(mask)])/np.average(fft_integral[mask])

    cx2th_power.append([g, sigma, cx2th_temp])

