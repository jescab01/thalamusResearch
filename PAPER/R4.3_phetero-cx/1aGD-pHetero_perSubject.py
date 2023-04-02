
import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from report.functions import p_adjust
import pandas as pd
import numpy as np


specific_folder = "data\\pHetero-SUBJECTs_" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + "\\"
if os.path.isdir(specific_folder) == False:
    os.mkdir(specific_folder)

subj_ids = [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

structure_th = ["pTh"]
g_vals = np.arange(0.5, 3, 0.5)

table = []
for emp_subj in subjects:
    for th in structure_th:
        for g in g_vals:

            print("\n%s | %s | g%0.2f" % (emp_subj, th, g))

            out = p_adjust(emp_subj, th, g=g, p_th=0.15, sigma=0.15, p_cx=0.09, iterations=40, report=True, plotmode=["html"], folder=specific_folder)

            p_array, signals, p_array_init, signals_init, timepoints, regionLabels, degree, degree_avg, degree_fromth, degree_fromth_avg, results = out

            # Save raw output
            with open(specific_folder + "pHetero_%s_%s_g%s.pkl" % (emp_subj, th, str(g)), "wb") as file:
                pickle.dump([emp_subj, th, g, out], file)

            temp_tbl = [[emp_subj, th, g, 0.09, 0.15, 0.15, p_array[i], degree[i], degree_avg, degree_fromth[i], degree_fromth_avg, roi]
                        for i, roi in enumerate(regionLabels) if "Thal" not in roi]

            table = table + temp_tbl

# Save table for simulations
table = pd.DataFrame(table, columns=["subject", "th", "g", "initpcx", "pth", "sigma", "p_adjusted", "degree",
                                     "degree_avg", "degree_fromth", "degree_fromth_avg", "roi"])

pd.to_pickle(table, specific_folder + ".1pHeteroTABLE-SUBJECTS.pkl")



## Merge all pkls in one large file
# specific_folder = "E:\LCCN_Local\PycharmProjects\\thalamusResearch\PAPER\R4.3_phetero-cx\data\pHetero-SUBJECTs_m12d08y2022-t11h.19m.24s\\"

output = []
for emp_subj in subjects:
    for th in structure_th:
        for g in g_vals:
            with open(specific_folder + "pHetero_%s_%s_g%s.pkl" % (emp_subj, th, str(g)), "rb") as file:
                output.append(pickle.load(file))


with open(specific_folder + ".1pHeteroFULL-SUBJECTS.pkl", "wb") as file:
    pickle.dump(output, file)
