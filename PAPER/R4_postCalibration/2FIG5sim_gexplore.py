
import os
import pickle
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from report.functions import simulate

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"

# Define folder with mpi data
sim_tag = ""

sim_folder = "data/" + sim_tag if "PAPER" in os.getcwd() else "PAPER/R4_postCalibration/data/" + sim_tag

## Define param combinations
# Common simulation requirements
subj_ids = [58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

pn_vals = [(0.09, 0.022, 0.09),
           (0.15, 0.022, 0.09),
           (0.15, 0.15, 0.09),
           (0.15, 0.15, "MLR")]

g = 2

for subj in subjects:
    output = []
    for pth, sigma, pcx in pn_vals:
        print([subj, g, pth, sigma, pcx])

        # Simulate -
        output.append([[subj, g, pth, sigma, pcx], [simulate(subj, "jr", g=g, p_th=pth, sigma=sigma, p_cx=pcx, th='pTh', t=60, mode="FIG")]])

    ## Save simulations results using pickle
    open_file = open(sim_folder + "/g_explore-"+subj+"inTestingGroup-PrePost.pkl", "wb")
    pickle.dump(output, open_file)
    open_file.close()



