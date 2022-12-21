
import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from functions import p_adjust
import pandas as pd
import numpy as np
from mpi4py import MPI

"""
UNFINISHED.

Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python mpi_thcer2.py
"""

name = "GD-pHetero"
save_output = True

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


if rank == 0:
    ## Save resutls
    ## Folder structure - Local
    if "Jesus CabreraAlvarez" in os.getcwd():
        wd = os.getcwd()

        main_folder = wd + "\\" + "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)
        specific_folder = main_folder + "\\PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")

        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

    ## Folder structure - CLUSTER
    else:
        main_folder = "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)

        os.chdir(main_folder)

        specific_folder = "PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

else:
    # Pre-allocate variable
    specific_folder = ""

# send specific folder to other nodes
comm.barrier()
specific_folder = comm.bcast(np.asarray(specific_folder, dtype=object), root=0)
print(specific_folder)
os.chdir(specific_folder)

subj_ids = [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

structure_th = ["pTh"]
g_vals = np.arange(0.5, 3, 0.5)


params = [[subj, "pTh", g] for subj in subjects for g in g_vals]

params = np.asarray(params, dtype=object)
n = params.shape[0]

## Distribution of task load in ranks
count = n // size  # number of catchments for each process to analyze
remainder = n % size  # extra catchments if n is not a multiple of size

if rank < remainder:  # processes with rank < remainder analyze one extra catchment
    start = rank * (count + 1)  # index of first catchment to analyze
    stop = start + count + 1  # index of last catchment to analyze
else:
    start = rank * count + remainder
    stop = start + count

local_params = params[start:stop, :]  # get the portion of the array to be analyzed by each rank

table = []
for (subj, th, g) in local_params:

    print("\n%s | %s | g%0.2f" % (subj, th, g))

    out = p_adjust(subj, th, g=g, p_th=0.15, sigma=0.15, p_cx=0.09, iterations=40, report=False, plotmode=["html"], folder=specific_folder)

    p_array, signals, p_array_init, signals_init, timepoints, regionLabels, degree, degree_avg, degree_fromth, degree_fromth_avg, results = out

    # Save raw output
    if save_output:
        if "Jesus CabreraAlvarez" in os.getcwd():
            with open(specific_folder + "\\pHetero_%s_%s_g%s.pkl" % (subj, th, str(g)), "wb") as file:
                pickle.dump([subj, th, g, out], file)
        else:
            with open("pHetero_%s_%s_g%s.pkl" % (subj, th, str(g)), "wb") as file:
                pickle.dump([subj, th, g, out], file)

    temp_tbl = [[subj, th, g, 0.15, 0.15, 0.09, p_array[i], degree[i], degree_avg, degree_fromth[i], degree_fromth_avg, roi]
                for i, roi in enumerate(regionLabels) if "Thal" not in roi]

    table = table + temp_tbl


if rank > 0:  # WORKERS _send to rank 0
    comm.send(table, dest=0, tag=14)  # send results to process 0

else:  ## MASTER PROCESS _receive, merge and save results
    final_table = np.copy(table)  # initialize final results with results from process 0
    for i in range(1, size):  # determine the size of the array to be received from each process
        # if i < remainder:
        #     rank_size = count + 1
        # else:
        #     rank_size = count
        # tmp = np.empty((rank_size, final_results.shape[1]))  # create empty array to receive results
        tmp = comm.recv(source=i, tag=14)  # receive results from the process

        if tmp is not None:  # Sometimes temp is a Nonetype wo/ apparent cause
            # print(final_results.shape)
            # print(tmp.shape)  # debugging
            # print(i)
            final_results = np.vstack((final_table, tmp))  # add the received results to the final results

    # print("Results")
    # print(final_results)

    fTable = pd.DataFrame(final_table, columns=["subject", "th", "g", "pth", "sigma", "initpcx", "p_adjusted", "degree",
                                         "degree_avg", "degree_fromth", "degree_fromth_avg", "roi"])

    ## Save resutls
    ## Folder structure - Local
    if "Jesus CabreraAlvarez" in os.getcwd():
        pd.to_pickle(fTable, specific_folder + "\\.GD-pHetero_TABLE-SUBJECTS.pkl")

        # output = []
        # for (subj, th, g) in params:
        #     with open(specific_folder + "\\pHetero_%s_%s_g%s.pkl" % (subj, th, str(g)), "rb") as file:
        #         output.append(pickle.load(file))
        #
        # with open(specific_folder + ".GD-pHetero_alldata-SUBJECTS.pkl", "wb") as file:
        #     pickle.dump(output, file)

    ## Folder structure - CLUSTER
    else:
        pd.to_pickle(table, ".GD-pHetero_TABLE-SUBJECTS.pkl")

        # output = []
        # for (subj, th, g) in params:
        #     with open("pHetero_%s_%s_g%s.pkl" % (subj, th, str(g)), "rb") as file:
        #         output.append(pickle.load(file))
        #
        # with open(specific_folder + ".GD-pHetero_alldata-SUBJECTS.pkl", "wb") as file:
        #     pickle.dump(output, file)

