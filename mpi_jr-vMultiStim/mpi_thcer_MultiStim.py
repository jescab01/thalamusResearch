
import os
import time
import pandas as pd
import numpy as np

from mpi4py import MPI
from ThCer_parallel_MultiStim import ThCer_parallel

"""
Following a tutorial: 
https://towardsdatascience.com/parallel-programming-in-python-with-message-passing-interface-mpi4py-551e3f198053

execute in terminal with : mpiexec -n 4 python mpi_thcer_MultiStim.py
"""

name = "MultiStim"

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## Define param combinations
# Common simulation requirements
subj_ids = [35]#, 49, 50, 58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]
# subjects.append("NEMOS_AVG")

# Fixed parameters
models = ["jr"]
structure_th = ["pTh"]
structure_cer = ["pCer"]
coupling_vals = [3]  # g==3 before bifurcation  # np.arange(0, 100, 0.5)  # 0.5
pth_vals = [0.22]  # np.arange(0.09, 0.4, 0.005)  # p(cortex)=0.09
nth_vals = [0.15]  # np.logspace(-8, 2, 30)  # [0, 0.022]  # np.linspace(0, 0.1, 40)  # 20
n_rep = 3

# Explored parameters :: multi-stimulation
ms_gains = np.arange(0.01, 1, 0.1)
ms_nstates = np.arange(2, 11, 2)
ms_tstates = np.arange(500, 10000, 1000)
ms_pinclusion = np.arange(0.05, 0.3, 0.05)

params = [[subj, model, th, cer, g, pth, nth, r, ["sinusoid", gain, nstates, tstates, pinclusion, False]]
          for subj in subjects for model in models for th in structure_th for cer in structure_cer
          for g in coupling_vals for pth in pth_vals for nth in nth_vals for r in range(n_rep)
          for gain in ms_gains for nstates in ms_nstates
          for tstates in ms_tstates for pinclusion in ms_pinclusion]

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

local_results = ThCer_parallel(local_params)  # run the function for each parameter set and rank


if rank > 0:  # WORKERS _send to rank 0
    comm.send(local_results, dest=0, tag=14)  # send results to process 0

else:  ## MASTER PROCESS _receive, merge and save results
    final_results = np.copy(local_results)  # initialize final results with results from process 0
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

            final_results = np.vstack((final_results, tmp))  # add the received results to the final results

    # print("Results")
    # print(final_results)

    fResults_df = pd.DataFrame(final_results, columns=["subject", "model", "th", "cer", "g", "p", "sigma", "rep", "stim_params",
                                                       "min_cx", "max_cx", "min_th", "max_th", "IAF", "module", "bModule", "band",
                                                       "rPLV", "dFC_KSD", "KOstd", "KOstd_emp", "states", "timing"])

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

        fResults_df.to_pickle(specific_folder + "/results.pkl")

    ## Folder structure - CLUSTER
    else:
        main_folder = "PSE"
        if os.path.isdir(main_folder) == False:
            os.mkdir(main_folder)

        os.chdir(main_folder)

        specific_folder = "PSEmpi_" + name + "-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
        if os.path.isdir(specific_folder) == False:
            os.mkdir(specific_folder)

        os.chdir(specific_folder)

        fResults_df.to_pickle("results.pkl")
