
import time
import scipy
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.

import plotly.graph_objects as go
from tvb.simulator.lab import connectivity

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from report.functions import simulate

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2pTh_pass.zip")

# load text with FC rois; check if match SC
SClabs = list(conn.region_labels)
SC_notTh_idx = [SClabs.index(roi) for roi in conn.region_labels if "Thal" not in roi]


emp_subj = "NEMOS_035"

FC_matrices, output = [], []

# Load adjusted p_array for NEMOS_035 (g==2);
with open("report/data/p_adjust-ch1pHetero-m08d24y2022-t21h.25m.43s.pkl", "rb") as input_file:
    p_array, _, _, _, _, _, _ = pickle.load(input_file)

# sim with input x2sim x240s; + 2sim x30sec w/input
sets = [[30, 0, 0.8, 5500, 0.15]]

for set in sets:

    simLength, nstates_s, gain, tstates, pinc = set

    if nstates_s == 0:
        stim = False
    else:
        stim = ["sinusoid", gain, np.int(nstates_s * simLength), tstates, pinc, False]  # stim_type, gain, nstates, tstates, pinclusion, deterministic

    output.append(simulate(emp_subj, "jr", g=2, p_array=p_array, sigma=0.15, th='pTh', t=simLength, stimulate=stim, mode="FC"))

## Save simulations results using pickle
open_file = open("report/gexplore_data/Networks_LongSim-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "wb")
pickle.dump(output, open_file)
open_file.close()


# Load long simulations
# open_file = open("gexplore_data/Networks_LongSim-ch2Networks-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "rb")
# output = pickle.load(open_file)
# open_file.close()


# Gather all plv_matrices
linearized_fc = []
for sim in output:

    # linearize fcs: upper triangle
    linearized_fc = linearized_fc + [plv[np.triu_indices(len(plv), 1)] for plv in sim[0]]


# K-means clustering as many features as connections (upper triangular):: first adjust k (n clusters)
# Range for n_clusters
range_nclusters = range(2, 20)
kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300}

sse, silhouette_coefs = [], []
for k in range_nclusters:

    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(linearized_fc)

    sse.append(kmeans.inertia_)
    silhouette_coefs.append(silhouette_score(linearized_fc, kmeans.labels_))

elbow = KneeLocator(range_nclusters, sse, curve="convex", direction="decreasing").elbow

# plot
fig = go.Figure(go.Scatter(x=list(range_nclusters), y=np.asarray(sse)))
fig.add_vline(x=elbow)
fig.show("browser")

fig = go.Figure(go.Scatter(x=list(range_nclusters), y=np.asarray(silhouette_coefs)))
fig.add_vline(x=elbow)
fig.show("browser")


# Onces k adjusted: fit K means
kmeans = KMeans(n_clusters=elbow, **kmeans_kwargs)
kmeans.fit(linearized_fc)

kmeans.labels_


# Average clusters



# Test clusters pair of connections against global distribution for that connection


# Correct for multiple comparisons




## testing example of k means
import matplotlib.pyplot as plt




data = make_blobs(n_features=4)

kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300)

kmeans.fit(data[0])
kmeans.inertia_
kmeans.cluster_centers_
kmeans.n_iter_

kmeans.labels_

# elbow






# silhouwttw
