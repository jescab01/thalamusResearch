
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

"""
Following https://realpython.com/k-means-clustering-python/
"""

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
conn = connectivity.Connectivity.from_file(ctb_folder + "NEMOS_035_AAL2pTh_pass.zip")

emp_subj = "NEMOS_035"

FC_matrices, output = [], []

# Load adjusted p_array for NEMOS_035 (g==2);
with open("report/data/p_adjust-ch1pHetero-m08d24y2022-t21h.25m.43s.pkl", "rb") as input_file:
    p_array, _, _, _, _, _, _ = pickle.load(input_file)


output = simulate(emp_subj, "jr", g=2, p_array=p_array, sigma=0.15, th='pTh', t=120, stimulate=False, mode="FC")

## Save simulations results using pickle
open_file = open("report/gexplore_data/Networks_LongSim-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "wb")
pickle.dump(output, open_file)
open_file.close()


# Load long simulations
# open_file = open("gexplore_data/Networks_LongSim-ch2Networks-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "rb")
# output = pickle.load(open_file)
# open_file.close()


# Gather all plv_matrices
    # linearize fcs: upper triangle
linearized_fc = [plv[np.triu_indices(len(plv), 1)] for plv in output[0]]


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

# Plot SSE (sum of squared errors) and trying to find elbow
fig = go.Figure(go.Scatter(x=list(range_nclusters), y=np.asarray(sse)))
fig.add_vline(x=elbow)
fig.update_layout(yaxis=dict(title="SSE"), xaxis=dict(title="Nº of clusters"),
                  title="Sum of distances between cluster centroid and points in that clusters")
fig.show("browser")

# Plot silhouette: How well a point fits into its cluster based on:
# closeness to other points, distance to points in other clusters
fig = go.Figure(go.Scatter(x=list(range_nclusters), y=np.asarray(silhouette_coefs)))
fig.add_vline(x=elbow)
fig.update_layout(yaxis=dict(title="Silhouette coefficient"), xaxis=dict(title="Nº of clusters"),
                  title="Silhouette coefficient [-1, 1] larger values indicate points closer to their clusters than to others")
fig.show("browser")


# Onces k adjusted: fit K means
kmeans = KMeans(n_clusters=elbow, **kmeans_kwargs)
kmeans.fit(linearized_fc)

kmeans.labels_


# Average clusters



# Test clusters pair of connections against global distribution for that connection



# Correct for multiple comparisons




## example of k means
import matplotlib.pyplot as plt
data = make_blobs(n_features=4)
kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300)
kmeans.fit(data[0])
kmeans.inertia_
kmeans.cluster_centers_
kmeans.n_iter_
kmeans.labels_



## Do it on EMPIRICAL DATA
import scipy.io
ctb_folder, emp_subj, band = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\", "NEMOS_035", "3-alpha"
emp_plvs = scipy.io.loadmat(ctb_folder + "FCrms_" + emp_subj + "/" + band + "_all_plvs_rms.mat")
emp_plvs = emp_plvs["plv_rms"]



# Gather all plv_matrices
    # linearize fcs: upper triangle
linearized_fc = [emp_plvs[:, :, i][np.triu_indices(len(emp_plvs[:, :, i]), 1)] for i in range(len(emp_plvs[0, 0, :]))]


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

# Plot SSE (sum of squared errors) and trying to find elbow
fig = go.Figure(go.Scatter(x=list(range_nclusters), y=np.asarray(sse)))
fig.add_vline(x=elbow)
fig.update_layout(yaxis=dict(title="SSE"), xaxis=dict(title="Nº of clusters"),
                  title="Sum of distances between cluster centroid and points in that clusters")
fig.show("browser")

# Plot silhouette: How well a point fits into its cluster based on:
# closeness to other points, distance to points in other clusters
fig = go.Figure(go.Scatter(x=list(range_nclusters), y=np.asarray(silhouette_coefs)))
fig.add_vline(x=elbow)
fig.update_layout(yaxis=dict(title="Silhouette coefficient"), xaxis=dict(title="Nº of clusters"),
                  title="Silhouette coefficient [-1, 1] larger values indicate points closer to their clusters than to others")
fig.show("browser")