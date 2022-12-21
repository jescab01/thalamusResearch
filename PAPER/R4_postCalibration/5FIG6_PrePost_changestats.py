'''
Creating figures and calculating statistics for ThalamusResearch
A priori using already computed gexplore_data: to get an idea of the plotting.

TODO after the idea is ready recompute calculus

  -  WITH OR WITHOUT THALAMUS  -
'''

import os
import pandas as pd
import pingouin as pg
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

## DATA for reference rPLV & KSD: PRE-POST && DATA for g_explore
simulations_tag = "PSEmpi_CalibProc_prepost-m12d09y2022-t20h.07m.01s"

if "PAPER" in os.getcwd():
    folder = 'data\\' + simulations_tag + '\\'
else:
    folder = 'PAPER\\R4_postCalibration\\data\\' + simulations_tag + '\\'

df_prepost = pd.read_csv(folder + "results.csv")
# Average out repetitions
df_prepost_avg = df_prepost.groupby(["subject", "model", "th", "cer", "g", "pth", "sigma", "pcx"]).mean().reset_index()

pn_vals = [("0", 0.09, 0.022, "0.09"), ("A", 0.15, 0.022, "0.09"), ("B", 0.15, 0.15, "0.09"), ("C", 0.15, 0.15, "MLR")]
sp_t = [[r"$%s.   p_{th}=%0.2f;  \eta_{th}=%0.2f;  p_{\neq th}=%s$" % (id, pth, sigma, pcx), ""] for
        i, (id, pth, sigma, pcx) in enumerate(pn_vals)]
sp_t = [elem for sp in sp_t for elem in sp]

df_prepost_avg["stage"] = None

for i, (stage, pth, sigma, pcx) in enumerate(pn_vals):
    mask = (df_prepost_avg["pth"] == pth) & (df_prepost_avg["sigma"] == sigma) & (df_prepost_avg["pcx"] == pcx)
    df_prepost_avg.loc[mask, "stage"] = stage

# PRE-BIF
prebif_lowg, prebif_topg = 1, 6
df_prepost_avg_prebif = df_prepost_avg.loc[
    (df_prepost_avg["th"] == "pTh") & (df_prepost_avg["g"] < prebif_topg) & (df_prepost_avg["g"] > prebif_lowg)].copy()

# Calculate differences to baseline
df_prepost_avg_prebif["diff_2base_rPLV"] = [row.rPLV - np.max(df_prepost_avg_prebif["rPLV"].loc
                       [(df_prepost_avg_prebif["subject"]==row.subject) & (df_prepost_avg_prebif["pth"]==0.09) &
                        (df_prepost_avg_prebif["sigma"]==0.022) & (df_prepost_avg_prebif["pth"]==0.09)])
     for i, row in df_prepost_avg_prebif.iterrows()]

df_prepost_avg_prebif["diff_2base_KSD"] = [row.dFC_KSD - np.min(df_prepost_avg_prebif["dFC_KSD"].loc
                       [(df_prepost_avg_prebif["subject"]==row.subject) & (df_prepost_avg_prebif["pth"]==0.09) &
                        (df_prepost_avg_prebif["sigma"]==0.022) & (df_prepost_avg_prebif["pth"]==0.09)])
     for i, row in df_prepost_avg_prebif.iterrows()]


## STATISTICAL ANALYSIS with KRUSKAL WALLIS
# rPLV
df_sub = df_prepost_avg_prebif.groupby(["subject", "stage"]).max(["rPLV"]).reset_index()
pg.sphericity(df_sub, dv="rPLV", within="stage", subject="subject")
pg.normality(df_sub, dv="rPLV", group="stage")

test_rplv_friedman = pg.friedman(df_sub, dv="rPLV", within="stage", subject="subject")

test_rplv_pwc = pg.pairwise_tests(df_sub, dv="rPLV", within="stage", subject="subject", parametric=False, padjust='fdr_bh', effsize="cohen")

# KSD
df_sub = df_prepost_avg_prebif.groupby(["subject", "stage"]).min(["dFC_KSD"]).reset_index()

pg.sphericity(df_sub, dv="dFC_KSD", within="stage", subject="subject")
pg.normality(df_sub, dv="dFC_KSD", group="stage")

test_ksd_friedman = pg.friedman(df_sub, dv="dFC_KSD", within="stage", subject="subject")

test_ksd_pwc = pg.pairwise_tests(df_sub, dv="dFC_KSD", within="stage", subject="subject", parametric=False, padjust='fdr_bh', effsize="cohen")



# PLOT paired
cmap=px.colors.qualitative.Plotly
colors = [cmap[i] for i, subj in enumerate(set(df_prepost_avg_prebif.subject.values))]
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.35)

for stage in ["A", "B", "C"]:

    # ADD rPLV trace
    y = df_prepost_avg_prebif.loc[df_prepost_avg_prebif["stage"]==stage].groupby(["subject"]).max(["rPLV"])
    y = y.diff_2base_rPLV.values
    fig.add_trace(
        go.Box(x=[stage]*len(y), y=y,
               marker_color="lightgray", opacity=0.9, showlegend=False), row=1, col=1)


    if test_rplv_pwc["p-corr"].loc[(test_rplv_pwc["A"]=="0") & (test_rplv_pwc["B"]==stage)].values[0] < 0.05:
        fig.add_trace(go.Scatter(x=[stage], y=[max(y)+0.15], mode="text", text="*", textposition="middle center", showlegend=False), row=1, col=1)

    # ADD KSD trace
    y = df_prepost_avg_prebif.loc[df_prepost_avg_prebif["stage"] == stage].groupby(["subject"]).min(["dFC_KSD"])
    y = y.diff_2base_KSD.values
    fig.add_trace(go.Box(x=[stage]*len(y), y=y,
                         marker_color="lightgray",  opacity=0.9, showlegend=False), row=1, col=2)

    if test_ksd_pwc["p-corr"].loc[(test_ksd_pwc["A"]=="0") & (test_ksd_pwc["B"]==stage)].values[0] < 0.05:
        fig.add_trace(go.Scatter(x=[stage], y=[max(y)+0.15], mode="text", text="*", textposition="middle center", showlegend=False), row=1, col=2)

for i, subj in enumerate(set(df_prepost_avg_prebif.subject.values)):

    y = df_prepost_avg_prebif.loc[(df_prepost_avg_prebif["subject"] == subj) & (df_prepost_avg_prebif["stage"] != "0")].groupby(["stage"]).max(["rPLV"]).reset_index()
    fig.add_trace(
        go.Scatter(x=y.stage.values, y=y.diff_2base_rPLV, mode="markers", marker=dict(color=cmap[i]), line=dict(color=cmap[i]),
                   showlegend=False, name="Subject " + str(i+4).zfill(2), legendgroup="Subject " + str(i+4).zfill(2),
               opacity=0.7), row=1, col=1)

    y = df_prepost_avg_prebif.loc[(df_prepost_avg_prebif["subject"] == subj) & (df_prepost_avg_prebif["stage"] != "0")].groupby(["stage"]).min(["dFC_KSD"]).reset_index()
    fig.add_trace(
        go.Scatter(x=y.stage.values, y=y.diff_2base_KSD, mode="markers", marker=dict(color=cmap[i]), line=dict(color=cmap[i], width=1),
                   name="Subject " + str(i + 4).zfill(2), legendgroup="Subject " + str(i + 4).zfill(2),
               opacity=0.7, showlegend=False), row=1, col=2)

fig.update_layout(template="plotly_white", width=500, height=500,
                  yaxis1=dict(title=r"$r_{PLV}$", range=[-0.09, 0.09]), yaxis2=dict(title="$KSD$", range=[-0.5, 0.5]))

pio.write_html(fig, file=folder + "PrePost_BOXpaired.html", auto_open=True, include_mathjax="cdn")
pio.write_image(fig, file=folder + "PrePost_BOXpaired.svg")




#  .OLD
#
#     for i, _ in enumerate(df_prebif_maxrPLV_post["rPLV"].values):
#         c = cmap[2] if df_prebif_maxrPLV_pre["rPLV"].values[i] < df_prebif_maxrPLV_post["rPLV"].values[i] else cmap[0]
#         fig.add_trace(go.Scatter(x=["Pre", "Post"],
#                                  y=[df_prebif_maxrPLV_pre["rPLV"].values[i], df_prebif_maxrPLV_post["rPLV"].values[i]],
#                                  line=dict(color=c), showlegend=False, opacity=0.5), row=1, col=1)
#
#     # KSD
#     fig.add_trace(go.Box(x=["Pre" for i in df_prebif_minKSD_pre["dFC_KSD"].values], y=df_prebif_minKSD_pre["dFC_KSD"].values,
#                          marker_color="lightgray", opacity=0.9, showlegend=False), row=1, col=2)
#     fig.add_trace(go.Box(x=["Post" for i in df_prebif_minKSD_post["dFC_KSD"].values], y=df_prebif_minKSD_post["dFC_KSD"].values,
#                          marker_color="lightgray",  opacity=0.9, showlegend=False), row=1, col=2)
#
#     for i, _ in enumerate(df_prebif_minKSD_pre["dFC_KSD"].values):
#         c = cmap[2] if df_prebif_minKSD_pre["dFC_KSD"].values[i] < df_prebif_minKSD_post["dFC_KSD"].values[i] else cmap[0]
#         fig.add_trace(go.Scatter(x=["Pre", "Post"],
#                                  y=[df_prebif_minKSD_pre["dFC_KSD"].values[i], df_prebif_minKSD_post["dFC_KSD"].values[i]],
#                                  line=dict(color=c), showlegend=False, opacity=0.5), row=1, col=2)
#
#
# ###  rPLV
# ttests_rplv = pd.DataFrame()
# for stage in ["A", "B", "C"]:
#     subset = df_prepost_avg_prebif.loc[df_prepost_avg_prebif["stage"] == stage].groupby(["subject"]).max(["diff_2base_rPLV"])
#     ttests_rplv = ttests_rplv.append(pg.ttest(x=subset["diff_2base_rPLV"].values, y=0))
#
#
# ###  KSD
# ttests_ksd = pd.DataFrame()
# for stage in ["A", "B", "C"]:
#     subset = df_prepost_avg_prebif.loc[df_prepost_avg_prebif["stage"] == stage].groupby(["subject"]).max(["diff_2base_KSD"])
#     ttests_ksd = ttests_ksd.append(pg.ttest(x=subset["diff_2base_KSD"].values, y=0))
#
#
# fig = px.box(df_prepost_avg_prebif, x="stage", y="rPLV")
# fig.show("browser")
#
# fig = px.box(df_prepost_avg_prebif, x="stage", y="dFC_KSD")
# fig.show("browser")
#
#
#
#
#
#
#
#
# df_prebif_maxrPLV_pre = df_prepost_avg.loc[(df_prepost_avg["th"] == "pTh") & (df_prepost_avg["g"] < prebif_topg) & (df_prepost_avg["g"] > prebif_lowg)
#                                       & (df_prepost_avg["sigma"] == 0.022) & (df_prepost_avg["pth"]==0.09)].groupby(["subject", "model", "th", "cer"]).max(["rPLV"]).reset_index()
#
# df_prebif_maxrPLV_post = df_prepost_avg.loc[(df_prepost_avg["th"] == "pTh") & (df_prepost_avg["g"] < prebif_topg) & (df_prepost_avg["g"] > prebif_lowg) &
#                                        (df_prepost_avg["sigma"] == 0.15) & (df_prepost_avg["pcx"]=="MLR")].groupby(["subject", "model", "th", "cer", "sigma"]).max(["rPLV"]).reset_index()
#
# # STATISTICAL ANALYSIS directly wilcoxon (due to small sample)
# # rPLV
# g1 = df_prebif_maxrPLV_pre["rPLV"].values
# g2 = df_prebif_maxrPLV_post["rPLV"].values
# test_rplv = pg.wilcoxon(x=g1, y=g2)
#
# df_prebif_minKSD_pre = df_prepost_avg.loc[(df_prepost_avg["th"] == "pTh") & (df_prepost_avg["g"] < prebif_topg) & (df_prepost_avg["g"] > prebif_lowg)
#                                       & (df_prepost_avg["sigma"] == 0.022) & (df_prepost_avg["pth"] == 0.09)].groupby(["subject", "model", "th", "cer"]).min(["dFC_KSD"]).reset_index()
#
# df_prebif_minKSD_post = df_prepost_avg.loc[(df_prepost_avg["th"] == "pTh") & (df_prepost_avg["g"] < prebif_topg) & (df_prepost_avg["g"] > prebif_lowg) &
#                                        (df_prepost_avg["sigma"] == 0.15) & (df_prepost_avg["pcx"]=="MLR")].groupby(["subject", "model", "th", "cer", "sigma"]).min(["dFC_KSD"]).reset_index()
#
#
# ### KSD
# g1 = df_prebif_minKSD_pre["dFC_KSD"].values
# g2 = df_prebif_minKSD_post["dFC_KSD"].values
# test_ksd = pg.wilcoxon(x=g1, y=g2)
#
