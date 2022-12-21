'''
Creating figures and calculating statistics for ThalamusResearch
A priori using already computed gexplore_data: to get an idea of the plotting.

TODO after the idea is ready recompute calculus

  -  WITH OR WITHOUT THALAMUS  -
'''

import pandas as pd
import pingouin as pg
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

simulations_tag = "PSEmpi_JR_v2-m12d19y2022-t14h.39m.50s"
folder = 'PAPER2\\R1_TH-type&noise\\' + simulations_tag + '\\'

df = pd.read_csv(folder + "results.csv")

# Average out repetitions
df_avg = df.groupby(["subject", "model", "th", "cer", "g", "sigmath"]).mean().reset_index()


# Define bifurcation
bif_val = 7

df_avg["bif"] = ["prebif" if row.g < bif_val else "postbif" for id, row in df_avg.iterrows()]


# Calculate max rPLV per subject and condition
df_max = df_avg.loc[df_avg["g"]>0].groupby(["subject", "model", "th", "cer", "sigmath", "bif"]).max(["rPLV"]).reset_index()
# df_max.to_csv(folder + "df_max.csv")

# Calculate min KSD per subject and condition
df_KSDmin = df_avg.loc[df_avg["g"]>0].groupby(["subject", "model", "th", "cer", "sigmath", "bif"]).min(["dFC_KSD"]).reset_index()


# STATISTICS on rPLV
## Simple effects:: ANOVA on each level of sigma
res_aov, ass_norm, ass_sph = [], [], []
for sigma in set(df_avg.sigmath):
    for bif in ["prebif", "postbif"]:
        df_max_cond = df_max.loc[(df_max["sigmath"] == sigma) & (df_max["bif"] == bif)]

        ass_norm.append(pg.normality(df_max_cond, dv="rPLV", group="th"))
        ass_sph.append(pg.sphericity(df_max_cond, dv="rPLV"))

        test = pg.friedman(data=df_max_cond, dv="rPLV", within="th", subject="subject")
        res_aov.append([sigma, bif] + list(test.values[0]))

res_aov = pd.DataFrame(res_aov, columns=["sigmath", "bif"]+list(test.columns.values))
res_aov["p-corr"] = pg.multicomp(pvals=res_aov["p-unc"].values, method="fdr_bh")[1]

## Multiple comparisons on sigma==0.022: wilcoxon (due to small sample size)
res_mc = pd.DataFrame()
for sigma in set(df_avg.sigmath):
    for bif in ["prebif", "postbif"]:
        df_max_cond = df_max.loc[(df_max["sigmath"] == sigma) & (df_max["bif"] == bif)]
        if res_aov["p-corr"].loc[(res_aov["sigmath"] == sigma) & (res_aov["bif"] == bif)].values < 0.05:
            test = pg.pairwise_tests(df_max_cond, dv="rPLV", within="th", subject="subject", effsize="cohen", parametric=False)
            test["sigmath"], test["bif"] = sigma, bif
            res_mc = res_mc.append(test)

res_mc = pd.DataFrame(res_mc, columns=list(test.columns.values))
# Multiple comparisons correction
res_mc["p-corr"] = pg.multicomp(res_mc["p-unc"].values, alpha=0.05, method="fdr_bh")[1]


# STATISTICS on KSD
## Simple effects:: ANOVA on each level of sigma
res_aov_ksd, ass_norm_ksd, ass_sph_ksd = [], [], []
for sigma in set(df_avg.sigmath):
    for bif in ["prebif", "postbif"]:
        df_KSDmin_cond = df_KSDmin.loc[(df_KSDmin["sigmath"] == sigma) & (df_KSDmin["bif"] == bif)]

        ass_norm.append(pg.normality(df_KSDmin_cond, dv="rPLV", group="th"))
        ass_sph.append(pg.sphericity(df_KSDmin_cond, dv="rPLV"))

        test = pg.friedman(data=df_KSDmin_cond, dv="rPLV", within="th", subject="subject")
        res_aov_ksd.append([sigma, bif] + list(test.values[0]))

res_aov_ksd = pd.DataFrame(res_aov_ksd, columns=["sigmath", "bif"]+list(test.columns.values))
res_aov_ksd["p-corr"] = pg.multicomp(pvals=res_aov_ksd["p-unc"].values, method="fdr_bh")[1]

## Multiple comparisons on sigma==0.022: wilcoxon (due to small sample size)
res_mc_ksd = pd.DataFrame()
for sigma in set(df_avg.sigmath):
    for bif in ["prebif", "postbif"]:
        df_max_cond = df_KSDmin.loc[(df_KSDmin["sigmath"] == sigma) & (df_KSDmin["bif"] == bif)]
        if res_aov_ksd["p-corr"].loc[(res_aov_ksd["sigmath"] == sigma) & (res_aov_ksd["bif"] == bif)].values < 0.05:
            test = pg.pairwise_tests(df_KSDmin_cond, dv="rPLV", within="th", subject="subject", effsize="cohen", parametric=False)
            test["sigmath"], test["bif"] = sigma, bif
            res_mc_ksd = res_mc.append(test)

res_mc_ksd = pd.DataFrame(res_mc_ksd, columns=list(test.columns.values))
# Multiple comparisons correction
res_mc_ksd["p-corr"] = pg.multicomp(res_mc_ksd["p-unc"].values, alpha=0.05, method="fdr_bh")[1]



#      FIGURE R1        ################
structure_th = ["woTh", "Th",  "pTh"]

df_groupavg = df_avg.groupby(["model", "th", "cer", "g", "sigmath"]).mean().reset_index()

# Colours
cmap_s, cmap_p = px.colors.qualitative.Set1, px.colors.qualitative.Pastel1
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2

c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.9

max_g_2show, bif_val = 60, 7

fig = make_subplots(rows=5, cols=3, column_widths=[0.2, 0.6, 0.2], shared_xaxes=True,
                    horizontal_spacing=0.125, vertical_spacing=0.03,
                    subplot_titles=["<b>Scenario 1</b> Active", "", "<b>Scenario 2</b> Relay", "", "", "", "", "", "",
                                    "<b>Scenario 3</b> Trans. Low", "", "<b>Scenario 4</b> Trans. High",  "", "", ""])

for i, th in enumerate(structure_th):
    # ADD LINEPLOTS
    name = "without Thalamus" if th == "woTh" else "single Thalamus" if th == "Th" else "parcelled Thalamus"
    c = c1 if th == "woTh" else c2 if th == "Th" else c3

    # Plot rPLV - scenarios 1 y 2
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 0.022) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=th, mode="lines",
                   line=dict(width=4, color=c), opacity=opacity, showlegend=True), row=1, col=2)

    # ADD BOXPLOTS sigma 0.022
    # datapoints - prebif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 0.022) & (df_max["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=1, col=1)
    # datapoints - postbif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 0.022) & (df_max["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=1, col=3)

    # Plot dFC_KSD - scenarios 1 y 2
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 0.022) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, legendgroup=th, mode="lines",
                              line=dict(dash='solid', color=c, width=2), opacity=opacity, showlegend=False, visible=True),
                   row=2, col=2)

    # ADD BOXPLOTS sigma 0.022
    # datapoints - prebif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 0.022) & (df_KSDmin["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=2, col=1)
    # datapoints - postbif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 0.022) & (df_KSDmin["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=2, col=3)

    # Plot rPLV - scenarios 3 y 4
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 2.2e-08) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=th, mode="lines",
                   line=dict(width=4, color=c), opacity=opacity, showlegend=False), row=4, col=2)

    # ADD BOXPLOTS sigma 0
    # datapoints - prebif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 2.2e-08) & (df_max["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=4, col=1)
    # datapoints - postbif
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigmath"] == 2.2e-08) & (df_max["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=4, col=3)

    # Plot dFC_KSD - scenarios 3 y 4
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigmath"] == 2.2e-08) & (df_avg["g"] < max_g_2show)]
    fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, name=name, legendgroup=th, mode="lines",
                              line=dict(dash='solid', color=c, width=2), opacity=opacity, showlegend=False, visible=True),
                   row=5, col=2)

    # ADD BOXPLOTS sigma 0.022
    # datapoints - prebif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 2.2e-08) & (df_KSDmin["bif"] == "prebif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=5, col=1)
    # datapoints - postbif
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigmath"] == 2.2e-08) & (df_KSDmin["bif"] == "postbif")]
    fig.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=5, col=3)

c4, c5 = "gray", "dimgray" #cmap_s2[-1], cmap_p2[-1]
# Plot CX bifurcation
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_cx, name="cortical ROIs",
                         legendgroup="cortical ROIs", mode="lines",
                         line=dict(width=4, color=c4), showlegend=False), row=3, col=2)

fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_cx, name="cortical ROIs",
                         legendgroup="cortical ROIs", mode="lines",
                         line=dict(width=4, color=c4), showlegend=False), row=3, col=2)

# Plot TH bifurcation
fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.max_th, name="thalamic ROIs",
                         legendgroup="thalamic ROIs", mode="lines",
                         line=dict(width=2, dash='dot', color=c5),
                         showlegend=False), row=3, col=2)

fig.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.min_th, name="thalamic ROIs",
                         legendgroup="thalamic ROIs", mode="lines",
                         line=dict(width=2, dash='dot', color=c5),
                         showlegend=False), row=3, col=2)

fig.update_layout(template="plotly_white",  # title="Importance of thalamus parcellation and input (10 subjects)",
                  yaxis1=dict(title=r"$\text{max. } r_{PLV}$"),  # , range=[0.25, 0.65]),
                  yaxis2=dict(title="$r_{PLV}$", color="black"),  #, range=[0, 0.5]),
                  yaxis3=dict(title=r"$\text{max. } r_{PLV}$"),  #, range=[0.25, 0.65]),

                  yaxis4=dict(title=r"$\text{min. KSD}$"),
                  yaxis5=dict(title="$KSD$"),  # , range=[0.35, 1]),
                  yaxis6=dict(title=r"$\text{min. KSD}$"),

                  yaxis8=dict(title="Voltage"),  #, range=[0, 0.5]),

                  yaxis10=dict(title=r"$\text{max. } r_{PLV}$"),  # , range=[0.25, 0.65]),
                  yaxis11=dict(title="$r_{PLV}$", color="black"),  # , range=[0, 0.5]),
                  yaxis12=dict(title=r"$\text{max. } r_{PLV}$"),  # , range=[0.25, 0.65]),

                  yaxis13=dict(title=r"$\text{min. KSD}$"),
                  yaxis14=dict(title="$KSD$"),  # , range=[0.35, 1]),
                  yaxis15=dict(title=r"$\text{min. KSD}$"),

                  xaxis14=dict(title="Coupling factor (g)"),

                  legend=dict(orientation="h", xanchor="right", x=0.8, yanchor="bottom", y=1.05), boxgroupgap=0.5,
                  height=700, width=900)


## ADD scenarios references
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=0.5, row=1, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=1, row=2, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=0.5, row=4, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
fig.add_shape(x0=0, x1=bif_val, y0=0, y1=1, row=5, col=2, line=dict(color="lightgray"), fillcolor="lightgray", opacity=0.3)
# s1, s2, s3, s4 = cmap_p[2], cmap_p[4], cmap_p[1], cmap_p[5]
# op_scenario = 0.35
# fig.add_vrect(x0=0, x1=bif_val, row=1, col=2, fillcolor=s1, opacity=op_scenario, line_width=0)
# fig.add_vrect(x0=bif_val, x1=max_g_2show, row=1, col=2, fillcolor=s2, opacity=op_scenario, line_width=0)
#
# fig.add_vrect(x0=0, x1=bif_val, row=2, col=2, fillcolor=s1, opacity=op_scenario, line_width=0)
# fig.add_vrect(x0=bif_val, x1=max_g_2show, row=2, col=2, fillcolor=s2, opacity=op_scenario, line_width=0)
#
# fig.add_vrect(x0=0, x1=bif_val, row=4, col=2, fillcolor=s3, opacity=op_scenario, line_width=0)
# fig.add_vrect(x0=bif_val, x1=max_g_2show, row=4, col=2, fillcolor=s4, opacity=op_scenario, line_width=0)
#
# fig.add_vrect(x0=0, x1=bif_val, row=5, col=2, fillcolor=s3, opacity=op_scenario, line_width=0)
# fig.add_vrect(x0=bif_val, x1=max_g_2show, row=5, col=2, fillcolor=s4, opacity=op_scenario, line_width=0)

# fig.show("browser")
#
# ## ADD significance lines
# # Annotations for significance
# basex_pth, basey_pth, y_add, ast_add = 0.944, 0.72, 0.02, 0.025
# # Long bar
# fig.add_shape(type="line", x0=basex_pth-0.11, x1=basex_pth, y0=basey_pth+y_add, y1=basey_pth+y_add, xref="paper", yref="paper", line=dict(color="black", width=2))
# fig.add_annotation(dict(font=dict(color="black", size=12), x=basex_pth - 0.045, y=basey_pth + y_add + ast_add, showarrow=False, text="*", xref="paper", yref="paper"))
# # Short bar
# fig.add_shape(type="line", x0=basex_pth-0.055, x1=basex_pth, y0=basey_pth, y1=basey_pth, xref="paper", yref="paper", line=dict(color="black", width=2))
# fig.add_annotation(dict(font=dict(color="black", size=12), x=basex_pth - 0.015, y=basey_pth + ast_add, showarrow=False, text="*", xref="paper", yref="paper"))


pio.write_image(fig, file=folder + "/PAPER-R1_lineSpace-boxplots_PLV-KSD.svg", engine="kaleido")
pio.write_html(fig, file=folder + "/PAPER-R1_lineSpace-boxplots_PLV-KSD.html", auto_open=True, include_mathjax="cdn")


