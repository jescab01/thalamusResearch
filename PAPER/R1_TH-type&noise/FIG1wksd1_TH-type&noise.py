'''
Creating figures and calculating statistics for ThalamusResearch
A priori using already computed gexplore_data: to get an idea of the plotting.

TODO after the idea is ready recompute calculus

  -  WITH OR WITHOUT THALAMUS  -
'''

import pandas as pd
import pingouin as pg

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

simulations_tag = "PSEmpi_JRstd0.022TH-m10d04y2022-t21h.16m.56s"
folder = 'PAPER\\R1_TH-type&noise\\' + simulations_tag + '\\'

df = pd.read_csv(folder + "results.csv")

# Average out repetitions
df_avg = df.groupby(["subject", "model", "th", "cer", "g", "sigma"]).mean().reset_index()

# Calculate max rPLV per subject and condition
df_max = df_avg.groupby(["subject", "model", "th", "cer", "sigma"]).max(["rPLV"]).reset_index()
df_max.to_csv(folder + "df_max.csv")

# Calculate min KSD per subject and condition
df_KSDmin = df_avg.groupby(["subject", "model", "th", "cer", "sigma"]).min(["dFC_KSD"]).reset_index()


# STATISTICS
## Simple effects
# ANOVA on each level of sigma
res_aov, ass_norm, ass_sph = [], [], []
for cond in [0, 0.022]:
    df_max_cond = df_max.loc[(df_max["sigma"] == cond)]

    ass_norm.append(pg.normality(df_max_cond, dv="rPLV", group="th"))
    ass_sph.append(pg.sphericity(df_max_cond, dv="rPLV"))

    test = pg.rm_anova(data=df_max_cond, dv="rPLV", within="th", subject="subject", correction=True)
    res_aov.append([cond] + list(test.values[0]))

res_aov = pd.DataFrame(res_aov, columns=["cond", "source", "ddof1", "ddof2", "F", "p-unc", "p-GG-corr", "ng2", "eps", "sphericity", "W-spher", "p-spher"])

# Multiple comparisons on sigma==0.022: wilcoxon (due to small sample size)
res_mc = pg.pairwise_tests(df_max, dv="rPLV", within=["sigma", "th"], subject="subject", effsize="cohen", parametric=False).iloc[4:]
res_mc = res_mc.append(pg.pairwise_tests(df_max, dv="rPLV", within=["th", "sigma"], subject="subject", effsize="cohen", parametric=False).iloc[4:])

# Multiple comparisons correction
res_mc["p-corr"] = pg.multicomp(res_mc["p-unc"].values, alpha=0.05, method="fdr_bh")[1]





#      FIGURE 1        ################
structure_th = ["woTh", "Th",  "pTh"]

df_groupavg = df_avg.groupby(["model", "th", "cer", "g", "sigma"]).mean().reset_index()

# Colours
cmap_s, cmap_p = px.colors.qualitative.Set1, px.colors.qualitative.Pastel1
cmap_s2, cmap_p2 = px.colors.qualitative.Set2, px.colors.qualitative.Pastel2

c1, c2, c3 = cmap_s2[1], cmap_s2[0], cmap_s2[2]  # "#fc8d62", "#66c2a5", "#8da0cb"  # red, green, blue
opacity = 0.9

fig1 = make_subplots(rows=4, cols=2, row_titles=["Passive", "Active", "Passive", "Active"], column_widths=[0.8, 0.2],
                     specs=[[{}, {}], [{}, {}], [{}, {}], [{}, {}]])

for i, th in enumerate(structure_th):
    # ADD LINEPLOTS
    name = "without Thalamus" if th == "woTh" else "single Thalamus" if th == "Th" else "parcelled Thalamus"
    c = c1 if th == "woTh" else c2 if th == "Th" else c3

    # Plot rPLV - passive
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0)]
    fig1.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=th, mode="lines",
                   line=dict(width=4, color=c), opacity=opacity, showlegend=True), row=1, col=1)
    # Plot rPLV - active
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0.022)]
    fig1.add_trace(
        go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name=name, legendgroup=th, mode="lines",
                   line=dict(width=4, color=c), opacity=opacity, showlegend=False), row=2, col=1)

    # ADD BOXPLOTS
    # datapoints - Passive
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigma"] == 0)]
    fig1.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=1, col=2)
    # datapoints - Active
    df_sub = df_max.loc[(df_max["th"] == th) & (df_max["sigma"] == 0.022)]
    fig1.add_trace(go.Box(x=df_sub.th.values, y=df_sub.rPLV.values, boxpoints="all", fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, opacity=0.8, showlegend=False), row=2, col=2)


    # Plot dFC_KSD - passive
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0)]
    fig1.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, name=name, legendgroup=th, mode="lines",
                              line=dict(dash='solid', color=c, width=2), opacity=opacity, showlegend=False, visible=True),
                   row=3, col=1)

    # Plot dFC_KSD - active
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0.022)]
    fig1.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, legendgroup=th, mode="lines",
                              line=dict(dash='solid', color=c, width=2), opacity=opacity, showlegend=False, visible=True),
                   row=4, col=1)

    # ADD BOXPLOTS
    # datapoints - Passive rPLV
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigma"] == 0)]
    fig1.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", opacity=opacity, fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, showlegend=False), row=3, col=2)
    # datapoints - Active
    df_sub = df_KSDmin.loc[(df_KSDmin["th"] == th) & (df_KSDmin["sigma"] == 0.022)]
    fig1.add_trace(go.Box(x=df_sub.th.values, y=df_sub.dFC_KSD.values, boxpoints="all", opacity=opacity, fillcolor=c, line=dict(color="black", width=1),
                          name=th, legendgroup=th, showlegend=False), row=4, col=2)


fig1.update_layout(template="plotly_white",  # title="Importance of thalamus parcellation and input (10 subjects)",
                   yaxis1=dict(title="$r_{PLV}$", color="black", range=[0, 0.5]),
                   yaxis2=dict(title=r"$\text{max. } r_{PLV}$", range=[0.25, 0.65]),
                   yaxis3=dict(title="$r_{PLV}$", color="black", range=[0, 0.5]),
                   yaxis4=dict(title=r"$\text{max. } r_{PLV}$", range=[0.25, 0.65]),

                   yaxis5=dict(title="$KSD$", range=[0.35, 1]),
                   yaxis6=dict(title=r"$\text{min. KSD}$"),
                   yaxis7=dict(title="$KSD$", range=[0.35, 1]), xaxis7=dict(title="Coupling factor (g)"),
                   yaxis8=dict(title=r"$\text{min. KSD}$"),

                   legend=dict(orientation="h", xanchor="right", x=0.9, yanchor="bottom", y=1.02), boxgroupgap=0.5,
                   height=700, width=900)

# Annotations for significance
basex_pth, basey_pth, y_add, ast_add = 0.944, 0.72, 0.02, 0.025
# Long bar
fig1.add_shape(type="line", x0=basex_pth-0.11, x1=basex_pth, y0=basey_pth+y_add, y1=basey_pth+y_add, xref="paper", yref="paper", line=dict(color="black", width=2))
fig1.add_annotation(dict(font=dict(color="black", size=12), x=basex_pth - 0.045, y=basey_pth + y_add + ast_add, showarrow=False, text="*", xref="paper", yref="paper"))
# Short bar
fig1.add_shape(type="line", x0=basex_pth-0.055, x1=basex_pth, y0=basey_pth, y1=basey_pth, xref="paper", yref="paper", line=dict(color="black", width=2))
fig1.add_annotation(dict(font=dict(color="black", size=12), x=basex_pth - 0.015, y=basey_pth + ast_add, showarrow=False, text="*", xref="paper", yref="paper"))

pio.write_image(fig1, file=folder + "/PAPER1_lineSpace-boxplots_PLV-KSD.svg", engine="kaleido")
pio.write_html(fig1, file=folder + "/PAPER1_lineSpace-boxplots_PLV-KSD.html", auto_open=True, include_mathjax="cdn")



# if g_sel:
#     for c, g in enumerate(g_sel):
#         fig_lines.add_vline(x=g, row=1, col=1+i) #line_color=px.colors.qualitative.Set2[c])
