
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px



# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\thalamusResearch\mpi_jr-vMultiStim\PSE\\'
simulations_tag = "PSEmpi_MultiStim-m08d10y2022-t15h.39m.51s"  # Tag cluster job
df = pd.read_pickle(main_folder + simulations_tag + "/results.pkl")


### Compute a Factorial ANOVA to check the influence of each variable on KSD




## What are the top10 KSD; what are the top10 rPLV?




### Plot the most relevant varibles














# Average out repetitions
df_avg = df.groupby(["subject", "model", "th", "cer", "g", "p", "sigma"]).mean().reset_index()


# Plot min-max

## Plot paramSpaces
for subject in list(set(df_avg.subject)):

    title = subject + "_paramSpace_bif"

    fig = make_subplots(rows=3, cols=3,
                        row_titles=("Without Thalamus", "Thalamus - Single node", "Thalamus - Parcellated"),
                        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], column_titles=["PSE", "bifs_th", "bifs_cx"],
                        shared_yaxes=True, shared_xaxes=True)

    for j, th in enumerate(structure_th):

        subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == th)]

        sl = True if j == 0 else False

        fig.add_trace(
            go.Heatmap(z=subset.rPLV, x=subset.p, y=subset.g, colorscale='RdBu', reversescale=True,
                       zmin=-0.5, zmax=0.5, showscale=False, colorbar=dict(thickness=7)),
            row=(1 + j), col=1)

        fig.add_trace(
            go.Heatmap(z=subset.max_th-subset.min_th, x=subset.p, y=subset.g, colorscale='Viridis',
                       showscale=sl, colorbar=dict(thickness=7), zmin=0, zmax=0.14),
            row=(1 + j), col=3)

        fig.add_trace(
            go.Heatmap(z=subset.max_cx-subset.min_cx, x=subset.p, y=subset.g, colorscale='Viridis',
                       showscale=sl, colorbar=dict(thickness=7), zmin=0, zmax=0.14),
            row=(1 + j), col=2)

    fig.update_layout(title_text=title)
    pio.write_html(fig, file=main_folder + simulations_tag + "/" + title + "-g&p_bif.html", auto_open=True)







## PLOT AVERAGE LINES -

df_groupavg = df.groupby(["model", "th", "cer", "g", "sigma"]).mean().reset_index()
df_groupstd = df.groupby(["model", "th", "cer", "g", "sigma"]).std().reset_index()

## Line plots for wo, w, and wp thalamus. Solid lines for active and dashed for passive.
cmap_s = px.colors.qualitative.Set1
cmap_p = px.colors.qualitative.Pastel1

# Graph objects approach
fig_lines = make_subplots(rows=1, cols=3, subplot_titles=("without Thalamus", "Thalamus single node", "Thalamus parcellated"),
                          specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
                          shared_yaxes=True, shared_xaxes=True, x_title="Coupling factor")

for i, th in enumerate(structure_th):

    sl = True if i < 1 else False

    # Plot rPLV - active
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0.022)]
    fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name='rPLV - active', legendgroup='rPLV - active', mode="lines",
                                   line=dict(width=4, color=cmap_p[1]), showlegend=sl), row=1, col=1+i)

    # Plot rPLV - passive
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0)]
    fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.rPLV, name='rPLV - passive', legendgroup='rPLV - passive', mode="lines",
                                   line=dict(dash='dash', color=cmap_s[1]), showlegend=sl), row=1, col=1+i)

    # Plot dFC_KSD - active
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0.022)]
    fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, name='dFC_KSD - active', legendgroup='dFC_KSD - active', mode="lines",
                                   line=dict(width=4, color=cmap_p[0]), showlegend=sl), secondary_y=True, row=1, col=1+i)

    # Plot dFC_KSD - passive
    df_sub_avg = df_groupavg.loc[(df_avg["th"] == th) & (df_avg["sigma"] == 0)]
    fig_lines.add_trace(go.Scatter(x=df_sub_avg.g, y=df_sub_avg.dFC_KSD, name='dFC_KSD - passive', legendgroup='dFC_KSD - passive', mode="lines",
                                   line=dict(dash='dash', color=cmap_s[0]), showlegend=sl), secondary_y=True, row=1, col=1+i)

fig_lines.update_layout(template="plotly_white", title="Importance of thalamus parcellation and input (avg. 10 subjects)",
                        yaxis1=dict(title="<b>rPLV<b>", color=cmap_s[1]), yaxis2=dict(title="<b>KSD<b>", color=cmap_p[0]),
                        yaxis3=dict(title="<b>rPLV<b>", color=cmap_s[1]), yaxis4=dict(title="<b>KSD<b>", color=cmap_p[0]),
                        yaxis5=dict(title="<b>rPLV<b>", color=cmap_s[1]), yaxis6=dict(title="<b>KSD<b>", color=cmap_p[0]))

pio.write_html(fig_lines, file=main_folder + simulations_tag + "/lineSpace-g&FC.html", auto_open=True)








## Maximum rPLV - statistical group comparisons
# Extract best rPLV per subject and structure
df_max = df_avg.groupby(["subject", "model", "th", "cer", "ct", "act"]).max().reset_index()


from statsmodels.stats.anova import AnovaRM
anova = AnovaRM(df_max, depvar="rPLV", subject="subject", within=["th", "model"]).fit().anova_table

import pingouin as pg
pg.plot_paired(df_max, dv="rPLV", within="th", subject="subject",)
pg.plot_paired(df_max, dv="rPLV", within="th", subject="subject",)



# JR stats
df_max_jr = df_max.loc[df_max["model"] == "jr"]
anova_jr = AnovaRM(df_max_jr, depvar="rPLV", subject="subject", within=["ct", "act", "th", "cer"]).fit().anova_table

pwc_jr_ct = pg.pairwise_ttests(df_max_jr, dv="rPLV", within=["ct"], subject="subject")
pwc_jr_th = pg.pairwise_ttests(df_max_jr, dv="rPLV", within=["th"], subject="subject")

fig = px.box(df_max_jr, x="th", y="rPLV", color="cer", facet_col="ct", facet_row="act",
             category_orders={"cer": ["woCer", "Cer", "pCer"], "th": ["pTh", "Th", "woTh"]}, title="JR model -")
pio.write_html(fig, file=main_folder + simulations_tag + "/JR_rPLV_performance-boxplots.html", auto_open=True)



# JRD stats
df_max_jrd = df_max.loc[df_max["model"] == "jrd"]
anova_jrd = AnovaRM(df_max_jrd, depvar="rPLV", subject="subject", within=["ct", "act", "th", "cer"]).fit().anova_table

pwc_jrd_ct = pg.pairwise_ttests(df_max_jrd, dv="rPLV", within=["ct"], subject="subject")
pwc_jrd_act = pg.pairwise_ttests(df_max_jrd, dv="rPLV", within=["act"], subject="subject")
# pwc_jrd_th = pg.pairwise_ttests(df_max_jrd, dv="rPLV", within=["th"], subject="subject") Just 2 categories


df_max_jrd = df_max.loc[df_max["model"] == "jrd"]
fig = px.box(df_max_jrd, x="th", y="rPLV", color="cer", facet_col="ct", facet_row="act",
             category_orders={"cer": ["woCer", "Cer", "pCer"], "th": ["pTh", "Th"]}, title="JRD model -")
pio.write_html(fig, file=main_folder + simulations_tag + "/JRD_rPLV_performance-boxplots.html", auto_open=True)












# dataframe for lineplot with err bars
df_line = df_max.groupby(["th", "cer", "ct", "act"]).mean()
df_line["std"] = df_max.groupby(["th", "cer", "ct", "act"]).std()["rPLV"]
df_line = df_line.reset_index()

fig = px.line(df_line, x="cer", y="rPLV", color="th", facet_col="ct", facet_row="act", error_y="std")
fig.show(renderer="browser")



anova = pg.rm_anova(df_max, dv="rPLV", subject="subject", within=["ct", "th"])

anova = pg.rm_anova(df_max, dv="rPLV", within=["ct", "act"], subject="subject",)
anova = pg.rm_anova(df_max, dv="rPLV", within=["th", "cer"], subject="subject",)




pwc = pg.pairwise_ttests(df_max, dv="rPLV", within=["ct"], subject="subject")
pwc = pg.pairwise_ttests(df_max, dv="rPLV", within=["th"], subject="subject")








for mode in modes:
    df_temp = df.loc[df["Mode"] == mode]
    df_temp = df_temp.groupby(["G", "speed"]).mean().reset_index()
    (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV

    specific_folder = main_folder + "\\PSE_allWPs-AVGg" + str(g) + "s" + str(s) + "_" + mode + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss")
    os.mkdir(specific_folder)

    for subj in list(set(df.Subject)):

        # subset data per mode and subject
        df_temp = df.loc[(df["Subject"] == subj) & (df["Mode"] == mode)]

        # Avg repetitions
        df_temp = df_temp.groupby(["G", "speed", "band"]).mean().reset_index()
        df_temp.drop("rep", inplace=True, axis=1)

        # Calculate WP
        (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV

        name = subj + "_" + mode + "-g" + str(g) + "s" + str(s)

        # save data
        df_temp.to_csv(specific_folder + "/" + name +"-3reps.csv")

        # plot paramspace
        WPplot(df_temp, z=0.5, title=name, type="linear", folder=specific_folder, auto_open=False)


# Plot 3 by 3 Alpha PSEs
for subj in list(set(df.Subject)):

    fig_thcer = make_subplots(rows=3, cols=3, column_titles=("Parcelled Thalamus", "Single node Thalamus", "Without Thalamus"),
                           row_titles=("Parcelled Cerebellum", "Single node Cerebellum", "Without Cerebellum"),
                        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title="Conduction speed (m/s)", y_title="Coupling factor")

    df_sub = df.loc[(df["band"] == "3-alpha") & (df["Subject"] == subj)]

    for i, mode in enumerate(modes):

        df_temp = df_sub.loc[df_sub["Mode"] == mode]

        df_temp = df_temp.groupby(["G", "speed", "band"]).mean().reset_index()
        df_temp.drop("rep", inplace=True, axis=1)

        fig_thcer.add_trace(go.Heatmap(z=df_temp.rPLV, x=df_temp.speed, y=df_temp.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
                             reversescale=True, zmin=-0.5, zmax=0.5), row=(i+3)//3, col=i%3+1)


    fig_thcer.update_layout(
        title_text='FC correlation (empirical - simulated data) by Coupling factor and Conduction speed || %s' % subj)
    pio.write_html(fig_thcer, file=main_folder + "/ThCer_paramSpace-g&s_%s.html" % subj, auto_open=True)

