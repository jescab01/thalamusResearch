
import os

import pandas as pd
import time

import plotly.graph_objects as go  # for gexplore_data visualisation
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px


def WPplot(df, z=None, title=None, type="linear", folder="figures", auto_open="True"):

    fig_fc = make_subplots(rows=1, cols=6, subplot_titles=("Delta", "Theta", "Alpha", "Beta", "Gamma", "Power"),
                        specs=[[{}, {}, {}, {}, {}, {}]], shared_yaxes=True, shared_xaxes=True,
                        x_title="Conduction speed (m/s)", y_title="Coupling factor")

    df_sub = df.loc[df["band"]=="1-delta"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', colorbar=dict(title="Pearson's r"),
                             reversescale=True, zmin=-z, zmax=z), row=1, col=1)

    df_sub = df.loc[df["band"] == "2-theta"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=2)
    df_sub = df.loc[df["band"] == "3-alpha"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=3)
    df_sub = df.loc[df["band"] == "4-beta"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=4)
    df_sub = df.loc[df["band"] == "5-gamma"]
    fig_fc.add_trace(go.Heatmap(z=df_sub.rPLV, x=df_sub.speed, y=df_sub.G, colorscale='RdBu', reversescale=True, zmin=-z, zmax=z,
                              showscale=False), row=1, col=5)

    fig_fc.add_trace(go.Heatmap(z=df_sub.bModule, x=df_sub.speed, y=df_sub.G, colorscale='Viridis',
                             reversescale=True), row=1, col=6)

    fig_fc.update_layout(yaxis1_type=type,yaxis2_type=type,yaxis3_type=type,yaxis4_type=type,yaxis5_type=type,
        title_text='FC correlation (empirical - simulated gexplore_data) by Coupling factor and Conduction speed || %s' % title)
    pio.write_html(fig_fc, file=folder + "/paramSpace-g&s_%s.html" % title, auto_open=auto_open)



# Define PSE folder
main_folder = 'E:\\LCCN_Local\PycharmProjects\\thalamusResearch\mpi_jrwc-vNoise\PSE\\'
simulations_tag = "PSEmpi_Log-JRWCstd0.022-m08d12y2022-t13h.52m.41s"  # Tag cluster job
df = pd.read_csv(main_folder + simulations_tag + "/results.csv")

modes = [""]
structure_th = ["woTh", "Th", "pTh"]
structure_cer = ["pCer"]

# Average out repetitions
df_avg = df.groupby(["subject", "th", "cer", "g_jr", "g_wc", "std_n"]).mean().reset_index()

## Plot paramSpace
for subject in list(set(df_avg.subject)):

    title = subject + "_paramSpace-std0.022"

    fig = make_subplots(rows=1, cols=3,
                        column_titles=("Without Thalamus", "Thalamus - Single node", "Thalamus - Parcellated"),
                        specs=[[{}, {}, {}]],
                        shared_yaxes=True, shared_xaxes=True,
                        x_title="g_wc (from thalamus nodes)", y_title="g_jr (from cortex nodes)")

    for j, th in enumerate(structure_th):

        subset = df_avg.loc[(df_avg["subject"] == subject) & (df_avg["th"] == th)]

        sl = True if j == 0 else False

        fig.add_trace(
            go.Heatmap(z=subset.rPLV, x=subset.g_wc, y=subset.g_jr, colorscale='RdBu', reversescale=True,
                       zmin=-0.5, zmax=0.5, showscale=sl, colorbar=dict(thickness=7)),
            row=1, col=(1 + j))

    fig.update_layout(title_text=title, xaxis=dict(type='log'),
                      xaxis2=dict(type='log'), xaxis3=dict(type='log'))
    pio.write_html(fig, file=main_folder + simulations_tag + "/" + title + "-g&s.html", auto_open=True)



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

        # subset gexplore_data per mode and subject
        df_temp = df.loc[(df["Subject"] == subj) & (df["Mode"] == mode)]

        # Avg repetitions
        df_temp = df_temp.groupby(["G", "speed", "band"]).mean().reset_index()
        df_temp.drop("rep", inplace=True, axis=1)

        # Calculate WP
        (g, s) = df_temp.groupby(["G", "speed"]).mean().idxmax(axis=0).rPLV

        name = subj + "_" + mode + "-g" + str(g) + "s" + str(s)

        # save gexplore_data
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
        title_text='FC correlation (empirical - simulated gexplore_data) by Coupling factor and Conduction speed || %s' % subj)
    pio.write_html(fig_thcer, file=main_folder + "/ThCer_paramSpace-g&s_%s.html" % subj, auto_open=True)

