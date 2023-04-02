
import numpy as np
from tvb.simulator.lab import connectivity


ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data3\\"

subj_ids = [35, 49, 50, 58, 59, 64, 65, 71, 75, 77]
subjects = ["NEMOS_0" + str(id) for id in subj_ids]

# Define regions implicated in Functional analysis: remove  Cerebelum, Thalamus, Caudate (i.e. subcorticals)
cortical_rois = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_2_L',
                 'Frontal_Sup_2_R', 'Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L',
                 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_2_L', 'Frontal_Inf_Orb_2_R',
                 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R',
                 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R',
                 'OFCmed_L', 'OFCmed_R', 'OFCant_L', 'OFCant_R', 'OFCpost_L',
                 'OFCpost_R', 'OFClat_L', 'OFClat_R', 'Insula_L', 'Insula_R',
                 'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Mid_L',
                 'Cingulate_Mid_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                 'ParaHippocampal_R', 'Calcarine_L',
                 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L',
                 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R',
                 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R',
                 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R',
                 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Heschl_L', 'Heschl_R',
                 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L',
                 'Temporal_Inf_R']


corrs = []
for emp_subj in subjects:

    conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2_pass.zip")

    # load text with FC rois; check if match SC
    FClabs = list(np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/roi_labels.txt", dtype=str))
    FC_cortex_idx = [FClabs.index(roi) for roi in
                     cortical_rois]  # find indexes in FClabs that matches cortical_rois
    SClabs = list(conn.region_labels)
    SC_cortex_idx = [SClabs.index(roi) for roi in cortical_rois]

    sc_emp = conn.weights[:, SC_cortex_idx][SC_cortex_idx]

    # Load empirical data to make simple comparisons
    plv_emp = \
        np.loadtxt(ctb_folder + "FCavg_" + emp_subj + "/3-alpha_plv_avg.txt", delimiter=',')[:,
        FC_cortex_idx][
            FC_cortex_idx]

    t3 = np.zeros(shape=(2, len(plv_emp) ** 2 // 2 - len(plv_emp) // 2))
    t3[0, :] = plv_emp[np.triu_indices(len(plv_emp), 1)]
    t3[1, :] = sc_emp[np.triu_indices(len(plv_emp), 1)]
    sc_r = np.corrcoef(t3)[0, 1]

    corrs.append(sc_r)

np.average(corrs)