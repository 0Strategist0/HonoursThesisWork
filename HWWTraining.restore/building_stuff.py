import pandas as pd
import numpy as np

shuffled_full_data = pd.read_pickle("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/shuffled_data.pkl")

for name in shuffled_full_data.columns:
    if "weight_" in name:
        shuffled_full_data[name] *= shuffled_full_data["weight"]

print(np.sum(shuffled_full_data["weight_sm"]))

shuffled_full_data[['runNumber', 'eventNumber', 'lep0_id', 'lep1_id', 'lep0_pt', 'lep1_pt',
       'lep0_eta', 'lep1_eta', 'lep0_phi', 'lep1_phi', 'lep0_m', 'lep1_m',
       'jet0_pt', 'jet1_pt', 'jet0_eta', 'jet1_eta', 'jet0_phi', 'jet1_phi',
       'jet0_m', 'jet1_m', 'met_et', 'met_phi', 'Mll', 'Ptll', 'DPhill',
       'DEtall', 'DYll', 'Mjj', 'Ptjj', 'DPhijj', 'DEtajj', 'DYjj', 'nJets',
       'sqrtHT', 'METSig', 'Ml0j0', 'Ml0j1', 'Ml1j0', 'Ml1j1', 'weight_sm',
       'weight_cHW_pos0p01', 'weight_cHW_pos0p02', 'weight_cHW_pos0p05',
       'weight_cHW_pos0p1', 'weight_cHW_pos0p2', 'weight_cHW_pos0p5',
       'weight_cHW_pos1p0', 'weight_cHW_pos2p0', 'weight_cHW_neg0p01',
       'weight_cHW_neg0p02', 'weight_cHW_neg0p05', 'weight_cHW_neg0p1',
       'weight_cHW_neg0p2', 'weight_cHW_neg0p5', 'weight_cHW_neg1p0',
       'weight_cHW_neg2p0']].to_pickle("/home/kye/projects/ctb-stelzer/kye/HWWTraining/Data/AnalysisTest/full_data.pkl")

print(shuffled_full_data.columns)