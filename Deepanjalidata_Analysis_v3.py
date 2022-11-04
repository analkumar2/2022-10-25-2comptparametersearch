## exec(open('Deepanjalidata_Analysis_v3.py').read())

# v3: Now also handles dat files

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import featuresv31_nonallen_2compt as fts
import plotexpv3 as pex
import os
# from sklearn.decomposition import PCA

aa = 'WT'
bb = 'post'

# //////////////////////////////////////////////////////////////////
# foldername = '../../Raw_data/Deepanjali_data/KO step input cells/' #Name of the folder where files are stored
foldername = f'../../Raw_data/Deepanjali_data/Organized_Anal/{aa}_prepostapa/IF_{bb}apa/'  #Name of the folder where files are stored
# fskip = ['2018_01_15_1.abf', '2018_08_17_1.abf', '2018_08_17_2.abf', '2018_08_17_2_alt.abf', '2018_08_16_2_1.abf', '2018_08_16_1_1.abf', '2018_01_17_3_check.abf'] #List of files that need to be skipped
# fskip = ['2018_01_15_1.abf', '2017_12_20_1.abf', '2018_01_15_1.abf', '2017_11_17_6.abf', '2018_02_13_2.abf']
# fskip = ['2017_12_20_1.abf', '2018_01_15_1.abf', '2017_11_17_6.abf', '2018_02_13_2.abf'] #WT_prepostapa/IF_postapa/
fskip = ['2018_01_24_4.abf'] #KO_prepostapa/IF_postapa/
# //////////////////////////////////////////////////////////////////

def chargingm25(t, R,C):
    return features[f'E_rest_m25'] - R*25e-12*(1-np.exp(-t/R/C))

def chargingm50(t, R,C):
    return features[f'E_rest_m50'] - R*50e-12*(1-np.exp(-t/R/C))


Sno = 0
features_pd = pd.DataFrame()
for filename in os.listdir(foldername):
    # stim1391 = ['Cell 3 of 181016.abf', 'cell 4 of 61016.abf', 'cell 4 of 111016.abf', 'cell 4 of 131016.abf', 'Cell 4 of 181016.abf', 'cell 5 of 61016.abf', 'Cell 5 of 181016.abf', 'Cell 2 of 19_10_2016', 'Cell 1 of 27_10_2016.abf', 'Cell 1 of 14_10_2016.abf', 'Cell 4 of 7_10_2016.abf', 'Cell 6 of 12_10_2016.abf', 'Cell 7 of 12_10_2016.abf', 'Cell 2 of 19_10_2016.abf']
    # if filename in stim1391:
    # stim_start = 139.1e-3
    # stim_end = 639.1e-3
    # else:
    stim_start = 81.3e-3 #in s
    stim_end = 581.3e-3 #in s

    if filename in fskip:
        print(f'{filename} skipped')
        continue

    if filename[-4:] == '.abf' or filename[-4:] == '.dat':
        print(filename)
        features = fts.expfeatures(foldername+filename, stim_start, stim_end, offset=0)
        print(features)
        if features == False:
            continue
        jarjar = pd.DataFrame(data=features,index = [Sno])
        # jarjar = pd.DataFrame.from_dict(data=features)
        features_pd = features_pd.append(jarjar, ignore_index=True)
        Sno = Sno +1
        print(Sno)
        # plt.axvline(x=stim_start)
        # plt.axvline(x=stim_end)
        # plt.axhline(y=features['sagSS_m50'])
        # tt = np.linspace(0,0.5,100)
        # plt.plot(tt+stim_start, chargingm25(tt,features['Input resistance'],features['Cell capacitance']))
        # plt.plot(tt+stim_start, chargingm50(tt,features['Input resistance'],features['Cell capacitance']))
        # plt.plot(*pex.expdata(foldername+filename, -25e-12))
        # plt.plot(*pex.expdata(foldername+filename, -50e-12))
        # plt.show()
    else:
        continue

## features_pd.to_csv('features.csv', sep='\t')
# features_ess_pd= features_pd.dropna(axis='columns')
# features_ess_pd = features_pd[['Cell name', 'Sampling rate', 'stim_start', 'stim_end', 'E_rest_0', 'E_rest_m50', 'E_rest_m25', 'E_rest_150', 'E_rest_300', 'Input resistance', 'Cell capacitance', 'sagSS_m50', 'sagV_m50', 'AP1_amp_1.5e-10', 'APp_amp_1.5e-10', 'AP1_time_1.5e-10', 'APp_time_1.5e-10', 'APavgpratio_amp_1.5e-10', 'AP1_width_1.5e-10', 'APp_width_1.5e-10', 'AP1_thresh_1.5e-10', 'APp_thresh_1.5e-10', 'AP1_lat_1.5e-10', 'ISI1_1.5e-10', 'ISIl_1.5e-10', 'ISIavg_1.5e-10', 'freq_1.5e-10', 'Adptn_id_1.5e-10', 'mAHP_stimend_amp_1.5e-10', 'sAHP_stimend_amp_1.5e-10', 'AHP_AP1_amp_1.5e-10', 'AHP_APp_amp_1.5e-10', 'AHP_AP1_time_1.5e-10', 'AHP_APp_time_1.5e-10', 'Upstroke_AP1_time_1.5e-10', 'Upstroke_APp_time_1.5e-10', 'Upstroke_AP1_amp_1.5e-10', 'Upstroke_APp_amp_1.5e-10', 'Upstroke_AP1_value_1.5e-10', 'Upstroke_APp_value_1.5e-10', 'Downstroke_AP1_time_1.5e-10', 'Downstroke_APp_time_1.5e-10', 'Downstroke_AP1_amp_1.5e-10', 'Downstroke_APp_amp_1.5e-10', 'Downstroke_AP1_value_1.5e-10', 'Downstroke_APp_value_1.5e-10', 'UpDn_AP1_ratio_1.5e-10', 'UpDn_APp_ratio_1.5e-10', 'UpThr_AP1_diff_1.5e-10', 'UpThr_APp_diff_1.5e-10', 'offset_1.5e-10', 'Absoffset_1.5e-10', 'AP1_amp_3e-10', 'APp_amp_3e-10', 'AP1_time_3e-10', 'APp_time_3e-10', 'APavgpratio_amp_3e-10', 'AP1_width_3e-10', 'APp_width_3e-10', 'AP1_thresh_3e-10', 'APp_thresh_3e-10', 'AP1_lat_3e-10', 'ISI1_3e-10', 'ISIl_3e-10', 'ISIavg_3e-10', 'freq_3e-10', 'Adptn_id_3e-10', 'mAHP_stimend_amp_3e-10', 'sAHP_stimend_amp_3e-10', 'AHP_AP1_amp_3e-10', 'AHP_APp_amp_3e-10', 'AHP_AP1_time_3e-10', 'AHP_APp_time_3e-10', 'Upstroke_AP1_time_3e-10', 'Upstroke_APp_time_3e-10', 'Upstroke_AP1_amp_3e-10', 'Upstroke_APp_amp_3e-10', 'Upstroke_AP1_value_3e-10', 'Upstroke_APp_value_3e-10', 'Downstroke_AP1_time_3e-10', 'Downstroke_APp_time_3e-10', 'Downstroke_AP1_amp_3e-10', 'Downstroke_APp_amp_3e-10', 'Downstroke_AP1_value_3e-10', 'Downstroke_APp_value_3e-10', 'UpDn_AP1_ratio_3e-10', 'UpDn_APp_ratio_3e-10', 'UpThr_AP1_diff_3e-10', 'UpThr_APp_diff_3e-10', 'offset_3e-10', 'Absoffset_3e-10', 'freq300to150ratio']]
features_ess_pd = features_pd
features_ess_pd= features_ess_pd.drop(['Sampling rate','stim_start','stim_end', 'E_rest_m50', 'E_rest_m25', 'E_rest_150', 'E_rest_300'], axis='columns')
features_ess_pd.to_csv(f'features_ess_{aa}_{bb}apa.csv', sep='\t')

### features_ess_KO = pd.read_csv('features_ess_KO.csv',sep='\t', index_col=0) #To load a csv to dataframe
