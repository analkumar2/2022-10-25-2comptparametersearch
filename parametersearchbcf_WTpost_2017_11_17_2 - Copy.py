from neo.io import AxonIO
import time
import pandas as pd
import plotexpv3 as pex
import MOOSEModel_2compt_18 as mm

import moose
import rdesigneur as rd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import numpy.random as nr
import featuresv31_nonallen_2compt as fts
from pprint import pprint
import scipy.signal as scs
import sys
import os
import io
import scipy.signal as scs
from pprint import pprint
import scipy
import brute_curvefit as bcf
import pickle
from glob import glob

# Importing the means and std
features_ess_WT = pd.read_csv("features_ess_WT_postapa.csv", sep="\t", index_col=0)
meanfeaturesWT = features_ess_WT.mean()
stdfeaturesWT = features_ess_WT.std()
minfeaturesWT = features_ess_WT.min()
maxfeaturesWT = features_ess_WT.max()

features_ess_KO = pd.read_csv("features_ess_KO_postapa.csv", sep="\t", index_col=0)
meanfeaturesKO = features_ess_KO.mean()
stdfeaturesKO = features_ess_KO.std()
minfeaturesKO = features_ess_KO.min()
maxfeaturesKO = features_ess_KO.max()

# dummy Model
sm_area = 14.2e-9
dummyModel = {
    "Parameters": {
        "notes": "",
        "Morphology": {"sm_len": 6.73077545020806e-05, "sm_diam": 6.73077545020806e-05},
        "Passive": {
            "Cm": 1.19e-10,
            "Rm": 368209825.6770873,
            "Em": -0.035928469851048674,
        },
        "Channels": {
            "Na_Chan": {
                "Gbar": 0.0001374566134032769,
                "Erev": 0.06,
                "Kinetics": "../../Compilations/Kinetics/Na_Chan_Custom4",
                # "KineticVars": {
                #     "m_vhalf_inf": -0.02014902574417892,
                #     "m_slope_inf": 0.008827954894080618,
                #     "m_A": -0.024531733582989323,
                #     "m_F": 9.059013421342436e-05,
                #     "h_vhalf_inf": -0.05891870737718587,
                #     "h_slope_inf": -0.005655091470556401,
                #     "h_A": -0.0384423080178879,
                #     "h_F": 0.0222499049405061 * 0.4,
                # },
            },
            "K_DR_Chan": {
                "Gbar": 1.0502259538910637e-7,
                "Erev": -0.09,
                "Kinetics": "../../Compilations/Kinetics/K_DR_Chan_Custom3",
                # "KineticVars": {
                #     "n_F": 0.00306,
                #     "n_vhalf_inf": 0.013,
                #     # "n_A": 1.26E-02,
                # },
            },
            "K_A_Chan": {
                "Gbar": 5.191082344779805e-07,
                "Erev": -0.09,
                "Kinetics": "../../Compilations/Kinetics/K_A_Chan_Custom3",
                # "KineticVars": {
                #     "n_vhalf_inf": 0.018626436921675663,
                #     "n_A": -0.0013535630783243387,
                #     "l_vhalf_inf": -0.04857356307832434,
                #     "l_cm": 0.057426436921675664,
                # },
            },
            "K_M_Chan": {
                "Gbar": 15e-9,
                "Erev": -0.09,
                "Kinetics": "../../Compilations/Kinetics/K_M_Chan_Custom2",
                # "KineticVars": {
                #     "m_vhalf_inf": -0.035,
                #     "m_slope_inf": 0.001,
                #     "m_F": 2.15e-1 * 0.1,
                # },
            },
            "h_Chan": {
                "Gbar": 5.3739087243907273e-11*10,
                "Erev": -0.04,
                "Kinetics": "../../Compilations/Kinetics/h_Chan_Custom1",
                # "KineticVars": {},
            },
        },
        "Ca_Conc": {
            "Ca_B": 75427936887.46373,
            "Ca_tau": 0.038,
            "Ca_base": 8e-05,
            "Kinetics": "../../Compilations/Kinetics/Ca_Conc_(Common)",
        },
    }
}
t_,v_,ca_ = mm.runModel(dummyModel, Truntime=0.001)

score_exp_0 = (meanfeaturesWT - meanfeaturesWT)/stdfeaturesWT
print(list(score_exp_0.values))

def getFforfit_helper(
    dummyModel, Na_Chan_Gbar, K_DR_Chan_Gbar, K_A_Chan_Gbar, K_M_Chan_Gbar, h_Chan_Gbar
):
    Model = deepcopy(dummyModel)
    Model["Parameters"]["Channels"]["Na_Chan"]["Gbar"] = 10**Na_Chan_Gbar
    Model["Parameters"]["Channels"]["K_DR_Chan"]["Gbar"] = 10**K_DR_Chan_Gbar
    Model["Parameters"]["Channels"]["K_A_Chan"]["Gbar"] = 10**K_A_Chan_Gbar
    Model["Parameters"]["Channels"]["K_M_Chan"]["Gbar"] = 10**K_M_Chan_Gbar
    Model["Parameters"]["Channels"]["h_Chan"]["Gbar"] = 10**h_Chan_Gbar
    return Model

def getFforfit(ttt, Na_Chan_Gbar, K_DR_Chan_Gbar, K_A_Chan_Gbar, K_M_Chan_Gbar, h_Chan_Gbar):
	Model = getFforfit_helper(dummyModel, Na_Chan_Gbar, K_DR_Chan_Gbar, K_A_Chan_Gbar, K_M_Chan_Gbar, h_Chan_Gbar)
	score_model = fts.modelscore(Model, meanfeaturesWT, stdfeaturesWT, minfeaturesWT, maxfeaturesWT,modelfeature=None,apa=False,refreshKin=False)
	score_model["E_rest_0"] = 5*score_model["E_rest_0"] #Giving weights
	score_model["Input resistance"] = 5*score_model["Input resistance"]
	score_model["Cell capacitance"] = 5*score_model["Cell capacitance"]

	score_model["AP1_time_1.5e-10"] = 5*score_model["AP1_time_1.5e-10"]
	score_model["APp_time_1.5e-10"] = 5*score_model["APp_time_1.5e-10"]
	score_model["freq_1.5e-10"] = 5*score_model["freq_1.5e-10"]
	score_model["Adptn_id_1.5e-10"] = 5*score_model["Adptn_id_1.5e-10"]
	score_model["offset_1.5e-10"] = 10*score_model["offset_1.5e-10"]
	score_model["Absoffset_1.5e-10"] = 10*score_model["Absoffset_1.5e-10"]

	score_model["AP1_time_3e-10"] = 5*score_model["AP1_time_3e-10"]
	score_model["APp_time_3e-10"] = 5*score_model["APp_time_3e-10"]
	score_model["freq_3e-10"] = 5*score_model["freq_3e-10"]
	score_model["Adptn_id_3e-10"] = 5*score_model["Adptn_id_3e-10"]
	score_model["offset_3e-10"] = 10*score_model["offset_3e-10"]
	score_model["Absoffset_3e-10"] = 10*score_model["Absoffset_3e-10"]
	return list(score_model.values())

restrict = [[-5, -9, -9, -9, -9], [-3, -6, -6, -6, -7]]
paramfitted, error = bcf.brute_scifit(
    getFforfit,
    [1,2,3],
    score_exp_0,
    restrict=restrict,
    ntol=10000,
    returnnfactor=0.001,
    maxfev=10000,
    printerrors=True,
    parallel=True,
)
print(paramfitted, error)

with open('Iclampfittedmodel_4.pkl', 'wb') as f:
    pickle.dump(getFforfit_helper(dummyModel, *paramfitted), f)

#########################################################
import pickle
import MOOSEModel_18 as mm
import matplotlib.pyplot as plt
with open('Iclampfittedmodel_4.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

t,v,ca = mm.runModel(loaded_dict, 150e-12)
plt.plot(t,v, label='150pA')
t,v,ca = mm.runModel(loaded_dict, 300e-12)
plt.plot(t,v, label='300pA')
plt.legend()
plt.show()
