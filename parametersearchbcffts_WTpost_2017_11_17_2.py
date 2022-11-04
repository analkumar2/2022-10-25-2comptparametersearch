from neo.io import AxonIO
import time
import pandas as pd
import plotexpv3 as pex
import MOOSEModel_2compt_19 as mm

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


Address = "../../Raw_data/Deepanjali_data/Organized_Anal/WT_prepostapa/IF_preapa/2017_11_17_2.abf"
Fexp = fts.expfeatures(Address, stim_start=0.0813, stim_end=0.0813+0.5, offset=0)
E_rest_exp = Fexp["E_rest_0"]
pprint(Fexp)

# Importing the means and std
features_ess_WT = pd.read_csv("features_ess_WT_postapa.csv", sep="\t", index_col=0)
meanfeaturesWT = features_ess_WT.mean()
stdfeaturesWT = features_ess_WT.std()
minfeaturesWT = features_ess_WT.min()
maxfeaturesWT = features_ess_WT.max()


# dummy Model
gbarscaling = 0.7
dummyModel = {
    "Parameters": {
        "notes": "",
        "Morphology": {
            "sm_len": 10.075e-6,
            "sm_diam": 10.075e-6,
            "dend_len": 500e-6,
            "dend_diam": 4e-6,
        },
        "Passive": {
            # "Rm_soma": 214407991, #0.11696,
            # "Cm_soma": 1.1988602747765764e-11,# 1e-2,
            "RM_soma": 0.06837242,
            "CM_soma": 0.03759487,
            "RA_soma": 1.49295246,
            "RM_dend": 14.9513608,
            "CM_dend": 0.02021069,
            "RA_dend": 1.2208058,
            "Em": -82e-3,
        },
        "Channels": {
            "Na_T_Chan": {
                "gbar": 1.7057586476320825e4 * gbarscaling,
                "Erev": 53e-3,
                "Kinetics": "../../Compilations/Kinetics/Na_T_Chan_Hay2011_exact",
            },
            "K_31_Chan": {
                "gbar": 0.23628414673253201e4 * gbarscaling,
                "Erev": -107e-3,
                "Kinetics": "../../Compilations/Kinetics/K_31_Chan_Hay2011_exact",
            },
            "K_P_Chan": {
                "gbar": 0.044066924433216317e4 * gbarscaling,
                "Erev": -107e-3,
                "Kinetics": "../../Compilations/Kinetics/K_P_Chan_Hay2011_exact",
            },
            "K_T_Chan": {
                "gbar": 0.0036854358538674914e4 * gbarscaling,
                "Erev": -107e-3,
                "Kinetics": "../../Compilations/Kinetics/K_T_Chan_Hay2011_exact",
            },
            "K_M_Chan": {
                "gbar": 3.5831046068113176e-03 * gbarscaling * 2000,
                "Erev": -107e-3,
                "Kinetics": "../../Compilations/Kinetics/K_M_Chan_Hay2011_exact",
            },
            "h_Chan": {
                "gbar": 0.00082455944527867655e4 * gbarscaling * 10,
                "Erev": -45e-3,
                "Kinetics": "../../Compilations/Kinetics/h_Chan_Hay2011_exact",
            },
        },
        "Ca_Conc": {
            "Ca_B": 3844559904.6760306,
            "Ca_tau": 0.038,
            "Ca_base": 8e-05,
            "Kinetics": "../../Compilations/Kinetics/Ca_Conc_(Common)",
        },
    }
}
# mm.plotModel(dummyModel)
# mm.plotModel(dummyModel)
# mm.plotModel(dummyModel)
t_, v_, ca_ = mm.runModel(dummyModel, Truntime=0.001)


def getFforfit_helper(
    dummyModel,
    RM_soma,
    CM_soma,
    RA_soma,
    RM_dend,
    CM_dend,
    RA_dend,
    Em,
    Na_T_Chan_gbar,
    K_31_Chan_gbar,
    K_P_Chan_gbar,
    K_T_Chan_gbar,
    K_M_Chan_gbar,
    h_Chan_gbar,
):
    Model = deepcopy(dummyModel)
    Model["Parameters"]["Passive"]["RM_soma"] = RM_soma
    Model["Parameters"]["Passive"]["CM_soma"] = CM_soma
    Model["Parameters"]["Passive"]["RA_soma"] = RA_soma
    Model["Parameters"]["Passive"]["RM_dend"] = RM_dend
    Model["Parameters"]["Passive"]["CM_dend"] = CM_dend
    Model["Parameters"]["Passive"]["RA_dend"] = RA_dend
    Model["Parameters"]["Passive"]["Em"] = Em
    Model["Parameters"]["Channels"]["Na_T_Chan"]["gbar"] = 10**Na_T_Chan_gbar
    Model["Parameters"]["Channels"]["K_31_Chan"]["gbar"] = 10**K_31_Chan_gbar
    Model["Parameters"]["Channels"]["K_P_Chan"]["gbar"] = 10**K_P_Chan_gbar
    Model["Parameters"]["Channels"]["K_T_Chan"]["gbar"] = 10**K_T_Chan_gbar
    Model["Parameters"]["Channels"]["K_M_Chan"]["gbar"] = 10**K_M_Chan_gbar
    Model["Parameters"]["Channels"]["h_Chan"]["gbar"] = 10**h_Chan_gbar
    return Model


def getFforfit(
    ttt,
    RM_soma=dummyModel["Parameters"]["Passive"]["RM_soma"],
    CM_soma=dummyModel["Parameters"]["Passive"]["CM_soma"],
    RA_soma=dummyModel["Parameters"]["Passive"]["RA_soma"],
    RM_dend=dummyModel["Parameters"]["Passive"]["RM_dend"],
    CM_dend=dummyModel["Parameters"]["Passive"]["CM_dend"],
    RA_dend=dummyModel["Parameters"]["Passive"]["RA_dend"],
    Em=dummyModel["Parameters"]["Passive"]["Em"],
    Na_T_Chan_gbar=np.log10(dummyModel["Parameters"]["Channels"]["Na_T_Chan"]["gbar"]),
    K_31_Chan_gbar=np.log10(dummyModel["Parameters"]["Channels"]["K_31_Chan"]["gbar"]),
    K_P_Chan_gbar=np.log10(dummyModel["Parameters"]["Channels"]["K_P_Chan"]["gbar"]),
    K_T_Chan_gbar=np.log10(dummyModel["Parameters"]["Channels"]["K_T_Chan"]["gbar"]),
    K_M_Chan_gbar=np.log10(dummyModel["Parameters"]["Channels"]["K_M_Chan"]["gbar"]),
    h_Chan_gbar=np.log10(dummyModel["Parameters"]["Channels"]["h_Chan"]["gbar"]),
):
    Model = getFforfit_helper(
        dummyModel,
        RM_soma,
        CM_soma,
        RA_soma,
        RM_dend,
        CM_dend,
        RA_dend,
        Em,
        Na_T_Chan_gbar,
        K_31_Chan_gbar,
        K_P_Chan_gbar,
        K_T_Chan_gbar,
        K_M_Chan_gbar,
        h_Chan_gbar,
    )

    t0, Vtrace0, Ca = mm.runModel(Model, 0e-12, refreshKin=False)
    offset = np.median(Vtrace0) - E_rest_exp
    # print(offset)
    tm50, Vtracem50, Ca = mm.runModel(Model, -50e-12, refreshKin=False)
    tm25, Vtracem25, Ca = mm.runModel(Model, -25e-12, refreshKin=False)
    t150, Vtrace150, Ca = mm.runModel(Model, 150e-12, refreshKin=False)
    t300, Vtrace300, Ca = mm.runModel(Model, 300e-12, refreshKin=False)

    Vtrace0 = Vtrace0 - offset
    Vtracem50 = Vtracem50 - offset
    Vtracem25 = Vtracem25 - offset
    Vtrace150 = Vtrace150 - offset
    Vtrace300 = Vtrace300 - offset

    Fmodel = {}
    Fmodel = fts.ftscalc_helper(
        Fmodel,
        t0,
        Vtrace0,
        tm50,
        Vtracem50,
        tm25,
        Vtracem25,
        t150,
        Vtrace150,
        t300,
        Vtrace300,
        stim_start=0.5,
        stim_end=1,
    )
    Model["Features"] = Fmodel

    if None in list(Fmodel.values()):
        return np.ones(20)*99999

    Fdiff = {}
    Fdiff["E_rest"] = (Fmodel["E_rest_0"] - Fexp["E_rest_0"])/stdfeaturesWT["E_rest_0"]
    Fdiff["Input resistance"] = (Fmodel["Input resistance"] - Fexp["Input resistance"])/stdfeaturesWT["Input resistance"]
    Fdiff["Cell capacitance"] = (Fmodel["Cell capacitance"] - Fexp["Cell capacitance"])/stdfeaturesWT["Cell capacitance"]
    Fdiff["sagSS_m50"] = (Fmodel["sagSS_m50"] - Fexp["sagSS_m50"])/stdfeaturesWT["sagSS_m50"]
    Fdiff["sagV_m50"] = (Fmodel["sagV_m50"] - Fexp["sagV_m50"])/stdfeaturesWT["sagV_m50"]
    Fdiff["freq_1.5e-10"] = (Fmodel["freq_1.5e-10"] - Fexp["freq_1.5e-10"])/stdfeaturesWT["freq_1.5e-10"]
    Fdiff["freq_3e-10"] = (Fmodel["freq_3e-10"] - Fexp["freq_3e-10"])/stdfeaturesWT["freq_3e-10"]
    Fdiff["offset_1.5e-10"] = (Fmodel["offset_1.5e-10"] - Fexp["offset_1.5e-10"])/stdfeaturesWT["offset_1.5e-10"]
    Fdiff["offset_3e-10"] = (Fmodel["offset_3e-10"] - Fexp["offset_3e-10"])/stdfeaturesWT["offset_3e-10"]
    Fdiff["freq300to150ratio"] = (Fmodel["freq300to150ratio"] - Fexp["freq300to150ratio"])/stdfeaturesWT["freq300to150ratio"]
    Fdiff["Adptn_id_1.5e-10"] = (Fmodel["Adptn_id_1.5e-10"] - Fexp["Adptn_id_1.5e-10"])/stdfeaturesWT["Adptn_id_1.5e-10"]
    Fdiff["Adptn_id_3e-10"] = (Fmodel["Adptn_id_3e-10"] - Fexp["Adptn_id_3e-10"])/stdfeaturesWT["Adptn_id_3e-10"]
    
    Fdiff["AP1_amp_1.5e-10"] = (Fmodel["AP1_amp_1.5e-10"] - Fexp["AP1_amp_1.5e-10"])/stdfeaturesWT["AP1_amp_1.5e-10"]
    Fdiff["APp_amp_1.5e-10"] = (Fmodel["APp_amp_1.5e-10"] - Fexp["APp_amp_1.5e-10"])/stdfeaturesWT["APp_amp_1.5e-10"]
    Fdiff["AP1_time_1.5e-10"] = (Fmodel["AP1_time_1.5e-10"] - Fexp["AP1_time_1.5e-10"])/stdfeaturesWT["AP1_time_1.5e-10"]
    Fdiff["APp_time_1.5e-10"] = (Fmodel["APp_time_1.5e-10"] - Fexp["APp_time_1.5e-10"])/stdfeaturesWT["APp_time_1.5e-10"]

    Fdiff["AP1_amp_3e-10"] = (Fmodel["AP1_amp_3e-10"] - Fexp["AP1_amp_3e-10"])/stdfeaturesWT["AP1_amp_3e-10"]
    Fdiff["APp_amp_3e-10"] = (Fmodel["APp_amp_3e-10"] - Fexp["APp_amp_3e-10"])/stdfeaturesWT["APp_amp_3e-10"]
    Fdiff["AP1_time_3e-10"] = (Fmodel["AP1_time_3e-10"] - Fexp["AP1_time_3e-10"])/stdfeaturesWT["AP1_time_3e-10"]
    Fdiff["APp_time_3e-10"] = (Fmodel["APp_time_3e-10"] - Fexp["APp_time_3e-10"])/stdfeaturesWT["APp_time_3e-10"]

    # Fdiff["Cell capacitance"] = (Fmodel["Cell capacitance"] - Fexp["Cell capacitance"])/stdfeaturesWT["Cell capacitance"]
    # Fdiff["Cell capacitance"] = (Fmodel["Cell capacitance"] - Fexp["Cell capacitance"])/stdfeaturesWT["Cell capacitance"]
    # Fdiff["Cell capacitance"] = (Fmodel["Cell capacitance"] - Fexp["Cell capacitance"])/stdfeaturesWT["Cell capacitance"]
    # Fdiff["Cell capacitance"] = (Fmodel["Cell capacitance"] - Fexp["Cell capacitance"])/stdfeaturesWT["Cell capacitance"]


    # pprint(Model)
    # plt.plot(t0, Vtrace0)
    # plt.plot(tm50, Vtracem50)
    # plt.plot(tm25, Vtracem25)
    # plt.plot(t150, Vtrace150)
    # plt.plot(t300, Vtrace300)
    # plt.show()

    return list(Fdiff.values())


Fdiff = getFforfit([1, 2, 3])
# pprint(Fdiff)

# plt.plot(t, Vexp)
# plt.plot(t, Vmodel)
# plt.show()


#### Actual parametersearch ##########################
restrict = [
    [
        0.06837242 * 0.5,
        0.03759487 * 0.5,
        1.49295246 * 0.5,
        14.9513608 * 0.5,
        0.02021069 * 0.5,
        1.2208058 * 0.5,
        -82e-3 - 18e-3,
        np.log10(1.7057586476320825e4 * 0.7) - 1,
        np.log10(0.23628414673253201e4 * 0.7) - 1,
        np.log10(0.044066924433216317e4 * 0.7) - 1,
        np.log10(0.0036854358538674914e4 * 0.7) - 1,
        np.log10(3.5831046068113176e-03 * 0.7) - 1,
        np.log10(0.00082455944527867655e4 * 0.7) - 1,
    ],
    [
        0.06837242 * 2,
        0.03759487 * 2,
        1.49295246 * 2,
        14.9513608 * 2,
        0.02021069 * 2,
        1.2208058 * 2,
        -82e-3 + 12e-3,
        np.log10(1.7057586476320825e4 * 0.7) + 1,
        np.log10(0.23628414673253201e4 * 0.7) + 1,
        np.log10(0.044066924433216317e4 * 0.7) + 1,
        np.log10(0.0036854358538674914e4 * 0.7) + 1,
        np.log10(3.5831046068113176e-03 * 0.7) + 1,
        np.log10(0.00082455944527867655e4 * 0.7) + 1,
    ],
]
paramfitted, error = bcf.brute_scifit(
    getFforfit,
    [1, 2, 3],
    np.zeros(len(Fdiff)),
    restrict=restrict,
    ntol=10000,
    returnnfactor=0.001,
    maxfev=1000,
    printerrors=True,
    parallel=True,
    savetofile='intermediatemodels_fts_2.pkl',
)
print(paramfitted, error)

# Vmodel = getFforfit([1,2,3], *paramfitted)
# plt.plot(t, Vexp, label='exp')
# plt.plot(t, Vmodel, label='fitted')
# plt.legend()
# plt.show()

with open("Iclampfittedmodel_fts_2.pkl", "wb") as f:
    pickle.dump(getFforfit_helper(dummyModel, *paramfitted), f)

# # #########################################################
# import pickle
# import MOOSEModel_2compt_19 as mm
# import matplotlib.pyplot as plt
# with open('Iclampfittedmodel_fts_2.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

# t,v,ca = mm.runModel(loaded_dict, 150e-12)
# plt.plot(t,v, label='150pA')
# t,v,ca = mm.runModel(loaded_dict, 300e-12)
# plt.plot(t,v, label='300pA')
# plt.legend()
# plt.show()
