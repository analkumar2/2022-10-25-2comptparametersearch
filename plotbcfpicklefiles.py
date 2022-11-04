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

with open('sf_intermediatemodels_fts_2.pkl', 'rb') as f:
    a = pickle.load(f)

for i in a[0]:
    Model = getFforfit_helper(dummyModel, *i)
    pprint(Model)
    t,v,ca = mm.runModel(Model, 150e-12)
    plt.plot(t,v, label='150pA')
    t,v,ca = mm.runModel(Model, 300e-12)
    plt.plot(t,v, label='300pA')
    plt.legend()
    plt.show()
