## exec(open('featuresv13.py').read())
## version 8 changes: cell capacitance calculated from -25pA current injection instead of 25pA injection
## version 9 changes: Now calculates -50pA current injection steady state voltage too (for sag)
## version 10 changes: Now calculates sag voltage too (m50 SS sag voltage - m50 voltage without sag)
## version 11: Now calculates upstrokes and downstrokes too. That is, maximum dv/dt time and voltage
## version 12: NOw calculates SK current too.
## version 13: SK current or apaCurr as it is called now is being calculated for -35mV and 35mV hold
## version 14: Adds an offset feature
## version 15: modelscore has an option of using minmax also. score is now without abs
## version 16: adds jitter and CVs500 function for abf files
## version 17: resolves conflicts in v14, v15 and v16
## version 18: jitter calulation for models
## version 19: jitter calculation is now parallelized
## version 20: jitter calculation has a sy option
## version 21_custom: frequency ratio 300pA/150pA added as one of the feature. And only features that can be achieved by Na, KDR, KA and KM. And freq at-25pA injection should be 0
## version 23: The jitter calculation now only uses the maximum number of spikes that occur in all the trials
## version 24: Proper CV calculation
## version 24_custom2: for CV_EPSPfreq_amp
## version 25: Compiles the various version of v24 in existence. Several other changes.
## version 25_nonallen: Does not use allensdk for most features
## version 26_nonallen: Adds the timing of the last spike as one of the feature to detect depolarization blocks. Also added Absolute offset
## version 27: Non allen and uses plotexp3
## version 28: Now APp Up and downstrokes being calculated properly. And added refreshkin
## version 29: Separate function for passive features
## version 30: Added an optional offset arguement in expfeature to deal with things like LJP. Also made the functions independent of elecPlotDt
## vesrion 31: If there is a problem with the model (not firing etc), the full feature list is returned with None there instead of just returning false. Removed modelcutoff and bettermodel functions
## version 31_2compt: For the 2compt model

import numpy as np
import numpy.random as nr
import quantities as pq
import matplotlib.pyplot as plt
from neo.io import AxonIO
import plotexpv3 as pex
import MOOSEModel_2compt_19 as mm
import moose
import os
import csv
import scipy.signal as scs
import scipy.interpolate as sciip
import scipy.optimize as scioz
import scipy.stats as scst
import pandas as pd
import brute_curvefit
from pprint import pprint
from copy import deepcopy
import argparse
import pickle
from scipy.signal import butter, filtfilt
import multiprocessing
from multiprocessing import Pool
import time
import argparse
from pprint import pprint
import pickle
from copy import deepcopy
import warnings


warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.filterwarnings("error", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor
from sklearn.linear_model import LinearRegression


def calcRinCin(modeldict, refreshKin=True, stim_start=1):
    ## Input resistance and Cell capacitance
    tm25, Vtracem25, Ca = mm.runModel(modeldict, -25e-12, refreshKin=refreshKin)
    E_rest = np.median(Vtracem25[tm25 < stim_start])

    def chargingm25(t, R1, R2, tau1, tau2):
        return (
            E_rest
            - R1 * 25e-12 * (1 - np.exp(-t / tau1))
            - R2 * 25e-12 * (1 - np.exp(-t / tau2))
        )

    tempv = Vtracem25[(tm25 >= stim_start) & (tm25 <= stim_start + 0.1)]
    RCfitted_chm25, errorm25 = brute_curvefit.brute_scifit(
        chargingm25,
        np.linspace(0, 0.1, len(tempv)),
        tempv,
        restrict=[[5e6, 5e6, 0, 0], [1000e6, 1000e6, 0.1, 0.1]],
        ntol=1000,
        printerrors=False,
    )
    Rin = RCfitted_chm25[0] + RCfitted_chm25[1]
    if RCfitted_chm25[2] > RCfitted_chm25[3]:
        Cin = RCfitted_chm25[2] / RCfitted_chm25[0]
    else:
        Cin = RCfitted_chm25[3] / RCfitted_chm25[1]

    # plt.plot(np.linspace(0, 0.1, len(tempv)), tempv, label='ori')
    # plt.plot(np.linspace(0, 0.1, len(tempv)), chargingm25(np.linspace(0, 0.1, len(tempv)), *RCfitted_chm25),  label='fitted')
    # plt.legend()
    # plt.show()

    return [Rin, Cin]


def ftscalc_helper(
    features,
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
    stim_start,
    stim_end,
):
    featurekeys_list = [
        "Sampling rate",
        "stim_start",
        "stim_end",
        "E_rest_0",
        "E_rest_m50",
        "E_rest_m25",
        "E_rest_150",
        "E_rest_300",
        "Input resistance",
        "Cell capacitance",
        "sagSS_m50",
        "sagV_m50",
        "AP1_amp_1.5e-10",
        "APp_amp_1.5e-10",
        "AP1_time_1.5e-10",
        "APp_time_1.5e-10",
        "APavgpratio_amp_1.5e-10",
        "AP1_width_1.5e-10",
        "APp_width_1.5e-10",
        "AP1_thresh_1.5e-10",
        "APp_thresh_1.5e-10",
        "AP1_lat_1.5e-10",
        "ISI1_1.5e-10",
        "ISIl_1.5e-10",
        "ISIavg_1.5e-10",
        "freq_1.5e-10",
        "Adptn_id_1.5e-10",
        "mAHP_stimend_amp_1.5e-10",
        "sAHP_stimend_amp_1.5e-10",
        "AHP_AP1_amp_1.5e-10",
        "AHP_APp_amp_1.5e-10",
        "AHP_AP1_time_1.5e-10",
        "AHP_APp_time_1.5e-10",
        "Upstroke_AP1_time_1.5e-10",
        "Upstroke_APp_time_1.5e-10",
        "Upstroke_AP1_amp_1.5e-10",
        "Upstroke_APp_amp_1.5e-10",
        "Upstroke_AP1_value_1.5e-10",
        "Upstroke_APp_value_1.5e-10",
        "Downstroke_AP1_time_1.5e-10",
        "Downstroke_APp_time_1.5e-10",
        "Downstroke_AP1_amp_1.5e-10",
        "Downstroke_APp_amp_1.5e-10",
        "Downstroke_AP1_value_1.5e-10",
        "Downstroke_APp_value_1.5e-10",
        "UpDn_AP1_ratio_1.5e-10",
        "UpDn_APp_ratio_1.5e-10",
        "UpThr_AP1_diff_1.5e-10",
        "UpThr_APp_diff_1.5e-10",
        "offset_1.5e-10",
        "Absoffset_1.5e-10",
        "AP1_amp_3e-10",
        "APp_amp_3e-10",
        "AP1_time_3e-10",
        "APp_time_3e-10",
        "APavgpratio_amp_3e-10",
        "AP1_width_3e-10",
        "APp_width_3e-10",
        "AP1_thresh_3e-10",
        "APp_thresh_3e-10",
        "AP1_lat_3e-10",
        "ISI1_3e-10",
        "ISIl_3e-10",
        "ISIavg_3e-10",
        "freq_3e-10",
        "Adptn_id_3e-10",
        "mAHP_stimend_amp_3e-10",
        "sAHP_stimend_amp_3e-10",
        "AHP_AP1_amp_3e-10",
        "AHP_APp_amp_3e-10",
        "AHP_AP1_time_3e-10",
        "AHP_APp_time_3e-10",
        "Upstroke_AP1_time_3e-10",
        "Upstroke_APp_time_3e-10",
        "Upstroke_AP1_amp_3e-10",
        "Upstroke_APp_amp_3e-10",
        "Upstroke_AP1_value_3e-10",
        "Upstroke_APp_value_3e-10",
        "Downstroke_AP1_time_3e-10",
        "Downstroke_APp_time_3e-10",
        "Downstroke_AP1_amp_3e-10",
        "Downstroke_APp_amp_3e-10",
        "Downstroke_AP1_value_3e-10",
        "Downstroke_APp_value_3e-10",
        "UpDn_AP1_ratio_3e-10",
        "UpDn_APp_ratio_3e-10",
        "UpThr_AP1_diff_3e-10",
        "UpThr_APp_diff_3e-10",
        "offset_3e-10",
        "Absoffset_3e-10",
        "freq300to150ratio",
    ]
    for fkey in featurekeys_list:
        if fkey not in features.keys():
            features[fkey] = None
    features["Sampling rate"] = len(tm25) / tm25[-1]
    Samprate = features["Sampling rate"]
    features["stim_start"] = stim_start
    features["stim_end"] = stim_end

    tt = tm25
    vv = Vtracem25
    ii = np.zeros(len(tt))
    ii[(tt >= stim_start) & (tt <= stim_end)] = -25e-12
    sweep_ext = EphysSweepFeatureExtractor(
        t=tt, v=Vtracem25 * 1e3, i=ii * 1e12, filter=len(tt) / tt[-1] / 2500
    )
    sweep_ext.process_spikes()
    if len(sweep_ext.spike_feature("width")) >= 4:
        return features

    ## Erest
    features[f"E_rest_0"] = np.median(Vtrace0)
    if stim_start < 0.2:
        features[f"E_rest_m50"] = np.median(Vtracem50[tm50 < stim_start])
        features[f"E_rest_m25"] = np.median(Vtracem25[tm25 < stim_start])
        features[f"E_rest_150"] = np.median(Vtrace150[t150 < stim_start])
        features[f"E_rest_300"] = np.median(Vtrace300[t300 < stim_start])
    else:
        features[f"E_rest_m50"] = np.median(
            Vtracem50[(tm50 < stim_start) & (tm50 > stim_start - 0.1)]
        )
        features[f"E_rest_m25"] = np.median(
            Vtracem25[(tm25 < stim_start) & (tm25 > stim_start - 0.1)]
        )
        features[f"E_rest_150"] = np.median(
            Vtrace150[(t150 < stim_start) & (t150 > stim_start - 0.1)]
        )
        features[f"E_rest_300"] = np.median(
            Vtrace300[(t300 < stim_start) & (t300 > stim_start - 0.1)]
        )

    ## Input resistance and Cell capacitance
    def chargingm25(t, R1, R2, tau1, tau2):
        return (
            features[f"E_rest_m25"]
            - R1 * 25e-12 * (1 - np.exp(-t / tau1))
            - R2 * 25e-12 * (1 - np.exp(-t / tau2))
        )

    tempv = Vtracem25[(tm25 > stim_start) & (tm25 < stim_start + 0.1)]
    RCfitted_chm25, errorm25 = brute_curvefit.brute_scifit(
        chargingm25,
        np.linspace(0, 0.1, len(tempv)),
        tempv,
        restrict=[[5e6, 5e6, 0, 0], [1000e6, 1000e6, 0.1, 0.1]],
        ntol=1000,
        printerrors=False,
    )
    features[f"Input resistance"] = RCfitted_chm25[0] + RCfitted_chm25[1]
    if RCfitted_chm25[2] > RCfitted_chm25[3]:
        features[f"Cell capacitance"] = RCfitted_chm25[2] / RCfitted_chm25[0]
    else:
        features[f"Cell capacitance"] = RCfitted_chm25[3] / RCfitted_chm25[1]

    features[f"sagSS_m50"] = np.median(
        Vtracem50[(tm50 < stim_end) & (tm50 > stim_end - 0.1)]
    )
    features[f"sagV_m50"] = features[f"sagSS_m50"] - (
        features[f"E_rest_m50"] - 50e-12 * features[f"Input resistance"]
    )

    for I in [150e-12, 300e-12]:
        if I == 150e-12:
            tt = t150
            vv = Vtrace150
            Erest = features[f"E_rest_150"]
        elif I == 300e-12:
            tt = t300
            vv = Vtrace300
            Erest = features[f"E_rest_300"]
        ii = np.zeros(len(tt))
        ii[(tt >= stim_start) & (tt <= stim_end)] = I
        try:
            sweep_ext = EphysSweepFeatureExtractor(
                t=tt,
                v=vv * 1e3,
                i=ii * 1e12,
                filter=len(tt) / tt[-1] / 2500,
                start=stim_start,
                end=stim_end,
            )
            sweep_ext.process_spikes()
            if len(sweep_ext.spike_feature("width")) <= 3:
                print("len width <4")
                return features

            # AP1_amp
            features[f"AP1_amp_{I}"] = (
                sweep_ext.spike_feature("peak_v")[0] * 1e-3 - Erest
            )

            # APp_amp
            features[f"APp_amp_{I}"] = (
                sweep_ext.spike_feature("peak_v")[-2] * 1e-3 - Erest
            )

            # AP1_time
            features[f"AP1_time_{I}"] = (
                sweep_ext.spike_feature("peak_t")[0] - stim_start
            )

            # APp_time
            features[f"APp_time_{I}"] = (
                sweep_ext.spike_feature("peak_t")[-2] - stim_start
            )

            # APavgpratio_amp
            APavg_amp = np.nanmean(sweep_ext.spike_feature("peak_v")) * 1e-3 - Erest
            features[f"APavgpratio_amp_{I}"] = APavg_amp / features[f"APp_amp_{I}"]

            # AP1_width
            p1 = sweep_ext.spike_feature("peak_index")[0]
            f2 = sciip.interp1d(
                vv[p1 - int(Samprate * 1e-3) : p1 + 1],
                tt[p1 - int(Samprate * 1e-3) : p1 + 1],
            )
            f3 = sciip.interp1d(
                vv[p1 : p1 + int(Samprate * 3e-3) + 1],
                tt[p1 : p1 + int(Samprate * 3e-3) + 1],
            )
            features[f"AP1_width_{I}"] = f3(0) - f2(0)
            # features[f"AP1_width_{I}"] = sweep_ext.spike_feature("width")[0]

            # APp_width
            p1p = sweep_ext.spike_feature("peak_index")[-2]
            f2 = sciip.interp1d(
                vv[p1p - int(Samprate * 1e-3) : p1p + 1],
                tt[p1p - int(Samprate * 1e-3) : p1p + 1],
            )
            f3 = sciip.interp1d(
                vv[p1p : p1p + int(Samprate * 3e-3) + 1],
                tt[p1p : p1p + int(Samprate * 3e-3) + 1],
            )
            features[f"APp_width_{I}"] = f3(0) - f2(0)
            # features[f"APp_width_{I}"] = sweep_ext.spike_feature("width")[-2]

            # AP1_thresh
            Vgrad = np.gradient(vv, tt)
            p1 = sweep_ext.spike_feature("peak_index")[0]
            f2 = sciip.interp1d(
                Vgrad[p1 - int(Samprate * 1e-3) : p1 + 1],
                tt[p1 - int(Samprate * 1e-3) : p1 + 1],
            )
            f3 = sciip.interp1d(
                tt[p1 - int(Samprate * 1e-3) : p1 + 1],
                vv[p1 - int(Samprate * 1e-3) : p1 + 1],
            )
            Thresh_t = f2(25)
            Thresh_v = f3(Thresh_t)
            features[f"AP1_thresh_{I}"] = float(Thresh_v)
            # features[f"AP1_thresh_{I}"] = (
            #     sweep_ext.spike_feature("threshold_v")[0] * 1e-3
            # )

            # APp_thresh
            Vgradp = np.gradient(vv, tt)
            p1p = sweep_ext.spike_feature("peak_index")[-2]
            f2p = sciip.interp1d(
                Vgradp[p1p - int(Samprate * 1e-3) : p1p + 1],
                tt[p1p - int(Samprate * 1e-3) : p1p + 1],
            )
            f3p = sciip.interp1d(
                tt[p1p - int(Samprate * 1e-3) : p1p + 1],
                vv[p1p - int(Samprate * 1e-3) : p1p + 1],
            )
            Thresh_tp = f2p(25)
            Thresh_vp = f3p(Thresh_tp)
            features[f"APp_thresh_{I}"] = float(Thresh_vp)
            # features[f"APp_thresh_{I}"] = (
            #     sweep_ext.spike_feature("threshold_v")[-2] * 1e-3
            # )

            # AP1_lat
            features[f"AP1_lat_{I}"] = float(Thresh_t) - stim_start
            # features[f"AP1_lat_{I}"] = (
            #     sweep_ext.spike_feature("threshold_t")[0] - stim_start
            # )

            # ISI1
            features[f"ISI1_{I}"] = (
                sweep_ext.spike_feature("peak_t")[1]
                - sweep_ext.spike_feature("peak_t")[0]
            )

            # ISIl
            features[f"ISIl_{I}"] = (
                sweep_ext.spike_feature("peak_t")[-1]
                - sweep_ext.spike_feature("peak_t")[-2]
            )

            # ISIavg
            pt = sweep_ext.spike_feature("peak_t")
            features[f"ISIavg_{I}"] = np.nanmean(
                [s - f for s, f in zip(pt[1:], pt[:-1])]
            )

            # freq
            features[f"freq_{I}"] = len(sweep_ext.spike_feature("peak_t")) / (
                stim_end - stim_start
            )

            # Adptn_id = 1-ISI1/ISIl
            features[f"Adptn_id_{I}"] = (
                1 - features[f"ISI1_{I}"] / features[f"ISIl_{I}"]
            )

            # # fAHP_AP1_amp
            # features[f"fAHP_AP1_amp_{I}"] = (
            #     sweep_ext.spike_feature("fast_trough_v")[0] * 1e-3 - Erest
            # )

            # # fAHP_APp_amp
            # features[f"fAHP_APp_amp_{I}"] = (
            #     sweep_ext.spike_feature("fast_trough_v")[-2] * 1e-3 - Erest
            # )

            # # mAHP_AP1_amp
            # features[f"mAHP_AP1_amp_{I}"] = (
            #     sweep_ext.spike_feature("slow_trough_v")[0] * 1e-3 - Erest
            # )

            # # mAHP_APp_amp
            # features[f"mAHP_APp_amp_{I}"] = (
            #     sweep_ext.spike_feature("slow_trough_v")[-2] * 1e-3 - Erest
            # )

            # # mAHP_AP1_time
            # features[f"mAHP_AP1_time_{I}"] = (
            #     sweep_ext.spike_feature("slow_trough_t")[0]
            #     - sweep_ext.spike_feature("peak_t")[0]
            # )

            # # mAHP_APp_time = mAHP of second last spike (penultimate)
            # features[f"mAHP_APp_time_{I}"] = (
            #     sweep_ext.spike_feature("slow_trough_t")[-2]
            #     - sweep_ext.spike_feature("peak_t")[-2]
            # )

            # # ADP_AP1_amp
            # features[f"ADP_AP1_amp_{I}"] = (
            #     sweep_ext.spike_feature("adp_v")[0] * 1e-3 - Erest
            # )

            # # ADP_APp_amp
            # features[f"ADP_APp_amp_{I}"] = (
            #     sweep_ext.spike_feature("adp_v")[-2] * 1e-3 - Erest
            # )

            # mAHP_stimend_amp = within 50ms
            features[f"mAHP_stimend_amp_{I}"] = (
                np.min(vv[(tt > stim_end) & (tt < stim_end + 0.050)]) - Erest
            )

            # sAHP_stimend_amp = within 200ms
            features[f"sAHP_stimend_amp_{I}"] = (
                np.min(vv[(tt > stim_end) & (tt < stim_end + 0.200)]) - Erest
            )

            # AHP_AP1_amp = Minimum between the first two spikes
            features[f"AHP_AP1_amp_{I}"] = (
                np.nanmin(
                    vv[
                        (tt > sweep_ext.spike_feature("peak_t")[0])
                        & (tt < sweep_ext.spike_feature("peak_t")[1])
                    ]
                )
                - Erest
            )

            # AHP_APp_amp = Minimum between the last two spikes
            features[f"AHP_APp_amp_{I}"] = (
                np.nanmin(
                    vv[
                        (tt > sweep_ext.spike_feature("peak_t")[-2])
                        & (tt < sweep_ext.spike_feature("peak_t")[-1])
                    ]
                )
                - Erest
            )

            # AHP_AP1_time = Minimum between the first two spikes
            features[f"AHP_AP1_time_{I}"] = (
                np.argmin(
                    vv[
                        (tt >= sweep_ext.spike_feature("peak_t")[0])
                        & (tt < sweep_ext.spike_feature("peak_t")[1])
                    ]
                )
                / features["Sampling rate"]
            )

            # AHP_APp_time = Minimum between the last two spikes
            features[f"AHP_APp_time_{I}"] = (
                np.argmin(
                    vv[
                        (tt >= sweep_ext.spike_feature("peak_t")[-2])
                        & (tt < sweep_ext.spike_feature("peak_t")[-1])
                    ]
                )
                / features["Sampling rate"]
            )

            # Upstroke_AP1_time = Upstroke time of first spike
            Vgrad = np.gradient(vv, tt)
            p1 = sweep_ext.spike_feature("peak_index")[0]
            Vgradmaxidx = np.argmax(Vgrad[p1 - int(Samprate * 1e-3) : p1 + 1])
            Up_t = tt[Vgradmaxidx + p1 - int(Samprate * 1e-3)]
            Up_v = vv[Vgradmaxidx + p1 - int(Samprate * 1e-3)]
            features[f"Upstroke_AP1_time_{I}"] = (
                float(Up_t) - sweep_ext.spike_feature("peak_t")[0]
            )
            # features[f"Upstroke_AP1_time_{I}"] = (
            #     sweep_ext.spike_feature("upstroke_t")[0]
            #     - sweep_ext.spike_feature("peak_t")[0]
            # )

            # Upstroke_APp_time = Upstroke time of penultimate spike
            Vgradp = np.gradient(vv, tt)
            p1p = sweep_ext.spike_feature("peak_index")[-2]
            Vgradmaxidxp = np.argmax(Vgradp[p1p - int(Samprate * 1e-3) : p1p + 1])
            Up_tp = tt[Vgradmaxidxp + p1p - int(Samprate * 1e-3)]
            Up_vp = vv[Vgradmaxidxp + p1p - int(Samprate * 1e-3)]
            features[f"Upstroke_APp_time_{I}"] = (
                float(Up_tp) - sweep_ext.spike_feature("peak_t")[-2]
            )
            # features[f"Upstroke_APp_time_{I}"] = (
            #     sweep_ext.spike_feature("upstroke_t")[-2]
            #     - sweep_ext.spike_feature("peak_t")[-2]
            # )

            # Upstroke_AP1_amp = Upstroke amp of first spike
            features[f"Upstroke_AP1_amp_{I}"] = float(Up_v) * 1e-3 - Erest
            # features[f"Upstroke_AP1_amp_{I}"] = (
            #     sweep_ext.spike_feature("upstroke_v")[0] * 1e-3 - Erest
            # )

            # Upstroke_APp_amp = Upstroke amp of penultimate spike
            features[f"Upstroke_APp_amp_{I}"] = float(Up_vp) * 1e-3 - Erest
            # features[f"Upstroke_APp_amp_{I}"] = (
            #     sweep_ext.spike_feature("upstroke_v")[-2] * 1e-3 - Erest
            # )

            # Upstroke_AP1_value = Upstroke value of first spike
            features[f"Upstroke_AP1_value_{I}"] = Vgrad[
                Vgradmaxidx + p1 - int(Samprate * 1e-3)
            ]
            # features[f"Upstroke_AP1_value_{I}"] = sweep_ext.spike_feature("upstroke")[0]

            # Upstroke_APp_value = Upstroke value of penultimate spike
            features[f"Upstroke_APp_value_{I}"] = Vgradp[
                Vgradmaxidxp + p1p - int(Samprate * 1e-3)
            ]
            # features[f"Upstroke_APp_value_{I}"] = sweep_ext.spike_feature("upstroke")[
            #     -2
            # ]

            # Downstroke_AP1_time = Downstroke time of first spike
            Vgrad = np.gradient(vv, tt)
            p1 = sweep_ext.spike_feature("peak_index")[0]
            Vgradminidx = np.argmin(Vgrad[p1 : p1 + int(Samprate * 3e-3) + 1])
            Dn_t = tt[Vgradminidx + p1]
            Dn_v = vv[Vgradminidx + p1]
            features[f"Downstroke_AP1_time_{I}"] = (
                float(Dn_t) - sweep_ext.spike_feature("peak_t")[0]
            )
            # features[f"Downstroke_AP1_time_{I}"] = (
            #     sweep_ext.spike_feature("downstroke_t")[0]
            #     - sweep_ext.spike_feature("peak_t")[0]
            # )

            # Downstroke_APp_time = Downstroke time of penultimate spike
            Vgradp = np.gradient(vv, tt)
            p1p = sweep_ext.spike_feature("peak_index")[-2]
            Vgradminidxp = np.argmin(Vgradp[p1p : p1p + int(Samprate * 3e-3) + 1])
            Dn_tp = tt[Vgradminidxp + p1p]
            Dn_vp = vv[Vgradminidxp + p1p]
            features[f"Downstroke_APp_time_{I}"] = (
                float(Dn_tp) - sweep_ext.spike_feature("peak_t")[-2]
            )
            # features[f"Downstroke_APp_time_{I}"] = (
            #     sweep_ext.spike_feature("downstroke_t")[-2]
            #     - sweep_ext.spike_feature("peak_t")[-2]
            # )

            # Downstroke_AP1_amp = Downstroke amp of first spike
            features[f"Downstroke_AP1_amp_{I}"] = float(Dn_v) * 1e-3 - Erest
            # features[f"Downstroke_AP1_amp_{I}"] = (
            #     sweep_ext.spike_feature("downstroke_v")[0] * 1e-3 - Erest
            # )

            # Downstroke_APp_amp = Downstroke amp of penultimate spike
            features[f"Downstroke_APp_amp_{I}"] = float(Dn_vp) * 1e-3 - Erest
            # features[f"Downstroke_APp_amp_{I}"] = (
            #     sweep_ext.spike_feature("downstroke_v")[-2] * 1e-3 - Erest
            # )

            # Downstroke_AP1_value = Downstroke value of first spike
            features[f"Downstroke_AP1_value_{I}"] = Vgrad[
                Vgradminidx + p1 - int(Samprate * 1e-3)
            ]
            # features[f"Downstroke_AP1_value_{I}"] = sweep_ext.spike_feature(
            #     "downstroke"
            # )[0]

            # Downstroke_APp_value = Downstroke value of penultimate spike
            features[f"Downstroke_APp_value_{I}"] = Vgradp[
                Vgradminidxp + p1p - int(Samprate * 1e-3)
            ]
            # features[f"Downstroke_APp_value_{I}"] = sweep_ext.spike_feature(
            #     "downstroke"
            # )[-2]

            # UpDn_AP1_ratio = Upstroke/Downstroke ratio of first spike
            features[f"UpDn_AP1_ratio_{I}"] = (
                features[f"Upstroke_AP1_value_{I}"]
                / features[f"Downstroke_AP1_value_{I}"]
            )
            # features[f"UpDn_AP1_ratio_{I}"] = sweep_ext.spike_feature(
            #     "upstroke_downstroke_ratio"
            # )[0]

            # UpDn_APp_ratio = Upstroke/Downstroke ratio of penultimate spike
            features[f"UpDn_APp_ratio_{I}"] = (
                features[f"Upstroke_APp_value_{I}"]
                / features[f"Downstroke_APp_value_{I}"]
            )
            # features[f"UpDn_APp_ratio_{I}"] = sweep_ext.spike_feature(
            #     "upstroke_downstroke_ratio"
            # )[-2]

            # UpThr_AP1_diff = Upstroke_v - threshold_v of first spike
            features[f"UpThr_AP1_diff_{I}"] = (
                features[f"Upstroke_AP1_amp_{I}"] - features[f"AP1_thresh_{I}"] + Erest
            )
            # features[f"UpThr_AP1_diff_{I}"] = (
            #     sweep_ext.spike_feature("upstroke_v")[0] * 1e-3
            #     - sweep_ext.spike_feature("threshold_v")[0] * 1e-3
            # )

            # UpThr_APp_diff = Upstroke_v - threshold_v of penultimate spike
            features[f"UpThr_APp_diff_{I}"] = (
                features[f"Upstroke_APp_amp_{I}"] - features[f"APp_thresh_{I}"] + Erest
            )
            # features[f"UpThr_APp_diff_{I}"] = (
            #     sweep_ext.spike_feature("upstroke_v")[-2] * 1e-3
            #     - sweep_ext.spike_feature("threshold_v")[-2] * 1e-3
            # )

            # Offset
            features[f"offset_{I}"] = (
                np.nanmin(vv[(tt <= stim_end - 0.100) & (tt >= stim_start + 0.100)])
                - Erest
            )

            # Absolute Offset
            features[f"Absoffset_{I}"] = np.nanmin(
                vv[(tt <= stim_end - 0.100) & (tt >= stim_start + 0.100)]
            )

        except:
            print("len width <4 2")
            return features

    try:
        # frequency 300pA to frequency 150pA ratio
        features[f"freq300to150ratio"] = (
            features[f"freq_{300e-12}"] / features[f"freq_{150e-12}"]
        )
    except:
        print("len width <4 3")
        return features

    return features


def expfeatures(cellpath, stim_start, stim_end, offset=0):
    t0, Vtrace0 = pex.expdata(cellpath, 4)
    tm50, Vtracem50 = pex.expdata(cellpath, 2)
    tm25, Vtracem25 = pex.expdata(cellpath, 3)
    t150, Vtrace150 = pex.expdata(cellpath, 10)
    t300, Vtrace300 = pex.expdata(cellpath, 16)
    Vtrace0 = Vtrace0 - offset
    Vtracem50 = Vtracem50 - offset
    Vtracem25 = Vtracem25 - offset
    Vtrace150 = Vtrace150 - offset
    Vtrace300 = Vtrace300 - offset

    features = {}
    features["Cell name"] = cellpath.split("/")[-1]
    features = ftscalc_helper(
        features,
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
        stim_start,
        stim_end,
    )

    return features


def modelfeatures(modeldict, stim_start=1, stim_end=1.5, apa=True, refreshKin=True):
    t0, Vtrace0, Ca = mm.runModel(modeldict, 0e-12, refreshKin=refreshKin)
    tm50, Vtracem50, Ca = mm.runModel(modeldict, -50e-12, refreshKin=False)
    tm25, Vtracem25, Ca = mm.runModel(modeldict, -25e-12, refreshKin=False)
    t150, Vtrace150, Ca = mm.runModel(modeldict, 150e-12, refreshKin=False)
    t300, Vtrace300, Ca = mm.runModel(modeldict, 300e-12, refreshKin=False)

    features = {}
    # features['Model name'] = cellpath.split('/')[-1]
    features = ftscalc_helper(
        features,
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
        stim_start,
        stim_end,
    )
    if isinstance(features, dict) and apa:
        tbAp, IbAp, Ca = mm.runModel(
            modeldict,
            vClamp="-0.055 + (t>1 && t<1.1)*-0.010 + (t>1.1 && t<1.9)*0.020",
            refreshKin=False,
        )
        modeldict_temp = deepcopy(modeldict)
        modeldict_temp["Parameters"]["Channels"]["K_SK_Chan"]["Gbar"] = 1e-15
        taAp, IaAp, Ca = mm.runModel(
            modeldict_temp,
            vClamp="-0.055 + (t>1 && t<1.1)*-0.010 + (t>1.1 && t<1.9)*0.020",
            refreshKin=False,
        )
        features["apaCurrm35"] = (
            IbAp[np.argmin(np.abs(1.925 - tbAp))]
            - IaAp[np.argmin(np.abs(1.925 - taAp))]
        )

        tbAp, IbAp, Ca = mm.runModel(
            modeldict,
            vClamp="-0.055 + (t>1 && t<1.1)*-0.010 + (t>1.1 && t<1.9)*0.090",
            refreshKin=False,
        )
        modeldict_temp = deepcopy(modeldict)
        modeldict_temp["Parameters"]["Channels"]["K_SK_Chan"]["Gbar"] = 1e-15
        taAp, IaAp, Ca = mm.runModel(
            modeldict_temp,
            vClamp="-0.055 + (t>1 && t<1.1)*-0.010 + (t>1.1 && t<1.9)*0.090",
            refreshKin=False,
        )
        features["apaCurr35"] = (
            IbAp[np.argmin(np.abs(1.925 - tbAp))]
            - IaAp[np.argmin(np.abs(1.925 - taAp))]
        )

    return features


def modelscore(
    modeldict,
    meanfeature,
    stdfeature,
    minfeature=None,
    maxfeature=None,
    modelfeature=None,
    apa=True,
    refreshKin=True,
):
    if modelfeature == None:
        modelfeature = modelfeatures(
            modeldict, stim_start=0.5, stim_end=1, apa=apa, refreshKin=refreshKin
        )

    score = {fkey: 1000 for fkey in list(meanfeature.keys())}
    if None in modelfeature.values():
        return score

    def removestrcol(dictt):
        dicttk = dictt.keys()
        for keyy in dicttk:
            if isinstance(dictt[keyy], str):
                dictt.pop(keyy)

    [removestrcol(dictt) for dictt in [meanfeature, stdfeature, modelfeature]]

    for keyy in meanfeature.keys():
        if minfeature is None or maxfeature is None:
            score[keyy] = (modelfeature[keyy] - meanfeature[keyy]) / stdfeature[keyy]
        elif minfeature[keyy] < modelfeature[keyy] < maxfeature[keyy]:
            score[keyy] = 0
        else:
            score[keyy] = (modelfeature[keyy] - meanfeature[keyy]) / stdfeature[keyy]

    return score


def calcCV(ISIlist):
    if len(ISIlist) == 0:
        return np.nan
    return np.nanstd(ISIlist) / np.nanmean(ISIlist)


def calcjitslope(spiket_list, stim_start=1):
    minspikes = 100
    for spiket in spiket_list:
        if len(spiket) < minspikes:
            minspikes = len(spiket)

    for i in range(len(spiket_list)):
        spiket_list[i] = spiket_list[i][:minspikes]

    if len(spiket_list[0]) < 2:
        return [np.nan, np.nan]

    jitter = np.nanstd(spiket_list, 0)
    spikemean = np.nanmean(spiket_list, 0)

    x_entire = spikemean.reshape((-1, 1))
    y_entire = jitter
    model_entire = LinearRegression().fit(x_entire, y_entire)

    if (
        len(
            np.array(spikemean)[
                (np.array(spikemean) > stim_start + 0.5)
                & (np.array(spikemean) <= stim_start + 0.9)
            ]
        )
        < 2
    ):
        return [model_entire.coef_[0], np.nan]

    x_2 = np.array(spikemean)[
        (np.array(spikemean) > stim_start + 0.5)
        & (np.array(spikemean) <= stim_start + 0.9)
    ].reshape((-1, 1))
    y_2 = np.array(jitter)[
        (np.array(spikemean) > stim_start + 0.5)
        & (np.array(spikemean) <= stim_start + 0.9)
    ]
    model_2 = LinearRegression().fit(x_2, y_2)

    return [model_entire.coef_[0], model_2.coef_[0]]


def expjitterCV(cellpath, stim_start, stim_end):
    reader = AxonIO(filename=cellpath)
    Samprate = reader.get_signal_sampling_rate()
    spiketimes = []
    ISI500list = []
    ISI0list = []
    for seg in reader.read_block(signal_group_mode="split-all").segments:
        Tdur = np.array(seg.t_stop - seg.t_start)
        Vtrace150 = np.array(np.ravel(seg.analogsignals[0])) * 1e-3
        t150 = np.linspace(0, Tdur, len(Vtrace150))
        tt = t150
        vv = Vtrace150
        sweep_ext = EphysSweepFeatureExtractor(
            t=tt,
            v=vv * 1e3,
            filter=len(tt) / tt[-1] / 2500,
            start=stim_start,
            end=stim_end,
        )
        sweep_ext.process_spikes()
        spiket = sweep_ext.spike_feature("peak_t")
        spiketimes.append(spiket)
        a0p5spikes = [i for i in spiket if i > (stim_start + 0.5)]
        b0p5spikes = [i for i in spiket if i < (stim_start + 0.5)]
        if len(a0p5spikes) == 0 or len(b0p5spikes) == 0:
            continue
        ISI500list.append(
            min([i for i in spiket if i > (stim_start + 0.5)])
            - max([i for i in spiket if i < (stim_start + 0.5)])
        )
        ISI0list.append(spiket[1] - spiket[0])

    CV500 = calcCV(ISI500list)
    CV0 = calcCV(ISI0list)
    jitterslopes = calcjitslope(spiketimes, stim_start=stim_start)

    return [jitterslopes, CV0, CV500, spiketimes]


def modeljitterCVhelper(tI_II_modeldict_syn_synwg_synfq):
    # print(tI_II_modeldict_syn_synwg_synfq[0])
    # filepath = injfolderpath+'/'+tI_II_modeldict_syn_synwg_synfq[0]
    modeldict = tI_II_modeldict_syn_synwg_synfq[2]
    tI, II = tI_II_modeldict_syn_synwg_synfq[0], tI_II_modeldict_syn_synwg_synfq[1]
    syn = tI_II_modeldict_syn_synwg_synfq[3]
    synwg = tI_II_modeldict_syn_synwg_synfq[4]
    synfq = tI_II_modeldict_syn_synwg_synfq[5]

    # with open(filepath, 'rb') as file:
    #     tI, II = pickle.load(file)

    tempt, tempv, Ca = mm.runModel(
        modeldict,
        CurrInjection=150e-12,
        vClamp=None,
        refreshKin=False,
        Truntime=0.01,
        syn=syn,
        synwg=synwg,
        synfq=synfq,
    )
    moose.delete("/model/stims/stim0")
    stimtable = moose.StimulusTable("/model/stims/stim2")
    soma = moose.element("/model/elec/soma")
    moose.connect(stimtable, "output", soma, "setInject")

    stimtable.vector = II
    stimtable.stepSize = (
        0  # This forces use of current time as x value for interpolation
    )
    stimtable.stopTime = tI[-1]

    Tdur = tI[-1]
    moose.reinit()
    moose.start(tI[-1])
    Vmvec = moose.element("/model/graphs/plot0").vector
    tvec = moose.element("/Graphs/plott").vector

    tt = tvec
    vv = Vmvec
    sweep_ext = EphysSweepFeatureExtractor(
        t=tt,
        v=vv * 1e3,
        filter=len(tt) / tt[-1] / 2500,
        start=1,
        end=1.9,
    )
    sweep_ext.process_spikes()
    spiket = sweep_ext.spike_feature("peak_t")

    return spiket


def modeljitterCV(
    modeldict, injfolderpath=[100e-12, 150e-12], syn=False, synwg=0.05, synfq=5
):
    """
    If injfolderpath is a list of two float or int, the noisy currents are made manually with injfolderpath[0] as the std of noise and injfolderpath[1] as the amplitude of step current.
    If its a folderpath, the noisy cuurents are taken from there.

    """
    Samprate = 1 / mm.elecDt
    totalsec = 2
    # maxspikes = 0
    minspikes = 1000
    spiketimes = []
    tempt, tempv, Ca = mm.runModel(
        modeldict,
        CurrInjection=150e-12,
        vClamp=None,
        refreshKin=True,
        Truntime=0.01,
        syn=syn,
        synwg=synwg,
        synfq=synfq,
    )
    moose.delete("/model/stims/stim0")
    stimtable = moose.StimulusTable("/model/stims/stim2")
    soma = moose.element("/model/elec/soma")
    moose.connect(stimtable, "output", soma, "setInject")

    tI_list = []
    II_list = []
    if isinstance(injfolderpath, list):
        for i in range(500):
            curr = np.zeros(int(Samprate * totalsec))
            curr[int(1 * Samprate) : int(1.9 * Samprate)] = injfolderpath[1]
            noise = np.random.normal(0, injfolderpath[0], int(Samprate * totalsec))
            curr = curr + noise
            t = np.linspace(0, totalsec, int(totalsec * Samprate))
            tI_list.append(t)
            II_list.append(curr)
    else:
        for injfile in os.listdir(injfolderpath):
            filepath = injfolderpath + "/" + injfile
            with open(filepath, "rb") as file:
                tI, II = pickle.load(file)
                tI_list.append(tI)
                II_list.append(II)

    pool = Pool(processes=min(multiprocessing.cpu_count(), len(tI_list)))

    A = pool.map(
        modeljitterCVhelper,
        zip(
            tI_list,
            II_list,
            np.repeat(modeldict, len(II_list)),
            np.repeat(syn, len(II_list)),
            np.repeat(synwg, len(II_list)),
            np.repeat(synfq, len(II_list)),
        ),
    )

    ISI500list = []
    ISI0list = []
    for spiket in A:
        spiketimes.append(spiket)
        a0p5spikes = [i for i in spiket if i > 1.5]
        b0p5spikes = [i for i in spiket if i < 1.5]
        if len(a0p5spikes) == 0 or len(b0p5spikes) == 0:
            continue
        ISI500list.append(min(a0p5spikes) - max(b0p5spikes))
        ISI0list.append(spiket[1] - spiket[0])

    pool.terminate()

    CV500 = calcCV(ISI500list)
    CV0 = calcCV(ISI0list)
    jitterslopes = calcjitslope(spiketimes, stim_start=1)

    return [jitterslopes, CV0, CV500, spiketimes]


def modeljitterCV_seq(
    modeldict, injfolderpath=[100e-12, 150e-12], syn=False, synwg=0.05, synfq=5
):
    """
    If injfolderpath is a list of two float or int, the noisy currents are made manually with injfolderpath[0] as the std of noise and injfolderpath[1] as the amplitude of step current.
    If its a folderpath, the noisy cuurents are taken from there.

    """
    Samprate = 1 / mm.elecDt
    totalsec = 2
    # maxspikes = 0
    minspikes = 1000
    spiketimes = []
    tempt, tempv, Ca = mm.runModel(
        modeldict,
        CurrInjection=150e-12,
        vClamp=None,
        refreshKin=True,
        Truntime=0.01,
        syn=syn,
        synwg=synwg,
        synfq=synfq,
    )
    moose.delete("/model/stims/stim0")
    stimtable = moose.StimulusTable("/model/stims/stim2")
    soma = moose.element("/model/elec/soma")
    moose.connect(stimtable, "output", soma, "setInject")

    tI_list = []
    II_list = []
    if isinstance(injfolderpath, list):
        for i in range(500):
            curr = np.zeros(int(Samprate * totalsec))
            curr[int(1 * Samprate) : int(1.9 * Samprate)] = injfolderpath[1]
            noise = np.random.normal(0, injfolderpath[0], int(Samprate * totalsec))
            curr = curr + noise
            t = np.linspace(0, totalsec, int(totalsec * Samprate))
            tI_list.append(t)
            II_list.append(curr)
    else:
        for injfile in os.listdir(injfolderpath):
            filepath = injfolderpath + "/" + injfile
            with open(filepath, "rb") as file:
                tI, II = pickle.load(file)
                tI_list.append(tI)
                II_list.append(II)

    A = []
    for a in zip(
        tI_list,
        II_list,
        np.repeat(modeldict, len(II_list)),
        np.repeat(syn, len(II_list)),
        np.repeat(synwg, len(II_list)),
        np.repeat(synfq, len(II_list)),
    ):
        A.append(modeljitterCVhelper(a, model=model))

    ISI500list = []
    ISI0list = []
    for spiket in A:
        spiketimes.append(spiket)
        a0p5spikes = [i for i in spiket if i > 1.5]
        b0p5spikes = [i for i in spiket if i < 1.5]
        if len(a0p5spikes) == 0 or len(b0p5spikes) == 0:
            continue
        ISI500list.append(min(a0p5spikes) - max(b0p5spikes))
        ISI0list.append(spiket[1] - spiket[0])

    CV500 = calcCV(ISI500list)
    CV0 = calcCV(ISI0list)
    jitterslopes = calcjitslope(spiketimes, stim_start=1)

    return [jitterslopes, CV0, CV500, spiketimes]


if __name__ == "__main__":
    stim1391 = [
        "Cell 3 of 181016.abf",
        "cell 4 of 61016.abf",
        "cell 4 of 111016.abf",
        "cell 4 of 131016.abf",
        "Cell 4 of 181016.abf",
        "cell 5 of 61016.abf",
        "Cell 5 of 181016.abf",
        "Cell 2 of 19_10_2016",
        "Cell 1 of 27_10_2016.abf",
        "Cell 1 of 14_10_2016.abf",
        "Cell 4 of 7_10_2016.abf",
        "Cell 6 of 12_10_2016.abf",
        "Cell 7 of 12_10_2016.abf",
    ]
    filename = "cell 4 of 61016.abf"
    if filename in stim1391:
        stim_start = 139.1e-3
        stim_end = 639.1e-3
    else:
        stim_start = 81.4e-3  # in s
        stim_end = 581.4e-3  # in s
    features = expfeatures("Experimental recordings/" + filename, stim_start, stim_end)
    pprint(features)
    plt.plot(*pex.expdata("Experimental recordings/" + filename, 150e-12))
    plt.plot(*pex.expdata("Experimental recordings/" + filename, 300e-12))
    plt.show()
