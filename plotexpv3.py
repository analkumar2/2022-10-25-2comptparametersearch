# exec(open('plotexpv3.py').read())

#v2: now also plots dat files
#v3: Now handles Vclamp as well as Iclamp. Instead of currentlevel provide index.


import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from neo.io import AxonIO
import os
import argparse

def expdata(Address, Index=0, mode='Iclamp'):
    '''
    mode: Either Iclamp or Vclamp.
    '''
    if os.path.splitext(Address)[1] == '.abf':
        reader = AxonIO(filename=Address)
        Samprate = reader.get_signal_sampling_rate()
        seg = reader.read_block(signal_group_mode='split-all').segments[Index]
        Tdur = np.array(seg.t_stop - seg.t_start)
        if mode=='Iclamp':
            V = np.array(np.ravel(seg.analogsignals[0]))*1e-3
            return [np.linspace(0,Tdur+0,len(V)), V]
        elif mode=='Vclamp':
            I = np.array(np.ravel(seg.analogsignals[0]))*1e-12
            return [np.linspace(0,Tdur+0,len(I)), I]
    elif os.path.splitext(Address)[1] == '.dat':
        a = np.loadtxt(Address)
        b = np.transpose(a)
        T = b[0]
        if mode=='Iclamp':
            V = b[1:][Index]*1e-3
            return [T,V]
        elif mode=='Vclamp':
            I = b[1:][Index]*1e-12
            return [T,I]

def plotexp(Address, Index=0, mode='Iclamp', Title='Current clamp at 150pA'):
    '''
    mode: Either Iclamp or Vclamp.
    '''
    if mode=='Iclamp':
        T, V = expdata(Address, Index, mode)
        plt.plot(T,V-0.016, label=Address)
        plt.legend()
        plt.title(Title)
        plt.xlabel('Time (s)')
        plt.ylabel('Membrane potential (V)')
        plt.show()
    elif mode=='Vclamp':
        T, I = expdata(Address, Index, mode)
        plt.plot(T,I, label=Address)
        plt.legend()
        plt.title(Title)
        plt.xlabel('Time (s)')
        plt.ylabel('Holding current (A)')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saves all the plots of all the abf files in the folder recursively')
    parser.add_argument('--Address', type=str, default='../../Raw_data/Deepanjali_data/Organized_Anal/WT_prepostapa/IF_preapa/2017_12_20_1.abf', nargs='?')
    parser.add_argument('--Index', type=int, default=10, nargs='?')
    parser.add_argument('--mode', type=str, default='Iclamp', nargs='?')
    parser.add_argument('--Title', type=str, default='Current clamp at 150pA', nargs='?')
    args = parser.parse_args()
    plotexp(args.Address, args.Index, args.mode, args.Title)
