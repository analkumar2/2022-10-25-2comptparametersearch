import moose
import rdesigneur as rd
import numpy as np

numDendSegments = 1
numBranches = 1
comptLen = 500e-6
comptDia = 4e-6
RM_soma = 0.11696
CM_soma = 1e-2
RA_soma = 1.18
RM_dend = 1.86794368
CM_dend = 0.02411617
RA_dend = 1.18
Em = -82e-3
initVm = -82e-3
stimamp = 150e-12
stim_start = 0.2
elecDt = 5e-5
elecPlotDt = 5e-5
tstop = 1
# act_pas = [168, 138]
act_pas = [197e6,136.5e-12]
gbarscaling = 0.7
gbarscaling_singlecompt = gbarscaling*(np.pi*10.075e-6*10.075e-6)/(np.pi*500e-6*500e-6)

def makeBranchingneuronProto(numBranches=1, RM_dend=1.86794368, CM_dend=0.02411617, RA_dend=1.18):
    BNeuron = moose.Neuron( '/library/BNeuron' )
    soma = rd.buildCompt( BNeuron, 'soma', RM = RM_soma, RA = RA_soma, CM = CM_soma, dia = 10.075e-6, x = 0.0, y = 0.0, z = 0.0, dx = 10.075e-6, dy = 0.0, dz = 0.0, Em = Em, initVm = initVm)

    for dendbranch in range(numBranches):
        prev = soma
        dx = np.cos(np.pi*dendbranch/2/numBranches - np.pi/4) * comptLen
        dy = np.sin(np.pi*dendbranch/2/numBranches - np.pi/4) * comptLen
        for seg in range(numDendSegments):
            x = np.cos(np.pi*dendbranch/2/numBranches - np.pi/4) * (seg*comptLen + 15e-6)
            y = np.sin(np.pi*dendbranch/2/numBranches - np.pi/4) * (seg*comptLen + 15e-6)
            
            compt = rd.buildCompt( BNeuron, f'dend_{dendbranch}_{seg}', RM = RM_dend, RA = RA_dend, CM = CM_dend, dia = comptDia, x = x, y = y, z = 0.0, dx = dx, dy = dy, dz = 0.0, Em = Em, initVm = initVm)
            moose.connect(prev, 'axial', compt, 'raxial')
            prev = compt
            x = x+dx
            y = y+dy

    return BNeuron

moose.Neutral( '/library' )
makeBranchingneuronProto(numBranches,RM_dend, CM_dend, RA_dend)

rdes = rd.rdesigneur(
    # cellProto syntax: ['ballAndStick', 'name', somaDia, somaLength, dendDia, dendLength, numDendSegments ]
    # The numerical arguments are all optional
    # cellProto = [['ballAndStick', 'soma', 20e-6, 20e-6, 4e-6, 500e-6, 10]],
    # cellProto = [['somaProto', 'soma', 20e-6, 200e-6]],
    cellProto = [['elec','BNeuron']],
    chanProto = [['make_HH_Na()', 'Na'], ['make_HH_K()', 'K']],
    chanDistrib = [
        ['Na', 'soma', 'Gbar', '1200' ],
        ['K', 'soma', 'Gbar', '360' ],
        ['Na', 'dend#', 'Gbar', '400' ],
        ['K', 'dend#', 'Gbar', '120' ]
        ],
    stimList = [['soma', '1', '.', 'inject', '(t>0.01 && t<0.05) * 1e-9' ]],
    plotList = [['soma', '1', '.', 'Vm', 'Membrane potential']],
    # moogList = [['#', '1', '.', 'Vm', 'Vm (mV)']]
)
rdes.buildModel()
moose.reinit()
moose.start(0.1)
rdes.display()

moose.delete('/model')
rdes = rd.rdesigneur(
    # cellProto syntax: ['ballAndStick', 'name', somaDia, somaLength, dendDia, dendLength, numDendSegments ]
    # The numerical arguments are all optional
    # cellProto = [['ballAndStick', 'soma', 20e-6, 20e-6, 4e-6, 500e-6, 10]],
    # cellProto = [['somaProto', 'soma', 20e-6, 200e-6]],
    cellProto = [['elec','BNeuron']],
    chanProto = [['make_HH_Na()', 'Na'], ['make_HH_K()', 'K']],
    chanDistrib = [
        ['Na', 'soma', 'Gbar', '1200' ],
        ['K', 'soma', 'Gbar', '360' ],
        ['Na', 'dend#', 'Gbar', '400' ],
        ['K', 'dend#', 'Gbar', '120' ]
        ],
    stimList = [['soma', '1', '.', 'inject', '(t>0.01 && t<0.05) * 1e-9' ]],
    plotList = [['soma', '1', '.', 'Vm', 'Membrane potential']],
    # moogList = [['#', '1', '.', 'Vm', 'Vm (mV)']]
)
rdes.buildModel()
moose.reinit()
moose.start(0.1)
rdes.display()
