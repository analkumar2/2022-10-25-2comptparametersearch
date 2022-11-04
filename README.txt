Here we try to get a model to fit a single exp cell (say to 2017_11_17_2.abf).
Let first try to fit raw trace at -50pA, 0pA, 50pA, ... 400pA

Here also, calculate the offset = E_rest - E_rest of the cell

####################################################################################################
Then, we calculate features at 150pA and 300pA and fit to those features

Lets fix the kinetics
We change only gbars of Na_T, K_P, K_31, K_T, h, K_M
We also change slightly Rm_soma, Cm_soma, Rm_dend, Cm_soma, Ra, Em

We do the simulation, then calculate E_rest. Then, we define offset = E_rest - E_rest of the cell. We now subtract this offset from the model traces. We calculate the features from these traces. This is beacuse we do not know the proper offset
