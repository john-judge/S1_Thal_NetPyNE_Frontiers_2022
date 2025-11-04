import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# gather ACSF simulation data into a map for easy lookup during NBQX tuning
# To be run on Condor after grid_acsf.py has completed
# grid_acsf_map[propVelocity] = simData_acsf

data_dir = f'S1_Thal_NetPyNE_Frontiers_2022/grid/grid_acsf/'
grid_acsf_map = {}
for file in os.listdir(data_dir):
    if file.endswith('.pkl'):
        with open(os.path.join(data_dir, file), 'rb') as f:
            simData_acsf = pickle.load(f)
            grid_acsf_map[simData_acsf['simConfig']['propVelocity']] = simData_acsf

with open('../../grid_acsf_map.pkl', 'wb') as f:
    pickle.dump(grid_acsf_map, f)
