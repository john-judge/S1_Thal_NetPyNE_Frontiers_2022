import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tune_objective import average_voltage_traces_into_hVOS_pixels
import sys

# gather ACSF simulation data into a map for easy lookup during NBQX tuning
# To be run on Condor after grid_acsf.py has completed
# grid_acsf_map[propVelocity] = simData_acsf
n_jobs = 10
if len(sys.argv) > 1:
    job_id = int(sys.argv[1])
job_id %= n_jobs  # make sure job_id is in range 0 to n_jobs-1

data_dir = f'../grid/grid_acsf/'
grid_acsf_map = {}
for file in os.listdir(data_dir):
    if file.endswith('.pkl'):
        with open(os.path.join(data_dir, file), 'rb') as f:
            simData_acsf = pickle.load(f)
            grid_acsf_map[simData_acsf['simConfig']['propVelocity']] = simData_acsf

with open(f'../../grid_acsf_map{job_id}.pkl', 'wb') as f:
    pickle.dump(grid_acsf_map, f)
