import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tune_objective import load_morphologies, average_voltage_traces_into_hVOS_pixels, load_cell_id_to_me_type_map, intersect
import sys
from cam_params import cam_params_tune as cam_params
import gc


def process_and_save_traces(simData_acsf, propVelocity):
    #simData = simData['simData']
    
    #simData_acsf = simData['acsf']
    #simData_nbqx = simData['nbqx']

    cell_id_to_me_type_map = load_cell_id_to_me_type_map(simData_acsf['net'], curr_trial=propVelocity)
    cells_acsf, me_type_morphology_map = load_morphologies(simData_acsf, cell_id_to_me_type_map)
    
    # hVOS/optical processing
    rois_to_sample = []
    roi_size = 3  # 3x3 pixel ROIs
    n_rois = 400
    # randomly sample 60 non-overlapping ROIs of size 3x3 pixels
    np.random.seed(4321)
    for _ in range(n_rois):
        attempts = 10
        while True:
            x = np.random.randint(0, cam_params['cam_width'] - roi_size)
            y = np.random.randint(0, cam_params['cam_height'] - roi_size)
            roi = (x, y, x + roi_size, y + roi_size)
            if not any(intersect(roi, r) for r in rois_to_sample):
                rois_to_sample.append(roi)
                break
            attempts -= 1
            if attempts == 0:
                print("Could not find non-overlapping ROI after 10 attempts, stopping ROI selection.")
                break

    simData_traces_acsf, all_cells_rec_acsf = average_voltage_traces_into_hVOS_pixels(simData_acsf, cells_acsf, 
                                                                  me_type_morphology_map, rois_to_sample)


    # save all_cells_rec_acsf and all_cells_rec_nbqx to npy files
    #all_trial_save_folder = '../data/grid_acsf/'
    #label = str(propVelocity).replace('.', 'p')
    #np.save(os.path.join(all_trial_save_folder, f"simData_traces_acsf_{label}.npy"), simData_traces_acsf)
    del all_cells_rec_acsf
    gc.collect()
    return simData_traces_acsf

# gather ACSF simulation data into a map for easy lookup during NBQX tuning
# To be run on Condor after grid_acsf.py has completed
# grid_acsf_map[propVelocity] = simData_acsf
n_jobs = 10
if len(sys.argv) > 1:
    if sys.argv[1] == '':
        job_id = 0
    else:
        job_id = int(sys.argv[1])
job_id %= n_jobs  # make sure job_id is in range 0 to n_jobs-1

print(os.getcwd())
data_dir = f'../data/grid_acsf/'
grid_acsf_map = {}
for file in os.listdir(data_dir):
    if file.endswith('.pkl'):
        with open(os.path.join(data_dir, file), 'rb') as f:
            simData_acsf = pickle.load(f)
            processed_traces = process_and_save_traces(simData_acsf, simData_acsf['simConfig']['propVelocity'])
            grid_acsf_map[simData_acsf['simConfig']['propVelocity']] = processed_traces

with open(f'../../grid_acsf_map{job_id}.pkl', 'wb') as f:
    pickle.dump(grid_acsf_map, f)
print(f"Saved grid_acsf_map{job_id}.pkl with {len(grid_acsf_map)} entries.")