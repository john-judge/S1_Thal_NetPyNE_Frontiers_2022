import os
from neuron import h
from tune_objective import load_morphologies, average_voltage_traces_into_hVOS_pixels, load_cell_id_to_me_type_map, intersect
import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim, specs
import time
import copy
from netpyne import specs
import importlib.util, os
import sys
import numpy as np
import gc
from cam_params import cam_params_tune as cam_params


pc = h.ParallelContext()
rank = int(pc.id())
sim.clearAll()

def process_and_save_traces(simData_acsf, propVelocity):
    #simData = simData['simData']
    
    #simData_acsf = simData['acsf']
    #simData_nbqx = simData['nbqx']

    cell_id_to_me_type_map = load_cell_id_to_me_type_map('../data/cell_id_to_me_type_map.json')
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

    tvec = np.array(simData_acsf['t'])  # time vector is the same for both conditions


    # save all_cells_rec_acsf and all_cells_rec_nbqx to npy files
    all_trial_save_folder = '../data/grid_acsf/'
    np.save(os.path.join(all_trial_save_folder, f"all_cells_rec_acsf_trial{int(propVelocity)}.npy"), all_cells_rec_acsf)
    del all_cells_rec_acsf, simData_traces_acsf
    gc.collect()

def build_network(acsf=True):
    
    cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg-tune.py', netParamsDefault='netParams.py')

    sim.initialize(
        simConfig = cfg, 	
        netParams = netParams)  				# create network object and set cfg and net params

    sim.net.createPops()               			# instantiate network populations
    sim.net.createCells()              			# instantiate network cells based on defined populations
    sim.net.connectCells()  
    sim.net.addStims() 							# add network stimulation
    sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)

    try:
        print(sim.cfg.synWeightFractionEE[0], "cfg.synWeightFractionEE[0] for ACSF = ", acsf)
    except Exception as e:
        print("Error accessing synWeightFractionEE[0]:", e)
    net_p = None
    try:
        net_p = sim.net.connParams
    except Exception as e:
        try:
            net_p = sim.netParams.connParams
        except Exception as e2:
            print("Error accessing sim.netParams.connParams:", e2)
    if net_p is not None:
        key0 = list(net_p.keys())[0]
        synMechWeightFactor = net_p[key0].get('synMechWeightFactor', None)
        print(f"Example synMechWeightFactor from connParams: {synMechWeightFactor} for ACSF = ", acsf)

# start timer 
if rank == 0:
    start_time = time.time()


build_network()
print("Here is the ACSF value of sim.cfg.synWeightFractionEE from init-tune.py:", sim.cfg.synWeightFractionEE)

# ACSF trial first (no blockade; experiment_NBQX_global should be set to False in cfg-tune.py)
#sim.cfg.filename = 'acsf_run'
sim.runSim()                     			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.pc.barrier()   # Wait for all ranks to finish gatherData
if rank == 0:

    sim.saveData()
    process_and_save_traces(sim.allSimData, sim.cfg.propVelocity)
    end_time = time.time()
    print(f"Total iteration simulation and optical processing time (ACSF only): {(end_time - start_time)/60} minutes")

    
sim.clearAll()
sim.pc.done()

# free memory
del sim.allSimData
gc.collect()
