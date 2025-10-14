import os
from neuron import h
"""
init.py

Starting script to tune ACSF and NBQX simulations of NetPyNE-based S1 model.
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim, specs
import time
import copy


pc = h.ParallelContext()
rank = int(pc.id())

def set_syn_blockade(fraction):
    """
    Scale AMPA connection weights by `fraction`.
    fraction = 0.0 -> NBQX (block AMPA)
    fraction = 1.0 -> ACSF (normal)
    """
    modification_params = {
        'conds': {
            # target only AMPA synapses
            'synMech': ['AMPA']
        },
        'set': {
            'weight': lambda w: w * fraction  # scale relative to current weight
        }
    }
    sim.net.modifyConns(modification_params)

from netpyne import specs
import importlib.util, os

def load_cfg_and_netparams(cfg_file, netparams_file, acsf=True):
    """Reload cfg and netParams after modifying cfg."""
    # Load cfg file
    cfg_context = {}
    exec(open(cfg_file).read(), cfg_context)
    cfg = cfg_context['cfg']

    if not acsf:
        cfg.experiment_NBQX_global = True  # if ACSF is False, then NBQX is True
        cfg.synWeightFractionEE[0] = cfg.partial_blockade_fraction
        cfg.synWeightFractionEI[0] = cfg.partial_blockade_fraction

    # Make sure specs and cfg are visible when loading netParams
    np_context = {'cfg': cfg, 'specs': specs}
    exec(open(netparams_file).read(), np_context)
    netParams = np_context['netParams']

    return cfg, netParams


def build_network(acsf=True):
    
    #cfg, netParams = sim.readCmdLineArgs()
    cfg, netParams = load_cfg_and_netparams('cfg-tune.py', 'netParams.py', acsf=acsf)

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


# start timer 
if rank == 0:
    start_time = time.time()

build_network()
# ACSF trial first (no blockade; experiment_NBQX_global should be set to False in cfg-tune.py)
sim.cfg.filename = 'acsf_run'
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.pc.barrier()   # Wait for all ranks to finish gatherData
if rank == 0:
    try:
        acsf_data = copy.deepcopy(dict(sim.allSimData)) # save ACSF data, deep copy
    except Exception as e:
        print("Error copying ACSF data:", e)
        acsf_data = dict(sim.allSimData)

sim.allSimData = {}  # clear before next sim
build_network(acsf=False)
sim.cfg.filename = 'nbqx_run'
sim.runSim()                     
sim.gatherData()  
sim.pc.barrier()   # Wait for all ranks to finish gatherData
if rank == 0:
    try:
        nbqx_data = copy.deepcopy(dict(sim.allSimData))  # save NBQX data, deep copy
    except Exception as e:
        print("Error copying NBQX data:", e)
        nbqx_data = dict(sim.allSimData)

    sim.allSimData = {'simData': {'acsf': acsf_data, 'nbqx': nbqx_data}}
    sim.saveData()

    end_time = time.time()
    print(f"Total iteration simulation time (both ACSF and NBQX): {(end_time - start_time)/60} minutes")

