import os
from neuron import h
import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim, specs
import time
import copy
from netpyne import specs
import importlib.util, os
import sys
import numpy as np
import gc


pc = h.ParallelContext()
rank = int(pc.id())
sim.clearAll()


def build_network():
    
    cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg-tune.py', netParamsDefault='netParams.py')

    sim.initialize(
        simConfig = cfg, 	
        netParams = netParams)  				# create network object and set cfg and net params

    sim.net.createPops()               			# instantiate network populations
    sim.net.createCells()              			# instantiate network cells based on defined populations
    sim.net.connectCells()  
    sim.net.addStims() 							# add network stimulation
    sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)

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
    end_time = time.time()
    print(f"Total iteration simulation and optical processing time (ACSF only): {(end_time - start_time)/60} minutes")

    
sim.clearAll()
sim.pc.done()

# free memory
del sim.allSimData
gc.collect()
