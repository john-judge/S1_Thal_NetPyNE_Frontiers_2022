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
from netpyne import specs
import importlib.util, os
import sys


pc = h.ParallelContext()
rank = int(pc.id())
sim.clearAll()

def build_network():
    
    cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg-tune-nbqx.py', netParamsDefault='netParams.py')
    sim.initialize(
        simConfig = cfg, 	
        netParams = netParams)  				# create network object and set cfg and net params

    sim.net.createPops()               			# instantiate network populations
    sim.net.createCells()              			# instantiate network cells based on defined populations
    sim.net.connectCells()  
    sim.net.addStims() 							# add network stimulation
    sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)

    try:
        print(sim.cfg.synWeightFractionEE[0], "cfg.synWeightFractionEE[0] for NBQX")
    except Exception as e:
        print("Error accessing synWeightFractionEE[0]:", e)
    
# start timer 
if rank == 0:
    start_time = time.time()


build_network()
print("Here is the NBQX value of sim.cfg.synWeightFractionEE from init-tune.py:", sim.cfg.synWeightFractionEE)
#sim.cfg.filename = 'nbqx_run'
sim.runSim()                     
sim.gatherData()  
if rank == 0:
    propVelocity = sim.cfg.propVelocity
    sim.allSimData['propVelocity'] = propVelocity

    sim.saveData()
'''sim.pc.barrier()   # Wait for all ranks to finish gatherData
if rank == 0:
    try:
        nbqx_data = copy.deepcopy(dict(sim.allSimData))  # save NBQX data, deep copy
    except Exception as e:
        print("Error copying NBQX data:", e)
        nbqx_data = dict(sim.allSimData)
sim.pc.done()

if rank == 0:
    # get the propVelocity value for this trial from sim.cfg
    propVelocity = sim.cfg.propVelocity

# free memory
del sim.allSimData

sim.pc.done()
sim.clearAll()'''