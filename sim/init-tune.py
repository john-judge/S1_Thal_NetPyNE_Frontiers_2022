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

def build_network(acsf=True):
    
    cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg-tune-nbqx.py', netParamsDefault='netParams.py', acsf=acsf)
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


build_network(acsf=False)
print("Here is the NBQX value of sim.cfg.synWeightFractionEE from init-tune.py:", sim.cfg.synWeightFractionEE)
#sim.cfg.filename = 'nbqx_run'
sim.runSim()                     
sim.gatherData()  
sim.pc.barrier()   # Wait for all ranks to finish gatherData
if rank == 0:
    try:
        nbqx_data = copy.deepcopy(dict(sim.allSimData))  # save NBQX data, deep copy
    except Exception as e:
        print("Error copying NBQX data:", e)
        nbqx_data = dict(sim.allSimData)
sim.pc.done()
sim.clearAll()
pc = h.ParallelContext()

sim.allSimData = {}  # clear before next sim
build_network()
print("Here is the ACSF value of sim.cfg.synWeightFractionEE from init-tune.py:", sim.cfg.synWeightFractionEE)

# ACSF trial first (no blockade; experiment_NBQX_global should be set to False in cfg-tune.py)
#sim.cfg.filename = 'acsf_run'
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.pc.barrier()   # Wait for all ranks to finish gatherData
if rank == 0:
    try:
        acsf_data = copy.deepcopy(dict(sim.allSimData)) # save ACSF data, deep copy
    except Exception as e:
        print("Error copying ACSF data:", e)
        acsf_data = dict(sim.allSimData)


    sim.allSimData = {'simData': {'acsf': acsf_data, 'nbqx': nbqx_data}}
    sim.saveData()

    end_time = time.time()
    print(f"Total iteration simulation time (both ACSF and NBQX): {(end_time - start_time)/60} minutes")

sim.pc.done()
sim.clearAll()