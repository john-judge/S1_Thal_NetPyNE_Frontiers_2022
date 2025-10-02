from copy import deepcopy
import os
"""
init.py

Starting script to tune ACSF and NBQX simulations of NetPyNE-based S1 model.
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim, specs


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



cfg, netParams = sim.readCmdLineArgs()
netFile = os.path.join(cfg.simLabel, 'base_net_tuning.pkl')
if os.path.exists(netFile):
    print(f"Loading pre-built network from base_net_tuning...")
    cfg.loadFromFile = True  # ensures network is loaded instead of rebuilt
    sim.initialize(cfg=cfg, netParams=netParams) 
else:
    print(("building network from scratch with netParams"))

    sim.initialize(
        simConfig = cfg, 	
        netParams = netParams)  				# create network object and set cfg and net params

    fraction_blockade = cfg.partial_blockade_fraction 

    sim.net.createPops()               			# instantiate network populations
    sim.net.createCells()              			# instantiate network cells based on defined populations
    sim.net.connectCells()  
    sim.net.addStims() 							# add network stimulation
    sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
    # Save network for future runs
    os.makedirs(cfg.simLabel, exist_ok=True)
    cfg.filename = 'base_net_tuning'
    cfg.savePickle = True 
    sim.saveData()
    print(f"Network saved to 'base_net_tuning.pkl' in {cfg.simLabel}")
    

# ACSF trial first (no blockade; experiment_NBQX_global should be set to False in cfg-tune.py)
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
acsf_data = deepcopy(sim.allSimData)  # save ACSF data
sim.allSimData = {}

sim.clearAll()
# now run NBQX trial
set_syn_blockade(fraction=fraction_blockade)
sim.runSim()                     
sim.gatherData()  
nbqx_data = deepcopy(sim.allSimData)  # save NBQX data
sim.allSimData = {'nbqx': nbqx_data, 'acsf': acsf_data}

