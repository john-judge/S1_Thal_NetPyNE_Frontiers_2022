from copy import deepcopy
import os
"""
init.py

Starting script to tune ACSF and NBQX simulations of NetPyNE-based S1 model.
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim

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

net_filepath = os.path.join(cfg.saveFolder, "base_net_tuning.pkl")
if not os.path.isfile(net_filepath):
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
    sim.net.save(net_filepath)
    print(f"Network saved to {net_filepath}")

else:
    print(f"Loading pre-built network from {net_filepath}...")
    cfg.loadFromFile = True  # ensures network is loaded instead of rebuilt
    sim.initialize(cfg=cfg, netParams=netParams) 

# ACSF trial first (no blockade; experiment_NBQX_global should be set to False in cfg-tune.py)
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
acsf_data = deepcopy(sim.allSimData)  # save ACSF data

# now run NBQX trial
set_syn_blockade(fraction=fraction_blockade)
sim.setupAndRun()
nbqx_data = deepcopy(sim.allSimData)  # save NBQX data
sim.allSimData = {'nbqx': nbqx_data, 'acsf': acsf_data}

sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#

#sim.analysis.plotRaster(include=cfg.recordCells, timeRange=[0,cfg.duration], orderBy='gid', orderInverse=True, labels='legend', popRates=True, lw=5, marker='.', markerSize=15, figSize=(18, 12), fontSize=9, dpi=300, saveFig='../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_Raster_onecellperpop.png', showFig=False)
#sim.analysis.plotRaster(include=cfg.popParamLabels, timeRange=[0,cfg.duration], orderBy='gid', orderInverse=True, labels='legend', popRates=True, lw=1, marker='.', markerSize=2, figSize=(18, 12), fontSize=9, dpi=300, saveFig=True, showFig=False)
#sim.analysis.plotTraces(include=cfg.recordCells, overlay=True, oneFigPer='cell', figSize=(12, 4), fontSize=7, saveFig=True)
#sim.analysis.plotTraces(include=cfg.recordCells, overlay=False, oneFigPer='trace', figSize=(18, 12), fontSize=9, saveFig=True)
# features = ['numConns','convergence']
# groups =['pop']
# for feat in features:
#    for group in groups:
#        sim.analysis.plotConn(includePre=['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], includePost=['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], feature=feat, groupBy=group, figSize=(24,24), saveFig=True, orderBy='gid', graphType='matrix', fontSize=18, saveData='../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_' + group + '_' + feat+ '_matrix.json')
