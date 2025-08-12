"""
init.py

Starting script to run NetPyNE-based S1-thalamus model.

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py

Contributors: salvadordura@gmail.com, fernandodasilvaborges@gmail.com
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim

cfg, netParams = sim.readCmdLineArgs()
sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations

# Collect all synapse mechanism names from connection rules
all_conn_mechs = set()
print(f"Number of connection rules in netParams.connParams: {len(sim.net.params.connParams)}")
for key, val in list(sim.net.params.connParams.items())[:5]:
    print(f"Conn rule: {key}, synMech: {val.get('synMech')}")
    all_conn_mechs.add(val.get('synMech'))

print(f"Total synapse mechanisms in connection rules: {len(all_conn_mechs)}")

# Check if each conn mech has conds that match at least one post cellType
missing_mechs = []
all_cell_types = {pop['cellType'] for pop in sim.net.params.popParams.values()}
print("all synMechParams", sim.net.params.synMechParams.keys())
for mech in all_conn_mechs:
    if mech not in sim.net.params.synMechParams:
        print(f"ERROR: synMech '{mech}' is not defined in netParams.synMechParams!")
        continue
    conds = sim.net.params.synMechParams[mech].get('conds', {})
    cond_types = set()
    if 'cellType' in conds:
        if isinstance(conds['cellType'], list):
            cond_types.update(conds['cellType'])
        else:
            cond_types.add(conds['cellType'])
    missing_targets = all_cell_types - cond_types
    if missing_targets == all_cell_types:
        missing_mechs.append(mech)
        print(f"WARNING: synMech '{mech}' has no matching cellTypes in network!")
    else:
        # Optionally check which cellTypes are NOT covered by this synMech
        uncovered = all_cell_types - cond_types
        if uncovered:
            print(f"synMech '{mech}' missing {len(uncovered)} cellTypes (showing first 5): {list(uncovered)[:5]}")

if missing_mechs:
    print("\nSynapse mechanisms with zero matching post cellTypes:", missing_mechs)

raise Exception("Simulation cannot proceed due to missing synapse mechanisms!")


sim.net.connectCells()            			# create connections between cells based on params
#for c in sim.net.cells:
#    if 'label' not in c.tags:
#        print("c.gid, c.tags", c.gid, c.tags)
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc

#sim.analysis.plotRaster(include=cfg.recordCells, timeRange=[0,cfg.duration], orderBy='gid', orderInverse=True, labels='legend', popRates=True, lw=5, marker='.', markerSize=15, figSize=(18, 12), fontSize=9, dpi=300, saveFig='../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_Raster_onecellperpop.png', showFig=False)
#sim.analysis.plotRaster(include=cfg.popParamLabels, timeRange=[0,cfg.duration], orderBy='gid', orderInverse=True, labels='legend', popRates=True, lw=1, marker='.', markerSize=2, figSize=(18, 12), fontSize=9, dpi=300, saveFig=True, showFig=False)
#sim.analysis.plotTraces(include=cfg.recordCells, overlay=True, oneFigPer='cell', figSize=(12, 4), fontSize=7, saveFig=True)
#sim.analysis.plotTraces(include=cfg.recordCells, overlay=False, oneFigPer='trace', figSize=(18, 12), fontSize=9, saveFig=True)
# features = ['numConns','convergence']
# groups =['pop']
# for feat in features:
#    for group in groups:
#        sim.analysis.plotConn(includePre=['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], includePost=['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], feature=feat, groupBy=group, figSize=(24,24), saveFig=True, orderBy='gid', graphType='matrix', fontSize=18, saveData='../data/'+cfg.simLabel[0:9]+'/'+cfg.simLabel + '_' + group + '_' + feat+ '_matrix.json')
