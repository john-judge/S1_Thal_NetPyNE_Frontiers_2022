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

all_conn_mechs = set()
for connName, conn in sim.net.params.connParams.items():
    synMech = conn.get('synMech')
    if isinstance(synMech, list):
        all_conn_mechs.update(synMech)
    elif synMech:
        all_conn_mechs.add(synMech)

all_defined_mechs = set(sim.net.params.synMechParams.keys())

print(f"Total synMechs referenced in connParams: {len(all_conn_mechs)}")
print(f"Total synMechs defined in synMechParams: {len(all_defined_mechs)}")

# Step 3: Missing synMechs
missing_mechs = all_conn_mechs - all_defined_mechs
if missing_mechs:
    print(f"\nERROR: SynMechs referenced in connParams but missing in synMechParams ({len(missing_mechs)}):")
    for mech in missing_mechs:
        print(f"  - {mech}")
else:
    print("\nNo synMechs missing from synMechParams.")

# Step 4: Check synMech conds cellType coverage
all_cell_types = {pop['cellType'] for pop in sim.net.params.popParams.values()}

print("\nChecking synMech conds against populations:")
for mech in all_defined_mechs:
    conds = sim.net.params.synMechParams[mech].get('conds', {})
    cond_types = set()
    if 'cellType' in conds:
        if isinstance(conds['cellType'], list):
            cond_types.update(conds['cellType'])
        else:
            cond_types.add(conds['cellType'])
    else:
        # No cellType condition means synMech applies to all cellTypes
        continue

    missing_targets = cond_types - all_cell_types
    unused_targets = all_cell_types - cond_types
    if cond_types.isdisjoint(all_cell_types):
        print(f"WARNING: synMech '{mech}' conds cellType(s) {cond_types} do not match ANY populations' cellTypes!")
    elif missing_targets:
        print(f"WARNING: synMech '{mech}' conds include cellTypes not in populations: {missing_targets}")
    elif unused_targets:
        print(f"NOTE: synMech '{mech}' does not cover cellTypes: {unused_targets}")

print("\nDone checking synMechs.")



print("Verify cell populations contain cells with correct cellType and tags...")
num_checks = 50
for popName, pop in sim.net.params.popParams.items():
    size = pop.get('numCells', 'unknown')
    cellType = pop.get('cellType', 'unknown')
    tags = pop.get('tags', {})
    print(f"Pop {popName}: size={size}, cellType={cellType}, tags={tags}")
    num_checks -= 1
    if num_checks <= 0:
        break

print("Check connection probabilities")
num_checks = 50
for connName, conn in sim.net.params.connParams.items():
    prob = conn.get('probability', None)
    print(f"{connName}: prob={prob}")
    num_checks -= 1
    if num_checks <= 0:
        break

raise Exception("Check the console output for any warnings or errors related to synMechs, cellTypes, and connection probabilities.")

sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations




sim.net.connectCells()            			# create connections between cells based on params




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
