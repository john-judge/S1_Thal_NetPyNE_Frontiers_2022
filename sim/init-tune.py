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

def readCmdLineArgs_nbqx(simConfigDefault='cfg.py', netParamsDefault='netParams.py', acsf=True):
    """
    Based on `netpyne.sim.setup.readCmdLineArgs` but allows
    cfg to be modified to enable NBQX (cfg.experiment_NBQX_global = True)
    before netParams is loaded.

    Parameters
    ----------
    simConfigDefault : str

    netParamsDefault : str


    """

    import __main__
    from netpyne import sim, specs

    if len(sys.argv) > 1:
        print(
            '\nReading command line arguments using syntax: python file.py [simConfig=filepath] [netParams=filepath]'
        )
    cfgPath = None
    netParamsPath = None

    # read simConfig and netParams paths
    for arg in sys.argv:
        if arg.startswith('simConfig='):
            cfgPath = arg.split('simConfig=')[1]

        elif arg.startswith('netParams='):
            netParamsPath = arg.split('netParams=')[1]

    print(f'cmd line cfgPath: {cfgPath}, netParamsPath: {netParamsPath}')

    if cfgPath is None and simConfigDefault is not None:
        cfgPath = simConfigDefault
    if netParamsPath is None and netParamsDefault is not None:
        netParamsPath = netParamsDefault

    if cfgPath:
        print(f'Importing simConfig from {cfgPath}')
        if cfgPath.endswith('.py'):
            cfgModule = sim.loadPythonModule(cfgPath)
            cfg = cfgModule.cfg
        else:
            cfg = sim.loadSimCfg(cfgPath, setLoaded=False)
        __main__.cfg = cfg

        if not cfg:
            print('\nWarning: Could not load cfg from command line path or from default cfg.py')
            print('This usually occurs when cfg.py crashes.  Please ensure that your cfg.py file')
            print('completes successfully on its own (i.e. execute "python cfg.py" and fix any bugs).')
    else:
        print('\nNo command line argument or default value for cfg provided.')
        cfg = None

    # modify cfg here before loading netParams
    # to enable NBQX
    # http://doc.netpyne.org/user_documentation.html#running-a-batch-job-beta
    if not acsf:
        print("Modifying cfg for NBQX simulation")
        cfg.update({'experiment_NBQX_global': True,
                    'synWeightFractionEE': [cfg.partial_blockade_fraction, 1.0],
                    'synWeightFractionEI': [cfg.partial_blockade_fraction, 1.0]}, force_match=True)
        #cfg.experiment_NBQX_global = True  # if ACSF is False, then NBQX is True
        cfg.synWeightFractionEE[0] = cfg.partial_blockade_fraction
        cfg.synWeightFractionEI[0] = cfg.partial_blockade_fraction

    if netParamsPath:
        print(f'CFG after modification for NBQX: {cfg.synWeightFractionEE}, Importing netParams from {netParamsPath}')
        if netParamsPath.endswith('py'):
            netParamsModule = sim.loadPythonModule(netParamsPath)
            netParams = netParamsModule.netParams
        else:
            netParams = sim.loadNetParams(netParamsPath, setLoaded=False)

        if not netParams:
            print('\nWarning: Could not load netParams from command line path or from default netParams.py')
            print('This usually occurs when netParams.py crashes.  Please ensure that your netParams.py file')
            print('completes successfully on its own (i.e. execute "python netParams.py" and fix any bugs).')
    else:
        print('\nNo command line argument or default value for netParams provided.')
        netParams = None

    return cfg, netParams

def build_network(acsf=True):
    
    cfg, netParams = None, None
    cfg, netParams = readCmdLineArgs_nbqx(simConfigDefault='cfg-tune.py', netParamsDefault='netParams.py', acsf=acsf)
    if acsf:
        cfg.filename = 'acsf_run'
    else:
        cfg.filename = 'nbqx_run'
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