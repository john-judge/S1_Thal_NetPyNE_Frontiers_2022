"""
cfg.py 

Simulation configuration for S1-thalamus model (using NetPyNE)
This file has sim configs as well as specification for parameterized values in netParams.py 

Contributors: salvadordura@gmail.com, fernandodasilvaborges@gmail.com
"""

from netpyne import specs
import pickle
import os
import numpy as np

from recordTraceBatchSettings import record_trace_setting


cfg = specs.SimConfig()  

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------

# (224., 752., 94.) is the location of cell index1 soma
# (145.,844.,34)) soma index0
# near center L4: (185., 800., 64.,)
xStimLocation =[185.,800.,64.]
# z_recording_radius = 1000 # microns (box shape)
cfg.simType='S1_TH_coreneuron'
cfg.coreneuron = False


#------------------------------------------------------------------------------
# Experiments
#------------------------------------------------------------------------------

#run 10: no stim
#run 8: baseline stim
#run 11: NBQX + no stim
cfg.experiment_NBQX_global = False  # For run9
cfg.experiment_dendritic_somatic_inhibition = False  # for run12
cfg.experiment_amp_stim = False  # for run13

#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.duration = 99.8 ## Duration of the sim, in ms  
cfg.dt = 0.025
cfg.seeds = {'cell': 4322, 'conn': 4322, 'stim': 4322, 'loc': 4322} 
cfg.hParams = {'celsius': 34, 'v_init': -65} # room temperature (slice)
cfg.verbose = True
cfg.createNEURONObj = True
cfg.createPyStruct = True  
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.1

cfg.includeParamsLabel = True
cfg.printPopAvgRates = True
cfg.checkErrors = False

#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.rootFolder = os.getcwd()

# Load cells info from previously saved using netpyne (False: load from HOC BBP files, slower)
cfg.loadcellsfromJSON = True

cfg.poptypeNumber = 61 # max 55 + 6
cfg.celltypeNumber = 213 # max 207 + 6

cfg.cao_secs = 1.2

cfg.use_frac = {} # use[invivo] = cfg.use_frac * use[invitro]

cfg.use_frac['EIproximal'] = 0.75 # shallow dependence between PC-proximal targeting cell types (LBCs, NBCs, SBCs, ChC)
cfg.use_frac['Inh'] = 0.50 # Pathways that had not been studied experimentally were assumed to have an intermediate level of dependence
cfg.use_frac['EE'] = 0.25 # steep Ca2+ dependence for connections between PC-PC and PC-distal targeting cell types (DBC, BTC, MC, BP)
cfg.use_frac['EIdistal'] = 0.25 

# TO DEBUG - import and simulate only the Cell soma (to study only the Net)
cfg.reducedtest = False    

#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------  
# S1 Cells
# Load 55 Morphological Names and Cell pop numbers -> L1:6 L23:10 L4:12 L5:13 L6:14
# Load 207 Morpho-electrical Names used to import the cells from 'cell_data/' -> L1:14 L23:43 L4:46 L5:52 L6:52
# Create [Morphological,Electrical] = number of cell metype in the sub-pop

with open('cells/S1-cells-distributions-Rat.txt') as mtype_file:
    mtype_content = mtype_file.read()       

cfg.popNumber = {}
cfg.cellNumber = {} 
cfg.popLabel = {} 
popParam = []
cellParam = []
cfg.meParamLabels = {} 
cfg.popLabelEl = {} 
cfg.cellLabel = {}

for line in mtype_content.split('\n')[:-1]:
    cellname, mtype, etype, n, m = line.split()
    metype = mtype + '_' + etype[0:3]
    cfg.cellNumber[metype] = int(n)
    cfg.popLabel[metype] = mtype
    cfg.popNumber[mtype] = int(m)
    cfg.cellLabel[metype] = cellname

    if mtype not in popParam:
        popParam.append(mtype)
        cfg.popLabelEl[mtype] = [] 
               
    cfg.popLabelEl[mtype].append(metype)
    
    cellParam.append(mtype + '_' + etype[0:3])
    
cfg.S1pops = popParam[0:55]
cfg.S1cells = cellParam[0:207]

#------------------------------------------------------------------------------  
# Thalamic Cells

cfg.thalamicpops = ['ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1']

cfg.cellNumber['ss_RTN_o'] = int(382 * (210**2/150**2)) # from mouse model (r = 150 um)
cfg.cellNumber['ss_RTN_m'] = int(382 * (210**2/150**2))
cfg.cellNumber['ss_RTN_i'] = int(765 * (210**2/150**2))
cfg.cellNumber['VPL_sTC'] = int(656 * (210**2/150**2))
cfg.cellNumber['VPM_sTC'] = int(839 * (210**2/150**2))
cfg.cellNumber['POm_sTC_s1'] = int(685 * (210**2/150**2))

for mtype in cfg.thalamicpops: # No diversity
	metype = mtype
	popParam.append(mtype)
	cfg.popLabel[metype] = mtype
	cellParam.append(metype)

	cfg.popNumber[mtype] = cfg.cellNumber[metype]

#------------------------------------------------------------------------------  
cfg.popParamLabels = popParam
cfg.cellParamLabels = cellParam

#--------------------------------------------------------------------------
# Recording 
#--------------------------------------------------------------------------

## only L4 SS and L4 PC
target_me_types = ['L4_SS', 'L4_PC']  # only used if cfg.cellsrec = 2 or 3. if None, record all cells
fraction_record = 0.6  # fraction of cells to record (randomly selected) only used if cfg.cellsrec = 3

cfg.allpops = cfg.cellParamLabels
cfg.cellsrec = 0
if cfg.cellsrec == 0 or (cfg.cellsrec == 3 and target_me_types is None):  
    cfg.recordCells = cfg.allpops # record all cells
elif cfg.cellsrec == 1 or (cfg.cellsrec == 2 and target_me_types is None): 
    cfg.recordCells = [(pop,0) for pop in cfg.allpops] # record one cell of each pop
elif cfg.cellsrec == 2.1:  # record one cell of only target ME types, alternative way
    cfg.recordCells = [(pop,0) for pop in cfg.allpops
                            if any([target in pop for target in target_me_types])]
elif cfg.cellsrec == 2: 
    cfg.recordCells = []
    for metype in cfg.cellParamLabels:
        if any([target in metype for target in target_me_types]):
            if cfg.cellNumber[metype] < 5:
                for numberME in range(cfg.cellNumber[metype]):
                    cfg.recordCells.append((metype,numberME))
            else:
                numberME = 0
                diference = cfg.cellNumber[metype] - 5.0*int(cfg.cellNumber[metype]/5.0)
                
                for number in range(5):            
                    cfg.recordCells.append((metype,numberME))
                    
                    if number < diference:              
                        numberME+=int(np.ceil(cfg.cellNumber[metype]/5.0))  
                    else:
                        numberME+=int(cfg.cellNumber[metype]/5.0)
elif cfg.cellsrec == 3:  # record all cells of target ME types
    cfg.recordCells = []
    for metype in cfg.cellParamLabels:
        if any([target in metype for target in target_me_types]):
            for numberME in range(cfg.cellNumber[metype]):
                if np.random.rand() < fraction_record:
                    cfg.recordCells.append((metype,numberME))

#cfg.recordTraces = {'V_soma': {'sec':'soma', 'loc':0.5, 'var':'v'}}  ## Dict with traces to record
# record up to axon, dend, and apic 1000
if record_trace_setting['compartment'] == 'soma' and ((record_trace_setting['cell_num_start'] is None) or (record_trace_setting['cell_num_end'] is None)):
    cfg.recordTraces['V' + record_trace_setting['compartment']] = {'sec': record_trace_setting['compartment'],'loc':0.5,
                                                                   'var':'v'
                                                                   #'conds': {'z': [xStimLocation[2]-z_recording_radius, 
                                                                    #               xStimLocation[2]+z_recording_radius]}
                                                                    }
else:
    for i in range(record_trace_setting['cell_num_start'], record_trace_setting['cell_num_end']):
        cfg.recordTraces['V' + record_trace_setting['compartment'] + '_'+str(i)] = {'sec':record_trace_setting['compartment'] + '_'+str(i),'loc':0.5,
                                                                                    'var':'v',
                                                                                    #'conds': {'z': [xStimLocation[2]-z_recording_radius, 
                                                                                    #                xStimLocation[2]+z_recording_radius]}
                                                                                                    }
        #cfg.recordTraces['Vapic_'+str(i)] = {'sec':'apic_'+str(i),'loc':0.5,'var':'v'}
        #cfg.recordTraces['Vaxon_'+str(i)] = {'sec':'axon_'+str(i),'loc':0.5,'var':'v'}

cfg.recordStim = True			
cfg.recordTime = True  		
cfg.recordStep = 0.1            

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------
cfg.simLabel = 'v7_batch1'
cfg.saveFolder = '../data/'+cfg.simLabel
# cfg.filename =                	## Set file output name
cfg.savePickle = True         	## Save pkl file
cfg.saveJson = False	           	## Save json file
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net'] ## , 'simConfig', 'netParams'
cfg.backupCfgFile = None 		##  
cfg.gatherOnlySimData = False	##  
cfg.saveCellSecs = False			
cfg.saveCellConns = False	

#------------------------------------------------------------------------------
# Analysis and plotting 
# ------------------------------------------------------------------------------
#cfg.analysis['plotRaster'] = {'include': cfg.allpops, 'saveFig': True, 'showFig': False, 'orderInverse': True, 'timeRange': [0,cfg.duration], 'figSize': (36,18), 'popRates': False, 'fontSize':12, 'lw': 1, 'markerSize':2, 'marker': '.', 'dpi': 300} 
#cfg.analysis['plot2Dnet']   = {'include': cfg.allpops, 'saveFig': True, 'showConns': False, 'figSize': (24,24), 'fontSize':8}   # Plot 2D cells xy
# cfg.analysis['plotTraces'] = {'include': cfg.recordCells, 'oneFigPer': 'cell', 'overlay': True, 'timeRange': [0,cfg.duration], 'ylim': [-100,50], 'saveFig': True, 'showFig': False, 'figSize':(12,4)}
# cfg.analysis['plot2Dfiring']={'saveFig': True, 'figSize': (24,24), 'fontSize':16}
# cfg.analysis['plotConn'] = {'includePre': cfg.allpops, 'includePost': cfg.allpops, 'feature': 'numConns', 'groupBy': 'pop', 'figSize': (24,24), 'saveFig': True, 'orderBy': 'gid', 'graphType': 'matrix', 'saveData':'../data/v5_batch0/v5_batch0_matrix_numConn.json', 'fontSize': 18}
# cfg.analysis['plotConn'] = {'includePre': ['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], 'includePost': ['L1_DAC_cNA','L23_MC_cAC','L4_SS_cAD','L4_NBC_cNA','L5_TTPC2_cAD', 'L5_LBC_cNA', 'L6_TPC_L4_cAD', 'L6_LBC_cNA', 'ss_RTN_o', 'ss_RTN_m', 'ss_RTN_i', 'VPL_sTC', 'VPM_sTC', 'POm_sTC_s1'], 'feature': 'convergence', 'groupBy': 'pop', 'figSize': (24,24), 'saveFig': True, 'orderBy': 'gid', 'graphType': 'matrix', 'fontSize': 18}
# cfg.analysis['plot2Dnet']   = {'include': ['L5_LBC', 'VPM_sTC', 'POm_sTC_s1'], 'saveFig': True, 'showConns': True, 'figSize': (24,24), 'fontSize':16}   # Plot 2D net cells and connections
#cfg.analysis['plotShape'] = {'includePre': cfg.recordCells, 'includePost': cfg.recordCells, 'showFig': False, 'includeAxon': True, 
#                            'showSyns': False, 'saveFig': True, 'dist': 0.55, 'cvar': 'voltage', 'figSize': (24,12), 'dpi': 600}

#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
cfg.scale = 0.1 # reduce size
cfg.sizeY = 2082.0
cfg.sizeX = 420.0 # r = 210 um and hexagonal side length = 230.9 um
cfg.sizeZ = 420.0
cfg.scaleDensity = 1.0 # Number of cells = 31346


#------------------------------------------------------------------------------
# Spontaneous synapses + background - data from Rat
#------------------------------------------------------------------------------
cfg.addStimSynS1 = False
cfg.rateStimE = 9.0
cfg.rateStimI = 9.0

#------------------------------------------------------------------------------
# Connectivity
#------------------------------------------------------------------------------
## S1->S1
cfg.addConn = True

cfg.synWeightFractionEE = [1.0, 1.0] # E -> E AMPA to NMDA ratio
cfg.synWeightFractionEI = [1.0, 1.0] # E -> I AMPA to NMDA ratio
cfg.synWeightFractionII = [1.0, 1.0]  # I -> I GABAA to GABAB ratio
cfg.synWeightFractionIE = [1.0, 1.0]  # I -> E GABAA to GABAB ratio
cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IIGain = 1.0
cfg.IEGain = 1.0
if cfg.experiment_NBQX_global:
    cfg.synWeightFractionEE = [0.05, 1.0] # E -> E AMPA to NMDA ratio
    cfg.synWeightFractionEI = [0.05, 1.0] # E -> I AMPA to NMDA ratio
    cfg.EEGain = 0.05
    cfg.EIGain = 0.05

#------------------------------------------------------------------------------
## Th->Th 
cfg.connectTh = True
cfg.connect_RTN_RTN     = True
cfg.connect_TC_RTN      = True
cfg.connect_RTN_TC      = True

cfg.yConnFactor             = 10 # y-tolerance form connection distance based on the x and z-plane radial tolerances (1=100%; 2=50%; 5=20%; 10=10%)

cfg.intraThalamicGain = 1.0 

cfg.connWeight_RTN_RTN      = 2.0 # optimized to increase synchrony
cfg.connWeight_TC_RTN       = 1.5 # optimized to increase synchrony
cfg.connWeight_RTN_TC       = 0.35 # optimized to increase synchrony

cfg.connProb_RTN_RTN        = 0.5 #2021-06-23 - test
cfg.connProb_TC_RTN         = 1 #2021-06-23 - test
cfg.connProb_RTN_TC         = 1 #2021-06-23 - test

cfg.divergenceHO = 10

#------------------------------------------------------------------------------
## Th->S1
cfg.connect_Th_S1 = True
cfg.TC_S1 = {}
# Next 3 lines are only used if cfg.connect_Th_S1 = True
cfg.TC_S1['VPL_sTC'] = True  
cfg.TC_S1['VPM_sTC'] = True
cfg.TC_S1['POm_sTC_s1'] = True

cfg.frac_Th_S1 = 1.0
#------------------------------------------------------------------------------
## S1->Th 
cfg.connect_S1_Th = True

cfg.connect_S1_RTN = True
cfg.convergence_S1_RTN         = 30.0  # dist_2D<R
cfg.connWeight_S1_RTN       = 0.500

cfg.connect_S1_TC = True
cfg.convergence_S1_TC         = 30.0  # dist_2D<R
cfg.connWeight_S1_TC       = 0.250

#------------------------------------------------------------------------------
# Current inputs 
#------------------------------------------------------------------------------
cfg.addIClamp = False  # decrease the transient
 
cfg.IClamp = []
cfg.IClampnumber = 0

cfg.thalamocorticalconnections =  ['VPL_sTC', 'VPM_sTC', 'POm_sTC_s1']
for popName in cfg.thalamocorticalconnections:
    cfg.IClamp.append({'pop': popName, 'sec': 'soma', 'loc': 0.5, 'start': 0, 'dur': 5, 'amp': 2.0+10.0*cfg.IClampnumber}) #pA
    cfg.IClampnumber=cfg.IClampnumber+1

#------------------------------------------------------------------------------
# Extracellular stim
#------------------------------------------------------------------------------
cfg.addExtracellularStim = True

cfg.xStimLocation = xStimLocation 
cfg.xStimSigma = 0.276  # conductivity in mS/mm
cfg.xStimAmp = 2  # amplitude in mA
cfg.xStimDur = 4  # duration in ms
cfg.xStimDel = 25  # delay in ms

#------------------------------------------------------------------------------
# NetStim inputs 
#------------------------------------------------------------------------------
cfg.addNetStim=False
if cfg.addNetStim:
    
    cfg.numStims    = 100
    cfg.netWeight   = 0.005
    cfg.startStimTime = 0
    cfg.interStimInterval=0.1

    cfg.NetStim1    = { 'pop':              'VPM_sTC', 
                        'ynorm':            [0,1], 
                        'sec':              'soma', 
                        'loc':              0.5, 
                        'synMech':          ['AMPA_Th'], 
                        'synMechWeightFactor': [1.0],
                        'start':            cfg.startStimTime, 
                        'interval':         cfg.interStimInterval, 
                        'noise':            1, 
                        'number':           cfg.numStims, 
                        'weight':           cfg.netWeight, 
                        'delay':            0}

#------------------------------------------------------------------------------
# Targeted NetStim inputs 
#------------------------------------------------------------------------------
cfg.addTargetedNetStim=False
if cfg.addTargetedNetStim:
    
    cfg.startStimTime=None
    cfg.stimPop = None
    cfg.netWeight           = 20
    # cfg.startStimTime1      = 2000
    cfg.numStims            = 15
    cfg.interStimInterval   = 75 #125#1000/5

    cfg.numOfTargetCells=100

    cfg.TargetedNetStim1= { 
                        'pop':              'VPL_sTC', 
                        # 'pop':              cfg.stimPop, 
                        'ynorm':            [0,1], 
                        'sec':              'soma', 
                        'loc':              0.5, 
                        'synMech':          ['AMPA_Th'], 
                        'synMechWeightFactor': [1.0],
                        'start':            1500, 
                        'interval':         cfg.interStimInterval, 
                        'noise':            1, 
                        'number':           cfg.numStims, 
                        'weight':           cfg.netWeight, 
                        'delay':            0,
                        # 'targetCells':      [0]
                        # 'targetCells':      list(range(0,10,1))
                        'targetCells':      list(range(0,cfg.numOfTargetCells,1))
                        # 'targetCells':      [0,50,500,900]
                        }
