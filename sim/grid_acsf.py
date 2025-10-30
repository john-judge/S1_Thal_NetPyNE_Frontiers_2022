import numpy as np
from netpyne.batch import Batch
from netpyne import specs
from tune_objective import myObjective
from netpyne import sim

''' 
Running 2 simulations (ACSF and NBQX) per tuning iteration has 
caused issues.
Therefore, in this script we only run the ACSF simulation
for a grid of partial_blockade_fraction and propVelocity values.
We then store the data to a file to be referenced during
NBQX simulation tuning iterations.
'''

def grid():
    params = specs.ODict()
    params[('propVelocity')] = list(np.arange(0.0, 150.0, 10.0))
    params[('partial_blockade_fraction')] = list(np.arange(0.0, 0.05, 0.0025))

    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg-tune.py')
    return b

if __name__ == '__main__':
    b = grid()
    b.batchLabel = 'grid_acsf'
    b.saveFolder = '../data/' + b.batchLabel
    b.method = 'grid'
    b.runCfg = {'type': 'mpi_direct',
            'mpiCommand': 'mpiexec -n 8 nrniv -python -mpi -u init-grid.py', 
            'skip': True}

    b.run()
