import numpy as np
from netpyne.batch import Batch
from netpyne import specs
from tune_objective import myObjective
from netpyne import sim
import sys

''' 
Running 2 simulations (ACSF and NBQX) per tuning iteration has 
caused issues.
Therefore, in this script we only run the ACSF simulation
for a grid of partial_blockade_fraction and propVelocity values.
We then store the data to a file to be referenced during
NBQX simulation tuning iterations.
'''



def grid(job_id, n_jobs):
    params = specs.ODict()
    allPropVelocities = [0.001, 0.1, 0.5, 1] + \
                                    list(np.arange(2.0, 10.0, 2.0)) + \
                                    list(np.arange(10.0, 40.0, 3.0))  + \
                                    list(np.arange(40.0, 151.0, 5.0))
    # split allPropVelocities into n_jobs (10) approximately equal parts
    params[('propVelocity')] = np.array_split(allPropVelocities, n_jobs)[job_id]

    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg-tune-nbqx.py')
    return b

if __name__ == '__main__':
    n_jobs = 10
    if len(sys.argv) > 1:
        if sys.argv[1] == '':
            job_id = 0
        else:
            job_id = int(sys.argv[1])
    job_id %= n_jobs  # make sure job_id is in range 0 to n_jobs-1

    b = grid(job_id, n_jobs)
    b.batchLabel = 'grid_nbqx'
    b.saveFolder = '../data/' + b.batchLabel
    b.method = 'grid'
    b.runCfg = {'type': 'mpi_direct',
            'mpiCommand': 'mpiexec -n 1 nrniv -python -mpi -u init-grid-nbqx.py', 
            'skip': True}

    b.run()
