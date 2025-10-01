import numpy as np
from netpyne.batch import Batch
from netpyne import specs
from tune_objective import myObjective


def tune_optuna():
    params = specs.ODict()
    params[('propVelocity')] = [200.0, 400.0]  # range (min, max)
    params[('partial_blockade_fraction')] = [0.0, 1.0]

    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg-tune.py')
    return b

if __name__ == '__main__':
    b = tune_optuna()
    b.batchLabel = 'optuna_tuning'
    b.saveFolder = '../data/' + b.batchLabel
    b.method = 'optuna'
    b.runCfg = {'type': 'mpi_direct',
            'mpiCommand': 'mpiexec -n 8 nrniv -python -mpi init.py', 
            'skip': True}

    # Optuna-specific configs
    b.optimCfg = {
        'max_evals': 40,
        'num_workers': 16,
        'fitnessFunc': myObjective,
        'fitnessFuncArgs': {                # The custom arguments for the fitness function
        },
        'maxiters': 10000,
        'maxtime': 10000,
        'maxiter_wait': 500,
        'time_sleep': 25,

    }

    b.run()
