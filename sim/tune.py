import numpy as np
from netpyne.batch import Batch
from netpyne import specs
from tune_objective import myObjective
from netpyne import sim


def tune_optuna():
    params = specs.ODict()
    params[('propVelocity')] = [0.0, 200.0]  # range (min, max)
    params[('partial_blockade_fraction')] = [0.0, 0.05]

    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg-tune.py')
    return b

if __name__ == '__main__':
    b = tune_optuna()
    b.batchLabel = 'optuna_tuning'
    b.saveFolder = '../data/' + b.batchLabel
    b.method = 'optuna'
    b.runCfg = {'type': 'mpi_direct',
            'mpiCommand': 'mpiexec -n 8 nrniv -python -mpi -u init-tune.py', 
            'skip': True}

    # Optuna-specific configs
    b.optimCfg = {
        'max_evals': 40,
        'num_workers': 1,
        'fitnessFunc': myObjective,
        'fitnessFuncArgs': {    
        },
        'maxiters': 50,  # number of generations, passed to optuna's n_trials
        'maxtime': None,  # gets passed to optuna's timeout. None: no limit
        'maxiter_wait': 500,  # number of iterations (progress checks)
        'time_sleep': 120,  # seconds per iteration
        'maxFitness': 999999999,  # a large value approximating "infinity"
        #'directions': ['minimize', 'minimize', 'minimize'],  # 'minimize' or 'maximize' the fitness function

    }

    b.run()
