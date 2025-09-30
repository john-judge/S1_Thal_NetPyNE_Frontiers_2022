from netpyne.batch import Batch
import numpy as np


def tune_optuna():
    from netpyne import specs
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
        'objective': 'myObjective'  # function defined in your cfg.py
    }

    b.run()
