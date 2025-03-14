import numpy as np
from src.hVOS.optical_readout import OpticalReadout


class VSDReadout (OpticalReadout):
    def __init__(self, target_populations, cells, morphologies):
        super().__init__(target_populations, cells, morphologies)

    def _calculate_intensity_trace(self, voltage_trace):
        # VSD logic based on Newton et al 2013
        # the intensity trace is the intensity of the hVOS normalized to the baseline intensity, f

        intensity_trace = np.ones(len(voltage_trace))
        raise NotImplementedError("Implement VSD logic")
        return intensity_trace
