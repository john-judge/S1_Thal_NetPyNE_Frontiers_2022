import numpy as np
import matplotlib.pyplot as plt
from src.hVOS.optical_readout import OpticalReadout


class hVOSReadout (OpticalReadout):
    def __init__(self, target_populations, cells, morphologies, force_overwrite=False):
        super().__init__(target_populations, cells, morphologies, force_overwrite=force_overwrite)

    def _calculate_intensity_trace(self, voltage_trace):
        # Wang et al 2010: hVOS 2.0 in cultured PC12 cells
        #    
        #          for Vm in the range of -70 to +30 mV
        # assume that for Vm < -70 mV, df/f = 0.056  + 0.0008 * Vm
        # assume that for Vm > +30 mV, df/f = 0.196 + 0.00286 * Vm 
        # the intensity trace is the intensity of the hVOS normalized to the baseline intensity, f

        intensity_trace = np.ones(len(voltage_trace))
        voltage_trace = np.array(voltage_trace)
        intensity_trace[voltage_trace < -70] =  0.04592  + 0.000716 * voltage_trace[voltage_trace < -70]
        intensity_trace[voltage_trace >= -70] = 0.196 + 0.00286 * voltage_trace[voltage_trace >= -70]
        '''for i, voltage in enumerate(voltage_trace):
            if voltage < -70:
                intensity_trace[i] += 0.056  + 0.0008 * voltage
            elif voltage >= -70:
                intensity_trace[i] += 0.196 + 0.00286 * voltage'''
        return intensity_trace
    
    def show_voltage_to_intensity_curve(self):
        # voltage to df/f conversion for hVOS
        x2 = np.linspace(-70, 30, 100)
        x1 = np.linspace(-180, -70, 100)
        y1 =  0.04592 + 0.000716 * x1
        y2 = 0.196 + 0.00286 * x2
        plt.plot(x1, y1, label='<-70mV')
        plt.plot(x2, y2, label='>-70mV')
        plt.legend()
        plt.xlabel('Membrane Potential (mV)')
        plt.ylabel('df/f')
        #plt.xlim(-160, 30)
        #plt.ylim(-.4, 0.5)
        plt.show()
    
