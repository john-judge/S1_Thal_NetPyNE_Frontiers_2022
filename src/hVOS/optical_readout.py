import numpy as np
import matplotlib.pyplot as plt
import gc
import os


class OpticalReadout:
    """ Base class containing the generic logic for computing 
        optical signals from voltage traces. Optical signals are stored 
        to the Cell object. """
    def __init__(self, target_populations, cells, morphologies, force_overwrite=False):
        '''
        cells dict maps cell_id to Cell object
        me_type_morphology_map maps me_type to list of Morphology objects
        target populations is a list of me_types to compute signal for
        '''
        self.target_populations = target_populations
        self.cells = cells
        self.morphologies = morphologies
        self.force_overwrite = force_overwrite

    def compute_optical_signal(self, target_dir):
        """ Compute the optical signal for each cell in the network 
            for the target populations. Store the optical signal in the
            Cell object. 
        """
        for cell_id in self.cells:
            cell = self.cells[cell_id] 
            if any([t_pop in cell.get_me_type() for t_pop in self.target_populations]):
                print(f'Computing optical signal for {cell_id}')
                compart_ids = cell.get_list_compartment_ids()
                for compart_id in compart_ids:
                    # first see if optical trace file exists
                    # if it does, skip computation
                    if not self.force_overwrite:
                        optical_trace_file = cell.generate_mm_filename(compart_id, target_dir)
                        if os.path.exists(optical_trace_file):
                            cell.set_optical_trace_filename(compart_id, optical_trace_file)
                            continue

                    voltage_trace = cell.get_voltage_trace(compart_id)
                    intensity_trace, spike_mask = self._calculate_intensity_trace(voltage_trace)
                    cell.set_optical_trace(compart_id, intensity_trace, target_dir)
                    cell.set_spike_mask(compart_id, spike_mask, target_dir)
                    del voltage_trace, intensity_trace
                print("\tFinished computing optical signal for", 
                        len(compart_ids), 
                        "compartments" )
                gc.collect()

    def plot_optical_signal(self, cell_id):
        """ Plot the optical signal for each compartment in the cell. """
        cell = self.cells[cell_id]
        compart_ids = cell.get_list_compartment_ids()
        # make two subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        plt.figure()
        for compart_id in compart_ids:
            voltage_trace = cell.get_voltage_trace(compart_id)
            intensity_trace = cell.get_optical_trace(compart_id)
            plt.plot(voltage_trace, label='Voltage')
            plt.plot(intensity_trace, label='Intensity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Intensity')
        plt.title(f'Voltage & Optical Signals for {cell_id}')
        plt.legend()
        plt.show()

