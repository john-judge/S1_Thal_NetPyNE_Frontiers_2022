import numpy as np
import gc


class CellRecording:
    """ An object representing a recording of a cell's activity.
    It can be decomposed by compartment or activity type, as well as
    by the intersection of the two
    """
    def __init__(self, 
                 data_dir,
                 cell_id, 
                 time,
                 camera_width,
                 camera_height,
            ):
        self.data_dir = data_dir
        self.cell_id = cell_id
        self.time = time
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.compartments = ['soma', 'axon', 'dend', 'apic']
        self.recordings = {}
        for compart in self.compartments:
            self.recordings[compart] = self.create_compartment_recording(compart)

    def record_activity(self, i, j, weights, t, i_bounds=None, j_bounds=None,
                               compart=None, spike_mask=None):
        """ Record the activity of the cell in the specified compartment and activity type. """
        if compart not in self.compartments:
            raise ValueError("Compartment not found: " + compart)
        if spike_mask is None:
            spike_mask = np.zeros(weights.shape, dtype=bool)

        # if shape of spike mask does not match weights, tile it in axis 1 and 2 to match
        if spike_mask.shape[0] != weights.shape[0]:
            raise ValueError("Spike mask shape not compatible with weights shape: " + \
                             str(spike_mask.shape) + " vs " + str(weights.shape))
        if len(spike_mask.shape) == 1 and len(weights.shape) > 1:
            spike_mask = spike_mask.reshape((spike_mask.shape[0], 1, 1))
        
        if i_bounds is None and j_bounds is None:
            if t is None:
                self.recordings[compart]['synaptic'][:, i, j] += (weights * (1 - spike_mask))
                self.recordings[compart]['spiking'][:, i, j] += (weights * spike_mask)

            else:
                self.recordings[compart]['synaptic'][t, i, j] += (0 if spike_mask else weights)
                self.recordings[compart]['spiking'][t, i, j] += (weights if spike_mask else 0)
        else:
            if t is None:
                self.recordings[compart]['synaptic'][:, i_bounds[0]:i_bounds[1], 
                                         j_bounds[0]:j_bounds[1]] += (weights * (1 - spike_mask))
                self.recordings[compart]['spiking'][:, i_bounds[0]:i_bounds[1],
                                         j_bounds[0]:j_bounds[1]] += (weights * spike_mask)
            else:
                self.recordings[compart]['synaptic'][t, i_bounds[0]:i_bounds[1], 
                                         j_bounds[0]:j_bounds[1]] += (weights * (1 - spike_mask))
                self.recordings[compart]['spiking'][t, i_bounds[0]:i_bounds[1],
                                            j_bounds[0]:j_bounds[1]] += (weights * spike_mask)
        
    def get_raw_recording(self, compart_id=None, activity_type=None):
        if compart_id is None:
            return self.recordings
        if compart_id not in self.compartments:
            raise ValueError("Compartment not found: " + compart_id)
        if activity_type is None:
            return self.recordings[compart_id]
        if activity_type not in ['spiking', 'synaptic']:
            raise ValueError("Activity type not found: " + activity_type)
        return self.recordings[compart_id][activity_type]
    
    def get_combined_recording(self, compart_id=None, activity_type=None):
        """ Get the combined recording of the cell. """
        if compart_id is None and activity_type is None:
            return sum([self.recordings[compart] for compart in self.compartments])
        if compart_id is None:
            return sum([self.recordings[compart][activity_type] for compart in self.compartments])
        if compart_id not in self.compartments:
            raise ValueError("Compartment not found: " + compart_id)
        if activity_type is None:
            return self.recordings[compart_id]
        if activity_type not in ['spiking', 'synaptic']:
            raise ValueError("Activity type not found: " + activity_type)
        return self.recordings[compart_id][activity_type]

    def create_compartment_recording(self, compart_id):
        """ Create a compartment recording for the cell. """
        spk_mm_fp = self.get_mmap_filename(compart_id, file_keyword='spk_rec')
        syn_mm_fp = self.get_mmap_filename(compart_id, file_keyword='syn_rec')
        return {'spiking': np.memmap(spk_mm_fp, dtype='float32', mode='w+', shape=(len(self.time), self.camera_width, self.camera_height)),
                'synaptic': np.memmap(syn_mm_fp, dtype='float32', mode='w+', shape=(len(self.time), self.camera_width, self.camera_height))}
    
    def get_combined_compartment_recording(self, compart_rec):
        return compart_rec['spiking'] + compart_rec['synaptic']

    def get_mmap_filename(self, decomp_type=None, file_keyword='model_rec'):
        """ Get the filename of the memory-mapped numpy array. """
        if decomp_type is None:
            return self.data_dir + str(self.cell_id) + '-' + file_keyword + '.npy'
        return self.data_dir + str(self.cell_id) + '-' + file_keyword + '_' + decomp_type + '.npy'
        
    def flush_memmaps(self):
        """ Flush the memory-mapped numpy arrays to disk. """
        for compart in self.compartments:
            self.recordings[compart]['spiking'].flush()
            self.recordings[compart]['synaptic'].flush()
        gc.collect()

    def close_memmaps(self):
        """ Close the memory-mapped numpy arrays. """
        for compart in self.compartments:
            del self.recordings[compart]['spiking']
            del self.recordings[compart]['synaptic']
        gc.collect()