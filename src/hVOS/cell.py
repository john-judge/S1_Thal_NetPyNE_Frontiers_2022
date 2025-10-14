import numpy as np


class Cell:
    def __init__(self, cell_id, me_type, axons, apics, 
                 dends, soma, x, y, z, optical_filelabel='optical'):
        self.cell_id = cell_id
        self.me_type = me_type
        self.x = x
        self.y = y
        self.z = z

        # dicts to store filemaps for voltage traces (mmaps)
        self.axons = axons
        self.apics = apics
        self.dends = dends
        self.soma = {'Vsoma': soma}
        
        # dicts to store filemaps for optical traces (mmaps)
        self.axon_optical = {}
        self.apic_optical = {}
        self.dend_optical = {}
        self.soma_optical = {}

        self.morphology = None  # a Morphology object

    def get_soma_position(self):
        return self.x, self.y, self.z

    def set_morphology(self, morphology):
        self.morphology = morphology

    def get_morphology(self):
        return self.morphology
    
    def get_cell_id(self):
        return self.cell_id

    def __repr__(self):
        return f"Cell(id={self.cell_id}, me_type={self.me_type}, position={self.x}, {self.y}, {self.z})"

    def load_data(self, data_pointer):
        if type(data_pointer) is np.ndarray:
            return data_pointer
        i_data, mmap_fp = data_pointer
        return mmap_fp[i_data, :]
    
    def load_optical_data(self, mm_file):
        return np.memmap(mm_file, dtype='float32', mode='r')
    
    def write_data(self, mm_file, data):
        fp = np.memmap(mm_file, dtype='float32', mode='w+', shape=data.shape)
        fp[:] = data[:]
    
    def get_list_compartment_ids(self):
        return list(self.axons.keys()) + \
               list(self.apics.keys()) + \
               list(self.dends.keys()) + \
               list(self.soma.keys())

    def get_voltage_trace(self, compart_id):
        if 'axon' in compart_id:
            return self.load_data(self.axons[compart_id])
        if 'apic' in compart_id:
            return self.load_data(self.apics[compart_id])
        if 'dend' in compart_id:
            return self.load_data(self.dends[compart_id])
        if 'soma' in compart_id:
            return self.load_data(self.soma[compart_id])
        
    def get_optical_trace(self, compart_id):
        if 'soma' in compart_id:
            return self.load_optical_data(self.soma_optical['Vsoma'])
        compart_dict = {'axon': self.axon_optical, 
                        'apic': self.apic_optical, 
                        'dend': self.dend_optical}
        for key in compart_dict.keys():
            if key in compart_id:
                try:
                    return self.load_optical_data(compart_dict[key][compart_id])
                except KeyError:
                    print(f"Optical trace not found for {compart_id}")
                    print(f"Available optical traces:", compart_dict[key].keys())

    def get_axon_filemap(self, compart_id=None):
        if compart_id is None:
            return self.axons
        return self.axons[compart_id]
    
    def get_apic_filemap(self, compart_id=None):
        if compart_id is None:
            return self.apics
        return self.apics[compart_id]
    
    def get_dend_filemap(self, compart_id=None):
        if compart_id is None:
            return self.dends
        return self.dends[compart_id]
    
    def get_soma_filemap(self):
        return self.soma['Vsoma']
    
    def get_soma_voltage_trace(self):
        return self.load_data(self.soma['Vsoma'])
    
    def get_me_type(self):
        return self.me_type
    
    def generate_mm_filename(self, compart_id, target_dir):
        return target_dir + f"optical_{self.cell_id}_{compart_id}.mm"
    
    def set_optical_trace(self, compart_id, trace, target_dir):
        """ Create a memmap for the optical trace if it doesn't exist
            The memmap filename is stored in the appropriate dict
            If the memmap exists, overwrite it with the new trace """
        
        compart_dict = {'axon': self.axon_optical, 
                        'apic': self.apic_optical, 
                        'dend': self.dend_optical, 
                        'soma': self.soma_optical}
        for key in compart_dict.keys():
            if key in compart_id:
                if compart_id not in compart_dict[key]:
                    mm_file_name = self.generate_mm_filename(compart_id, target_dir)
                    self.set_optical_trace_filename(compart_id, mm_file_name)
                self.write_data(compart_dict[key][compart_id], trace)        
                return True
        print("Could not set optical trace for", compart_id)
        return False
    
    def set_optical_trace_filename(self, compart_id, mm_file_name):
        """ Set the memmap filename for the optical trace """
        compart_dict = {'axon': self.axon_optical, 
                        'apic': self.apic_optical, 
                        'dend': self.dend_optical, 
                        'soma': self.soma_optical}
        for key in compart_dict.keys():
            if key in compart_id:
                compart_dict[key][compart_id] = mm_file_name
                return True
        print("Could not set optical trace filename for", compart_id)
        return False
    
    def get_optical_trace_filename(self, compart_id):
        compart_dict = {'axon': self.axon_optical, 
                        'apic': self.apic_optical, 
                        'dend': self.dend_optical, 
                        'soma': self.soma_optical}
        for key in compart_dict.keys():
            if key in compart_id:
                return compart_dict[key][compart_id]
        return None