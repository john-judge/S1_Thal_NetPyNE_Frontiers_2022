import pickle
import numpy as np


class MemoryMappedCompartmentVoltages:
    """ A data structure mapping Cell id and compartment id to a 
    row of a 4-D memory mapped array
    """

    def __init__(self, data_dir):
        self.hash_map = {}
        self.mmap_filename = data_dir + 'S1_results.npy'
        self.mmap_fp = None
        self.shape = None
        self.fp_size_init = 10000

    def load_existing_mmap(self, hash_map_filename, mmap_filename):
        self.hash_map = pickle.load(open(hash_map_filename, 'rb'))
        self.mmap_fp = np.memmap(mmap_filename, dtype='float32', mode='r+')
        self.shape = self.mmap_fp.shape
        self.mmap_filename = mmap_filename

    def init_mmap(self, arr_shape):
        self.shape = (self.fp_size_init,) + arr_shape
        self.mmap_fp = np.require(np.memmap(self.mmap_filename, dtype='float32', mode='w+', shape=self.shape), requirements=['O'])

    def dump_hash_map(self, hash_map_filename):
        with open(hash_map_filename, 'wb') as f:
            pickle.dump(self.hash_map, f)

    def resize_mmap(self, arr_shape):
        self.shape = (self.shape[0] * 2,) + arr_shape
        self.mmap_fp.resize(self.shape)

    def add_item(self, cell_id, compart_id, data):
        if self.mmap_fp is None:
            self.init_mmap(data.shape)
        i_data = len(self.hash_map.keys())
        if cell_id not in self.hash_map:
            self.hash_map[cell_id] = {}
        self.hash_map[cell_id][compart_id] = i_data
        
        if i_data >= self.shape[0]:
            # resize the memory mapped file
            self.shape = (self.shape[0] * 2,) + data.shape
            self.mmap_fp.resize(self.shape)
        self.mmap_fp[i_data] = data
        self.mmap_fp.flush()

    def get_item(self, cell_id, compart_id):
        if cell_id not in self.hash_map:
            print(f"Cell ID {cell_id} not found.")
            return None
        i_data = self.hash_map[cell_id][compart_id]
        if type(i_data) != int:
            raise KeyError(f"Compartment ID {compart_id} not found for Cell ID {cell_id}.")
        return self.mmap_fp[i_data]