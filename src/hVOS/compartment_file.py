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

    def load_existing_mmap(self, hash_map_filename, mmap_filename, shape=(10000,)):
        self.hash_map = pickle.load(open(hash_map_filename, 'rb'))
        self.mmap_fp = np.memmap(mmap_filename, dtype='float32', mode='r')
        self.mmap_fp = self.mmap_fp.reshape(shape)
        self.shape = self.mmap_fp.shape
        self.mmap_filename = mmap_filename

    def init_mmap(self, arr_shape):
        print("Initializing memory mapped file with shape", (self.fp_size_init,) + arr_shape)
        self.shape = (self.fp_size_init,) + arr_shape
        self.mmap_fp = np.require(np.memmap(self.mmap_filename, dtype='float32', mode='w+', shape=self.shape), requirements=['O'])

    def dump_hash_map(self, hash_map_filename):
        with open(hash_map_filename, 'wb') as f:
            pickle.dump(self.hash_map, f)

    def save(self): 
        """ Due to requirements 'OWNDATA' flag set to true, saving must be done manually. 
            Defeats the purpose of memmaps on the writing portion, but this will not break
            if numpy ever fixes that issue. """
        np.save(self.mmap_filename, self.mmap_fp, allow_pickle=False)

    def resize_mmap(self, arr_shape):
        print("Resizing memory mapped file to shape", (self.shape[0] * 2,) + arr_shape)
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
        self.mmap_fp[i_data, :] = data[:]
        self.mmap_fp.flush()

    def get_item(self, cell_id, compart_id):
        if cell_id not in self.hash_map:
            print(f"Cell ID {cell_id} not found.")
            return None
        i_data = self.hash_map[cell_id][compart_id]
        if type(i_data) != int:
            print(f"Compartment ID {compart_id} not found for Cell ID {cell_id}: {i_data}")
            return None
        return (i_data, self.mmap_fp)