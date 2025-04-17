import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import subprocess
import sys
import gc
import pickle

from src.hVOS.cell import Cell
from src.hVOS.compartment_file import MemoryMappedCompartmentVoltages


#####################################
# extract all the .tar.gz files
# #####################################
run_id = 2
data_dir = '../'
compart_data = {}
should_create_mem_map = True  # if True, create mem mapped files. If False, load mem mapped files
target_dir = data_dir + 'run' + str(run_id) + '/'
output_dir_final = data_dir + 'analyze_output/'
if not os.path.exists(output_dir_final):
    os.makedirs(output_dir_final)

for file in os.listdir(target_dir):
    compart = file.replace('.tar.gz', "").replace('S1-Thal-output-',"")
    output_dir = target_dir + compart + '/'
    
    if file.endswith('.tar.gz'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            result = subprocess.run(['tar', '-xzvf', target_dir + file, "-C", output_dir],
                                     capture_output=True, text=True, check=True)
            
            print('Extracted ' + file)
        # else it already exists and was extracted
        #os.rename(data_dir + file, data_dir + 'archive/' + file)
        compart_data[compart] = output_dir
print("compart_data", compart_data)
#######################################
# load time 
#######################################
data_file = compart_data['soma'] + '/S1_Thal_NetPyNE_Frontiers_2022/data/v7_batch1/v7_batch1_0_0_data.pkl'
with open(data_file, 'rb') as f:
    try:
        data = pickle.load(f)
    except MemoryError:
        print("MemoryError with file:", data_file)
        raise MemoryError
    print(data.keys())
    t = np.array(data['simData']['t'])
len(t)

#########################################
# load one pkl to look at cell data
#########################################
cell_id_to_me_type_map = {}

me_type_map_file = output_dir_final + 'cell_id_to_me_type_map.pkl'

if os.path.exists(me_type_map_file):
    with open(me_type_map_file, 'rb') as f:
        cell_id_to_me_type_map = pickle.load(f)
else:

    compart = list(compart_data.keys())[0]
    target_dir_net = compart_data[compart] + 'S1_Thal_NetPyNE_Frontiers_2022/data/v7_batch1/v7_batch1_0_0_data.pkl'

    with open(target_dir_net, 'rb') as f:
        data = pickle.load(f)
        print(data.keys())
        for cell_dict in data['net']['cells']:
            cell_id_to_me_type_map[cell_dict['gid']] = {
                'me_type': cell_dict['tags']['cellType'],
                'x': cell_dict['tags']['x'],
                'y': cell_dict['tags']['y'],
                'z': cell_dict['tags']['z']
            }

    with open(me_type_map_file, 'wb') as f:
        pickle.dump(cell_id_to_me_type_map, f)
    print("wrote", me_type_map_file)

##########################################
# memory map all the files needed (prepped for optical model)
##########################################
loaded_compart_data = MemoryMappedCompartmentVoltages(output_dir_final)
time = None
if should_create_mem_map:
    print("Creating mem map at", loaded_compart_data.mmap_filename)
    for compart in compart_data.keys():
        print(compart)
        target_dir = compart_data[compart]
        if not os.path.exists(target_dir):
            print('Directory ' + target_dir + ' does not exist')
            continue
        target_dir += '/S1_Thal_NetPyNE_Frontiers_2022/data/v7_batch1/'
        data_file = target_dir + 'v7_batch1_0_0_data.pkl'

        if not os.path.exists(data_file):
            print("Data file not found:", data_file)
            continue

        # Load data
        with open(data_file, 'rb') as f:
            try:
                data = pickle.load(f)
            except MemoryError:
                print("MemoryError with file:", data_file)
                raise MemoryError
            if time is None:  # store t only once
                time = np.array(data['simData']['t'])

            for k in data['simData']:
                if compart[:4] in k:
                    for cell_id in data['simData'][k]:
                        loaded_compart_data.add_item(cell_id, k, np.array(data['simData'][k][cell_id]))

            # to avoid memory issues, delete data after it's been stored
            del data
            gc.collect()

    # also store time to memory mapped file
    if time is not None:
        mm_time_fp = output_dir_final + 'v7_batch1_0_0_time.dat'
        if not os.path.exists(mm_time_fp):
            with open(mm_time_fp, 'wb') as f:
                f.write(b'\0' * time.nbytes)
        mm_time_fp = np.memmap(mm_time_fp, dtype='float32', mode='w+', shape=time.shape)
        mm_time_fp[:] = time[:]
        mm_time_fp.flush()

loaded_compart_data.dump_hash_map(output_dir_final + 'v7_batch1_0_0_hash_map.pkl')

for cell_id in loaded_compart_data.hash_map:
    for comp in loaded_compart_data.hash_map[cell_id]:
        i_data, mmfp = loaded_compart_data.get_item(cell_id, comp)
        print(' check get in test_compfile', mmfp[i_data])
        break
    break
print("Total nonzero:", np.sum(loaded_compart_data.mmap_fp != 0))

loaded_compart_data.mmap_fp.flush()
del loaded_compart_data.mmap_fp


##########################################
# try re-opening the memory mapped file to test
print('time shape', time.shape)
test_compfile = MemoryMappedCompartmentVoltages(output_dir_final)
test_compfile.load_existing_mmap(output_dir_final + 'v7_batch1_0_0_hash_map.pkl',
                                        loaded_compart_data.mmap_filename, (-1, time.shape[0]))

# check get in loaded_compart_data
for cell_id in test_compfile.hash_map:
    for comp in test_compfile.hash_map[cell_id]:
        i_data, mmfp = test_compfile.get_item(cell_id, comp)
        print(' check get in test_compfile', mmfp[i_data])
        break
    break
print("Total nonzero:", np.sum(test_compfile.mmap_fp != 0))
                                    

