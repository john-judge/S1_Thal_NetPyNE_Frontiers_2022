import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import subprocess
import sys
import gc

# pickle
import pickle

from src.hVOS.cell import Cell


#####################################
# extract all the .tar.gz files
# #####################################
run_id = 2
data_dir = '../'
compart_data = {}
should_create_mem_map = True  # if True, create mem mapped files. If False, load mem mapped files
target_dir = data_dir + 'run' + str(run_id) + '/'

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

me_type_map_file = target_dir + 'cell_id_to_me_type_map.pkl'

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

##########################################
# memory map all the files needed (prepped for optical model)
##########################################
loaded_compart_data = {}
time = None
if should_create_mem_map:
    for compart in compart_data.keys():
        print(compart)
        if compart in loaded_compart_data:
            continue
        target_dir = compart_data[compart]
        if not os.path.exists(target_dir):
            print('Directory ' + target_dir + ' does not exist')
            continue
        target_dir += '/S1_Thal_NetPyNE_Frontiers_2022/data/v7_batch1/'
        data_file = target_dir + 'v7_batch1_0_0_data.pkl'

        if not os.path.exists(data_file):
            print("Data file not found:", data_file)
            continue
        # also skip if the .dat files already exist
        mmaps_exist = False
        for file in os.listdir(target_dir):
            if file.endswith('.dat') and 'v7_batch1_0_0' in file and 'cell' in file:
                print("\t", target_dir + file)
                mmaps_exist = True
                break
        if mmaps_exist:
            print("Memory mapped files already exist for compartment", compart)
            continue

        # Load data
        with open(data_file, 'rb') as f:
            try:
                data = pickle.load(f)
            except MemoryError:
                print("MemoryError with file:", data_file)
                raise MemoryError
            print(data.keys())
            if time is None:  # store t only once
                time = np.array(data['simData']['t'])

            for k in data['simData']:
                if compart[:4] in k:
                    for cell_id in data['simData'][k]:
                        print(k, cell_id)

                        # create memory mapped file for this cell's segment data
                        mm_data_fp = target_dir + 'v7_batch1_0_0_' + k + '_' + cell_id + '.dat'
                        cell_data = np.array(data['simData'][k][cell_id])

                        # create empty file if it doesn't exist
                        if not os.path.exists(mm_data_fp):
                            with open(mm_data_fp, 'wb') as f:
                                f.write(b'\0' * cell_data.nbytes)

                        # store as a memory mapped array for faster access / less memory usage
                        mm_data_fp = np.memmap(mm_data_fp, dtype='float32', mode='w+', shape=cell_data.shape)
                        mm_data_fp[:] = cell_data[:]
                        mm_data_fp.flush()
                        if compart not in loaded_compart_data:
                            loaded_compart_data[compart] = {}
                        if cell_id not in loaded_compart_data[compart]:
                            loaded_compart_data[compart][cell_id] = {}
                        loaded_compart_data[compart][cell_id] = mm_data_fp  # store just pointer
            # to avoid memory issues, delete data after it's been stored
            del data
            gc.collect()

    # also store time to memory mapped file
    if time is not None:
        target_dir = data_dir + 'archive/run' + str(run_id) + '/'
        mm_time_fp = target_dir + 'v7_batch1_0_0_time.dat'
        if not os.path.exists(mm_time_fp):
            with open(mm_time_fp, 'wb') as f:
                f.write(b'\0' * time.nbytes)
        mm_time_fp = np.memmap(mm_time_fp, dtype='float32', mode='w+', shape=time.shape)
        mm_time_fp[:] = time[:]
        mm_time_fp.flush()
        loaded_compart_data['time'] = mm_time_fp  # store just pointer
else: # load mem mapped files. Get file pointers from looking through data_dir dat files and load them into loaded_compart_data
    for compart in compart_data.keys(): 
        target_dir = data_dir + 'archive/run' + str(run_id) + '/' + compart
        if not os.path.exists(target_dir):
            print('Directory ' + target_dir + ' does not exist')
            continue
        target_dir += '/S1_Thal_NetPyNE_Frontiers_2022/data/v7_batch1/'
        for file in os.listdir(target_dir):
            if file.endswith('.dat') and 'v7_batch1_0_0' in file and 'cell' in file:
                file_name = file.replace(".dat", "").replace("v7_batch1_0_0_", "").split('_')
                compart = "_".join(file_name[:2])
                cell_id = "_".join(file_name[2:])
                if 'soma' in file:
                    compart = 'Vsoma'
                    cell_id = 'cell_' + cell_id
                    
                if compart not in loaded_compart_data:
                    loaded_compart_data[compart] = {}
                if cell_id not in loaded_compart_data[compart]:
                    loaded_compart_data[compart][cell_id] = {}
                loaded_compart_data[compart][cell_id] = target_dir + file
    # load time
    target_dir = data_dir + 'archive/run' + str(run_id) + '/'
    mm_time_fp = target_dir + 'v7_batch1_0_0_time.dat'
    loaded_compart_data['time'] = np.memmap(mm_time_fp, dtype='float32', mode='r')
