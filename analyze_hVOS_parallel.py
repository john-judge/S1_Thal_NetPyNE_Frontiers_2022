""" Analogous to analyze_hVOS.ipynb but for parallel processing on CHTC.
    Accepts a job_id from command line arg to decide how to divide up task.
    Each task is just a different cell in the network.
    The output is:
        optical output memmap files for each cell (broken down by compartment and activity type).
        animate gifs for each cell (broken down by compartment and activity type).
        voltage and hVOS intensity traces for each cell 
"""

import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
import random
import gc

# pickle
import pickle

from src.hVOS.cell import Cell
from src.hVOS.morphology import Morphology
from src.hVOS.hvos_readout import hVOSReadout
from src.hVOS.vsd_readout import VSDReadout
from src.hVOS.camera import Camera
from src.hVOS.psf import PSF
from src.hVOS.mcPSF import mcPSF

####################################
# read command line args
#####################################
if len(sys.argv) != 2:
    raise ValueError("Please provide the job_id as a command line argument.")
job_id = int(sys.argv[1])
if job_id < 0:
    raise ValueError("Job ID must be a positive integer.")

#####################################
# Find data in CHTC staging and extract data just for this job's cell
#####################################
target_hVOS_populations = ["L4_SS", "L4_PC"]
target_sparsity = 0.6
optical_type = "hVOS"

# 'S1_results/' contains a subdirectory 'cell_dat' 
#   which contains a memmap numpy file for each cell in the network
#   each memmap file contains the voltage trace for each compartment of the cell
#   in the format 'v7_batch_1_0_0_V<compartment_id>_<cell_id>.dat'
data_dir = 'S1_results/'
morphology_data_dir = 'NMC_model/NMC.NeuronML2/'
model_rec_out_dir = 'S1_results/model_rec/'

# list subdirectories of S1_results/ 
#   each subdirectory is a range of compart_ids
#   e.g. 'apic_0_10', 'dend_10_20', etc.
# compart_data dict maps compart_id range to subdirectory name
compart_data = {subdir: filepath for filepath, subdir in 
                zip(os.listdir(data_dir), os.listdir(data_dir)) 
                if os.path.isdir(os.path.join(data_dir, subdir) 
                                 and 'model_rec' not in subdir)}

# create a dict that maps compart_id 'Vcomp_#' to
# dicts, which each map cell_id 'cell_#' to the 
# cell's compartment data (the memmap file name) for that compart_id
# also, loaded_compart_data['time'] points to a loaded mmap pointer
loaded_compart_data = {}  
for compart in compart_data.keys(): 
    target_dir =  compart_data[compart]
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
mm_time_fp = data_dir + 'v7_batch1_0_0_time.dat'
loaded_compart_data['time'] = np.memmap(mm_time_fp, dtype='float32', mode='r')

# load cell_id to me_type map
me_type_map_file = data_dir + 'cell_id_to_me_type_map.pkl'
loaded_compart_data = {}
if os.path.exists(me_type_map_file):
    with open(me_type_map_file, 'rb') as f:
        cell_id_to_me_type_map = pickle.load(f)

#######################################
# for each soma, get a cell id and aggregate its axons, apics, dends
#######################################
cells = {}
me_type_morphology_map = {}
for cell_id in loaded_compart_data['Vsoma']:
    soma = loaded_compart_data['Vsoma'][cell_id]
    axons = {}
    apics = {}
    dends = {}
    for compart in loaded_compart_data.keys():
        if compart == 'time':
            continue
        if cell_id in loaded_compart_data[compart]:
            if 'axon' in compart:
                axons[compart] = loaded_compart_data[compart][cell_id]
            elif 'apic' in compart:
                apics[compart] = loaded_compart_data[compart][cell_id]
            elif 'dend' in compart:
                dends[compart] = loaded_compart_data[compart][cell_id]

    short_cell_id = int(cell_id.replace('cell_', ''))
    me_type = cell_id_to_me_type_map[short_cell_id]['me_type']
    x = cell_id_to_me_type_map[short_cell_id]['x']
    y = cell_id_to_me_type_map[short_cell_id]['y']
    z = cell_id_to_me_type_map[short_cell_id]['z']
    cells[cell_id] = Cell(cell_id, me_type, axons, apics, dends, soma, x, y, z, optical_filelabel=optical_type)

    if me_type not in me_type_morphology_map:
        # load morphology
        # find files in morphology_data_dir with me_type in the name
        m_type, e_type = me_type[:6], me_type[7:]
        me_type_files = [f for f in os.listdir(morphology_data_dir) if (m_type in f and e_type in f and f.endswith('.cell.nml'))]
        if len(me_type_files) == 0:
            assert me_type not in target_hVOS_populations  # we only care if we cannot load a target population
        
        # load all morphology file matches
        me_type_morphology_map[me_type] = [Morphology(me_type, morphology_data_dir + me_type_file) for me_type_file in me_type_files]

# cells dict maps cell_id to Cell object
# me_type_morphology_map maps me_type to list of Morphology objects
# target populations is a list of me_types to compute signal for
target_population_cells = [
    cells[cell_id] for cell_id in cells 
        if any([t_pop in cells[cell_id].get_me_type() for 
                    t_pop in target_hVOS_populations ]) 
]
i_target_cell = job_id % len(target_population_cells)  # index of cell to process for this job
 # sparsity sampling implemented here
target_population_cells = [cell for cell in target_population_cells
                                if random.random() < target_sparsity]
optical_readout = {'hVOS': hVOSReadout, 'VSD': VSDReadout}[optical_type]
hvos_readout = optical_readout(target_hVOS_populations, 
                               cells, 
                               me_type_morphology_map,
                               force_overwrite=False)
hvos_readout.compute_optical_signal(data_dir)
#hvos_readout.show_voltage_to_intensity_curve()


####################################
# Create PSF 
####################################
psf = PSF(
    radial_lim=(0, 100.0),  # radial limits, keep < image width 
    axial_lim=(-100.0, 100.0),  # axial limits
)
rad_psf = psf.get_radial_psf()  # returns a 3D PSF, but shows the radial-axial profile
mc_psf = mcPSF().get_mc_psf()

# compose PSFs via convolution
psf.convolve_radial_psf(mc_psf)

# build 3D PSF 
psf = psf.build_3D_PSF()

######################################
# Plot voltage and optical signal and savefig
######################################
target_cell = target_population_cells[i_target_cell]
compart_ids = target_cell.get_list_compartment_ids()
# make two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
for compart_id in compart_ids:
    voltage_trace = target_cell.get_voltage_trace(compart_id)
    intensity_trace = target_cell.get_optical_trace(compart_id)
    axs[0].plot(loaded_compart_data['time'],voltage_trace, label='Voltage')
    axs[1].plot(loaded_compart_data['time'],intensity_trace, label='Intensity')
axs[1].set_xlabel('Time (ms)')
axs[0].set_ylabel('Membrane Potential (mV)')
axs[1].set_ylabel('df/f')
for ax in axs:
    ax.set_xlim(0, 200)

axs[0].set_title('ME-type:' + target_cell.get_me_type())
plt.savefig(model_rec_out_dir + f'{target_cell.get_me_type()}_cell_{target_cell.get_cell_id()}_voltage_optical.png')

######################################
# determine which morphology to use for each cell
######################################
for morph_key in me_type_morphology_map:
    for morph in me_type_morphology_map[morph_key]:
        for cell in target_population_cells:
            if cell.get_me_type() == morph.me_type:

                if morph.does_cell_match_morphology(cell):
                    cell.set_morphology(morph)

print("Any target cells missing structure data?:", 
      any([cell.get_morphology() == None 
           for cell in target_population_cells]))


#######################################
# Draw cell with PSF
#######################################
cam_width = 300
cam_height = 300
t = loaded_compart_data['time']
time_step_size = t[1] - t[0]
soma_position = target_cell.get_soma_position()
cam = Camera([target_cell], 
             me_type_morphology_map, 
             loaded_compart_data['time'],
             fov_center=soma_position,
             camera_resolution=1.0,
             camera_width=cam_width,
             camera_height=cam_height,
             psf=psf,
             data_dir=model_rec_out_dir, 
             use_2d_psf=False)
cam._draw_cell(target_cell)


for compart_id in ['soma', 'axon', 'apic', 'dend']:
    for activity_type in ['synaptic', 'spiking']:
        rec = cam.get_cell_recording().get_raw_recording(compart_id=compart_id, 
                                                         activity_type=activity_type)
        cam.animate_frames_to_video(rec, 
                        frames=(0,500),
                        filename='w_psf_' + compart_id + "_" + activity_type + '.gif',
                        time_step_size=time_step_size,
                        vmin=0,
                        vmax=0.01)
cam.close_memmaps()


#########################################
# Draw cell without PSF
#########################################
cam_no_psf = Camera([target_cell], 
             me_type_morphology_map, 
             loaded_compart_data['time'],
             fov_center=soma_position,
             camera_resolution=1.0,
             camera_width=cam_width,
             camera_height=cam_height,
             psf=None,
             data_dir=model_rec_out_dir + 'tmp/',#, #
             use_2d_psf=True)
cam_no_psf._draw_cell(target_cell)
for compart_id in ['soma', 'axon', 'apic', 'dend']:
    for activity_type in ['synaptic', 'spiking']:
        rec = cam_no_psf.get_cell_recording().get_raw_recording(compart_id=compart_id, 
                                                         activity_type=activity_type)
        cam_no_psf.animate_frames_to_video(rec, 
                        frames=(0,500),
                        filename='no_psf_' + compart_id + "_" + activity_type + '.gif',
                        time_step_size=time_step_size)
cam_no_psf.close_memmaps()