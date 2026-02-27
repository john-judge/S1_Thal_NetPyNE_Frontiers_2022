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
from src.hVOS.compartment_file import MemoryMappedCompartmentVoltages
from cam_params import cam_params
from src.hVOS.subconn import SubConnMap
from collections import Counter

####################################
# read command line args
#####################################

job_id = 0
if len(sys.argv) < 2:
    job_id = 0  # testing mode has no job_id
else:

    try:
        job_id = int(sys.argv[1])
    except ValueError:
        job_id = 0

total_jobs = 50
if len(sys.argv) > 2:
    try:
        total_jobs = int(sys.argv[2])
    except ValueError:
        total_jobs = 50

# allow command line to choose which cell subpopulation to target
hVOS_pop_choice = None
if len(sys.argv) > 3:
    hVOS_pop_choice = sys.argv[3]

random.seed(4322)

no_psf_only = False
psf_only = False
if 'psf_only' in sys.argv:
    psf_only = True
if 'no_psf_only' in sys.argv:
    no_psf_only = True

# whether to use Monte Carlo scattering
use_mc_scattering_psf = False

# optical tuning: soma:dend hVOS expression ratio
soma_dend_hVOS_ratio = 0.5

#####################################
# Find data in CHTC staging and extract data just for this job's cell
#####################################

# scnn1a-cre
hVOS_pop_choice = "PV"
target_hVOS_populations = ["L4_SS", "L4_PC"]
if 'scnn1a' in hVOS_pop_choice:
    target_hVOS_populations = ["L4_SS", "L4_PC"]

# PV-Cre: LBC and NBC (basket cells)
if 'PV' in hVOS_pop_choice:
    print("Targeting PV+ interneurons")
    target_hVOS_populations = ['L23_LBC', 'L4_LBC', 'L5_LBC', 'L6_LBC',
                               'L23_NBC', 'L4_NBC', 'L5_NBC', 'L6_NBC',
                               'L23_ChC', 'L4_ChC', 'L5_ChC', 'L6_ChC']

target_sparsity = 1.0
optical_type = "hVOS"
t_max = 999 # number of points to write to disk
cam_width = cam_params['cam_width']
cam_height = cam_params['cam_height']
camera_resolution = cam_params['cam_resolution']

# 'run1/' contains a subdirectory 'cell_dat' 
#   which contains a memmap numpy file for each cell in the network
#   each memmap file contains the voltage trace for each compartment of the cell
#   in the format 'v7_batch_1_0_0_V<compartment_id>_<cell_id>.dat'
run_id = 2
data_dir = '../analyze_output/'
morphology_data_dir = '../NMC_model/NMC.NeuronML2/'
model_rec_out_dir = data_dir + 'model_rec/'
model_rec_final_out_dir = data_dir + 'model_rec_final/'
if not os.path.exists(model_rec_out_dir):
    os.makedirs(model_rec_out_dir)
if not os.path.exists(model_rec_final_out_dir):
    os.makedirs(model_rec_final_out_dir)
# list subdirectories of run1/ 
#   each subdirectory is a range of compart_ids
#   e.g. 'apic_0_10', 'dend_10_20', etc.
# compart_data dict maps compart_id range to subdirectory name
analyze_dir = '../analyze_output/'
loaded_compart_data = MemoryMappedCompartmentVoltages(analyze_dir)
loaded_compart_data.load_existing_mmap(analyze_dir + 'v7_batch1_0_0_hash_map.pkl', 
                        analyze_dir + 'S1_results.npy',
                        t_shape=t_max)

# check get in loaded_compart_data
for cell_id in loaded_compart_data.hash_map:
    for comp in loaded_compart_data.hash_map[cell_id]:
        i_data, mmfp = loaded_compart_data.get_item(cell_id, comp)
        #print(' check get in loaded_compart_data', mmfp[i_data])
        break
    break
print("Total nonzero:", np.sum(loaded_compart_data.mmap_fp != 0))

# create a dict that maps compart_id 'Vcomp_#' to
# dicts, which each map cell_id 'cell_#' to the 
# cell's compartment data (the memmap file name) for that compart_id

# load time
mm_time_fp = data_dir + 'v7_batch1_0_0_time.dat'
time = np.memmap(mm_time_fp, dtype='float32', mode='r')
assert len(time) == t_max, \
        "Time length mismatch: " + str(len(time)) + \
            " != " + str(t_max)

# load cell_id to me_type map
me_type_map_file = analyze_dir + 'cell_id_to_me_type_map.pkl'
if os.path.exists(me_type_map_file):
    with open(me_type_map_file, 'rb') as f:
        cell_id_to_me_type_map = pickle.load(f)

#######################################
# for each soma, get a cell id and aggregate its axons, apics, dends
#######################################
cells = {}
me_type_morphology_map = {}
for cell_id in loaded_compart_data.hash_map.keys():
    print(cell_id)
    axons, apics, dends, soma = {}, {}, {}, None
    for compart in loaded_compart_data.hash_map[cell_id].keys():
        data = loaded_compart_data.get_item(cell_id, compart)
        if data is None:
            print("Data not found for cell:", cell_id, "compartment:", compart)
            continue

        if 'soma' in compart:
            soma = data
        elif 'axon' in compart:
            axons[compart] = data
        elif 'apic' in compart:
            apics[compart] = data
        elif 'dend' in compart:
            dends[compart] = data
        else:
            print("Unknown compart:", compart)
            continue

    if soma is None:
        print("No soma found for cell:", cell_id)
        continue

    short_cell_id = int(cell_id.replace('cell_', ''))
    me_type = cell_id_to_me_type_map[short_cell_id]['me_type']
    x = cell_id_to_me_type_map[short_cell_id]['x']
    y = cell_id_to_me_type_map[short_cell_id]['y']
    z = cell_id_to_me_type_map[short_cell_id]['z']
    cells[cell_id] = Cell(cell_id, me_type, axons, apics, dends, soma, x, y, z, optical_filelabel=optical_type)

    me_type = me_type.split("_barrel")[0]  # remove barrel suffix if present
    if me_type not in me_type_morphology_map:
        # load morphology
        # find files in morphology_data_dir with me_type in the name
        print("Loading morphology for me_type:", me_type)
        me_type_part1, me_type_part2, me_type_part3 = me_type.split("_")
        m_type, e_type = me_type_part1 + "_" + me_type_part2, me_type_part3
        me_type_files = [f for f in os.listdir(morphology_data_dir) if (m_type in f and e_type in f and f.endswith('.cell.nml'))]
        if len(me_type_files) == 0:
            assert me_type not in target_hVOS_populations  # we only care if we cannot load a target population
        
        # load all morphology file matches
        me_type_morphology_map[me_type] = [Morphology(me_type, morphology_data_dir + me_type_file) for me_type_file in me_type_files]


#######################################
# Diagnostic: count cells in hVOS target populations
#######################################
# count by ME-type
me_types_all = [c.get_me_type().split("_barrel")[0] for c in cells.values()]
pop_counts = Counter(me_types_all)

print("=== Diagnostic Count 1: hVOS target population counts ===")
total = 0
for pop in target_hVOS_populations:
    count = pop_counts.get(pop, 0)
    print(f"  {pop}: {count} cells")
    total += count
print(f"  TOTAL across targets: {total}\n")


# cells dict maps cell_id to Cell object
# me_type_morphology_map maps me_type to list of Morphology objects
# target populations is a list of me_types to compute signal for
target_population_cells = [
    cells[cell_id] for cell_id in cells 
        if any([t_pop in cells[cell_id].get_me_type() for 
                    t_pop in target_hVOS_populations ]) 
]

# indexes of cells to process for this job
i_target_cells = []
i_t = job_id
while i_t < len(target_population_cells):
    if i_t < len(target_population_cells):
        i_target_cells.append(i_t)
    i_t += total_jobs
print("Job id:", job_id, "of", total_jobs, "processing cells:", i_target_cells)


print("=== Diagnostic Count 2: target_population_cells BEFORE sparsity ===")
print(f"  Total: {len(target_population_cells)}")
pop_counts_before = Counter([c.get_me_type().split("_barrel")[0] for c in target_population_cells])
for pop in target_hVOS_populations:
    print(f"  {pop}: {pop_counts_before.get(pop, 0)} cells")
print()

 # sparsity sampling implemented here
target_population_cells = [cell for cell in target_population_cells
                                if random.random() < target_sparsity]

print("=== Diagnostic Count 3: target_population_cells AFTER sparsity ===")
print(f"  Total: {len(target_population_cells)}")
pop_counts_after = Counter([c.get_me_type().split("_barrel")[0] for c in target_population_cells])
for pop in target_hVOS_populations:
    print(f"  {pop}: {pop_counts_after.get(pop, 0)} cells")
print()

# choose cells for this job to draw
cells_to_draw = None
try:
    cells_to_draw = [target_population_cells[i_target_cell] for i_target_cell in i_target_cells]
except IndexError as e:
    print("Tried to get ", i_target_cells, 
          "th cell from target_population_cells of length", len(target_population_cells))
    print("Error:", e)

optical_readout = {'hVOS': hVOSReadout, 'VSD': VSDReadout}[optical_type]
hvos_readout = optical_readout(target_hVOS_populations, 
                               {cell.get_cell_id(): cell for cell in cells_to_draw}, 
                               me_type_morphology_map,
                               force_overwrite=True)
hvos_readout.compute_optical_signal(data_dir)
#hvos_readout.show_voltage_to_intensity_curve()


##########################################
# plot signal, choose random cell
##########################################

cell = random.choice(cells_to_draw)
compart_ids = cell.get_list_compartment_ids()
plt.clf()
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
for compart_id in compart_ids:
    voltage_trace = cell.get_voltage_trace(compart_id)
    intensity_trace = cell.get_optical_trace(compart_id)
    axs[0].plot(time,voltage_trace, label='Voltage')
    axs[1].plot(time,intensity_trace, label='Intensity')
axs[1].set_xlabel('Time (ms)')
axs[0].set_ylabel('Membrane Potential (mV)')
axs[1].set_ylabel('df/f')

axs[0].set_title('ME-type:' + cell.get_me_type())
plt.savefig(model_rec_final_out_dir + 'signal.png')


####################################
# Create PSF 
####################################
psf = PSF(
    radial_lim=(0, 100.0),  # radial limits, keep < image width 
    axial_lim=(-100.0, 100.0),  # axial limits
)
rad_psf = psf.get_radial_psf()  # returns a 3D PSF, but shows the radial-axial profile

if use_mc_scattering_psf:
    mc_psf = mcPSF().get_mc_psf()

    # compose PSFs via convolution
    psf.convolve_radial_psf(mc_psf)

# build 3D PSF 
psf = psf.build_3D_PSF()


######################################
# determine which morphology to use for each cell
######################################

target_cell_0 = cells_to_draw[0]
compart_ids = target_cell_0.get_list_compartment_ids()
for morph_key in me_type_morphology_map:
    for morph in me_type_morphology_map[morph_key]:
        print("Seeking match for morphology:", morph.me_type)
        for cell in target_population_cells:
            if cell.get_me_type().split("_barrel")[0] == morph.me_type:

                if morph.does_cell_match_morphology(cell):
                    cell.set_morphology(morph)
        

if any([cell.get_morphology() == None 
           for cell in target_population_cells]):
    print(str(sum([cell.get_morphology() == None 
           for cell in target_population_cells])) + " of " + str(len(target_population_cells)) + " target cells are missing structure data:")
    # report which cells are missing morphology
    for cell in target_population_cells:
        if cell.get_morphology() is None:
            print("Cell", cell.get_cell_id(), "is missing morphology for me_type", cell.get_me_type())
        

#######################################
# Draw cells with PSF
#######################################
os.makedirs(model_rec_out_dir + 'psf/', exist_ok=True)

time_step_size = time[1] - time[0]


view_center_cell = 0  # view center cell is the cell to center on.
# other cells may or may not be in view.
soma_position = None
if cam_params['cam_fov'] is not None:
    if type(cam_params['cam_fov']) == int:
        view_center_cell = cam_params['cam_fov']
        soma_position = target_population_cells[view_center_cell].get_soma_position()
    elif type(cam_params['cam_fov']) == list:
        soma_position = cam_params['cam_fov']
    else:
        soma_position = target_population_cells[view_center_cell].get_soma_position()

print("location of soma of cell to center on:", soma_position)
if not no_psf_only:
    for target_cell in cells_to_draw:
        cell_model_rec_out_dir = model_rec_out_dir + 'psf/' + target_cell.get_cell_id() + '/'
        os.makedirs(cell_model_rec_out_dir, exist_ok=True)

        # look for precomputed geometry files in current directory
        geometry_cache = f'geometry_cache_{target_cell.get_cell_id()}.pkl'
        geometry_cache_file = None
        if os.path.exists(geometry_cache):
            print("Using existing geometry cache for cell:", target_cell.get_cell_id())
            geometry_cache_file = geometry_cache
        else:
            print("Geometry cache not found for cell:", target_cell.get_cell_id(),
                   "; will create new cache at", geometry_cache)
        cam = Camera([target_cell], 
                    me_type_morphology_map, 
                    time,
                    fov_center=soma_position,
                    camera_resolution=camera_resolution,
                    camera_width=cam_width,
                    camera_height=cam_height,
                    psf=psf,
                    data_dir=cell_model_rec_out_dir, 
                    use_2d_psf=False,
                    soma_dend_hVOS_ratio=soma_dend_hVOS_ratio,
                    geometry_cache_filename=geometry_cache_file,
                    precompute_geometry=True
                    )
        cam._draw_cell(target_cell)
        if geometry_cache_file is None:
            cam.save_geometry(filename=geometry_cache)

        psf_nonzero_files = cam.get_cell_recording().get_non_zero_file_list()
        print("PSF non-zero files:", psf_nonzero_files)
        cam.close_memmaps()


#########################################
# Draw cells without PSF
#########################################
os.makedirs(model_rec_out_dir + 'no_psf/', exist_ok=True)
if not psf_only:
    for target_cell in cells_to_draw:
        cell_model_rec_out_dir = model_rec_out_dir + 'no_psf/' + target_cell.get_cell_id() + '/'
        os.makedirs(cell_model_rec_out_dir, exist_ok=True)

        # look for precomputed geometry files in current directory
        geometry_cache = f'geometry_cache_{target_cell.get_cell_id()}.pkl'
        geometry_cache_file = None
        if os.path.exists(geometry_cache):
            print("Using existing geometry cache for cell:", target_cell.get_cell_id())
            geometry_cache_file = geometry_cache
        else:
            print("Geometry cache not found for cell:", target_cell.get_cell_id(),
                   "; will create new cache at", geometry_cache)

        cam_no_psf = Camera([target_cell], 
                    me_type_morphology_map, 
                    time,
                    fov_center=soma_position,
                    camera_resolution=camera_resolution,
                    camera_width=cam_width,
                    camera_height=cam_height,
                    psf=None,
                    data_dir=cell_model_rec_out_dir,#, #
                    use_2d_psf=True,
                    soma_dend_hVOS_ratio=soma_dend_hVOS_ratio,
                    geometry_cache_filename=geometry_cache_file,
                    precompute_geometry=True
                    )
        cam_no_psf._draw_cell(target_cell)
        
        if geometry_cache_file is None:
            cam_no_psf.save_geometry(filename=geometry_cache)

        '''for compart_id in ['soma', 'axon', 'apic', 'dend']:
            for activity_type in ['synaptic', 'spiking']:
                rec = cam_no_psf.get_cell_recording().get_raw_recording(compart_id=compart_id, 
                                                                activity_type=activity_type)
                cam_no_psf.animate_frames_to_video(rec, 
                                frames=(0,500),
                                filename='no_psf_' + compart_id + "_" + activity_type + '.gif',
                                time_step_size=time_step_size)'''
        no_psf_nonzero_files = cam_no_psf.get_cell_recording().get_non_zero_file_list()
        print("No-PSF non-zero files:", no_psf_nonzero_files)
        cam_no_psf.close_memmaps()

############################################
# draw synapse locations for one cell
############################################
# subconn map on a cell
syn_nonzero_files = []
if job_id ==0 and not psf_only:
    try:
        subconn_map = SubConnMap(run_filepath=analyze_dir + 'v7_batch1_0_0.run',
                                post_cell_id=target_population_cells[view_center_cell].get_cell_id())
        os.makedirs(model_rec_out_dir + 'syn/', exist_ok=True)
        cam_no_psf_syn = Camera([target_population_cells[view_center_cell]], 
                            me_type_morphology_map, 
                            time,
                            fov_center=soma_position,
                            camera_resolution=camera_resolution,
                            camera_width=cam_width,
                            camera_height=cam_height,
                            psf=None,
                            data_dir=model_rec_out_dir + 'syn/',
                            use_2d_psf=True,
                            draw_synapses=subconn_map,
                            soma_dend_hVOS_ratio=soma_dend_hVOS_ratio)
        cam_no_psf_syn._draw_cell(target_population_cells[view_center_cell])
        syn_map = cam_no_psf_syn.synapse_mask
        # write synapse map to memmap file in model_rec_out_dir
        syn_map_file = model_rec_out_dir + 'syn/' + target_population_cells[view_center_cell].get_cell_id() + '_syn_map.npy'
        mm_fp_syn = np.memmap(syn_map_file, dtype='float32', mode='w+', shape=(cam_height, cam_width))
        mm_fp_syn[:] = syn_map[:]
        mm_fp_syn.flush()
        syn_nonzero_files = cam_no_psf_syn.get_cell_recording().get_non_zero_file_list()
        syn_nonzero_files.append(syn_map_file)
        cam_no_psf_syn.close_memmaps()
    except Exception as e:
        print("Error in subconn map:", e)
        print("No synapse map created.")

###########################################
# Copy non-zero files to model_rec_final_out_dir
###########################################
os.makedirs(model_rec_final_out_dir, exist_ok=True)
if not no_psf_only:
    for file in psf_nonzero_files:
        file_name = "psf_" + file.split('/')[-1]
        cell_subdir = file.split('/')[-2] + '/'
        target_subdir = model_rec_final_out_dir + cell_subdir
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)
        target_file = model_rec_final_out_dir + cell_subdir + file_name
        if not os.path.exists(target_file):
            os.system('cp ' + file + ' ' + target_file)
if not psf_only:
    for file in no_psf_nonzero_files:
        file_name = "no_psf_" + file.split('/')[-1]
        cell_subdir = file.split('/')[-2] + '/'
        target_subdir = model_rec_final_out_dir + cell_subdir
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)
        target_file = model_rec_final_out_dir + cell_subdir + file_name
        if not os.path.exists(target_file):
            os.system('cp ' + file + ' ' + target_file)
    for file in syn_nonzero_files:
        file_name = "syn_" + file.split('/')[-1]
        cell_subdir = file.split('/')[-2] + '/'
        target_subdir = model_rec_final_out_dir + cell_subdir
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)
        target_file = model_rec_final_out_dir + cell_subdir + file_name
        if not os.path.exists(target_file):
            os.system('cp ' + file + ' ' + target_file)