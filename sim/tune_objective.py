import numpy as np
import gc
from measure_properties import TraceProperties
from netpyne import sim 
import os
import json
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.insert(0, '../src/hVOS')  # import from src/hVOS

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


# from recording 10/23/2024 slice 1 L2/3_Stim
start_time = 500

exp_data = {
    'nbqx_halfwidth_mean': 4.3623,
    'nbqx_latency_mean': 3.5624,
    'nbqx_amplitude_mean': 0.0672,
    'acsf_amplitude_mean': 0.0547,
    'nbqx_acsf_ratio_mean': 1.2354,
    'nbqx_halfwidth_std': 0.7185,
    'nbqx_latency_std': 0.3275,
    'nbqx_amplitude_std': 0.0128,
    'acsf_amplitude_std': 0.0102,
    'nbqx_acsf_ratio_std': 0.1335
}

mse_weights = {
    'halfwidth': 1.0,
    'latency': 1.0,
    'ratio': 1.0,
}

def extract_traces(simData):
    # return a dict populated by traces[cell_id][compartment] = trace
    traces = {}
    num_prints = 5
    for k in simData:
        for compart in ['soma', 'dend', 'apic', 'axon']:
            if compart in k:
                for cell_id in simData[k]:
                    if cell_id not in traces:
                        traces[cell_id] = {}
                    traces[cell_id][k] = np.array(simData[k][cell_id])
                    if num_prints > 0:
                        num_prints -= 1
                        print(f"Extracted trace for cell {cell_id} compartment {compart} with shape {traces[cell_id][k].shape}")

    # Average over each 5 cells and call it a pixel
    # this at least roughly preserves the soma:neurite ratio and multi-cell blurring
    avg_traces = {}
    for cell_id in traces:
        # avoid dividing by zero if no traces
        if len(traces[cell_id].keys()) == 0:
            continue
        avg_traces_ = np.array([traces[cell_id][k] for k in traces[cell_id]])
        # avoid dividing by zero if no traces
        if avg_traces_.shape[0] == 0:
            continue
        try:
            avg_traces[cell_id] = np.average(avg_traces_, axis=0)
        except ZeroDivisionError as e:
            print("ZeroDivisionError for cell_id:", cell_id, "with avg_traces shape:", avg_traces.shape)
    # create random subsets of 5 cells and average them to create "pixels"
    cell_ids = list(avg_traces.keys())
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(cell_ids)
    pixel_traces = {}
    for i in range(0, len(cell_ids), 5):
        pixel_id = f"pixel_{i//5}"
        subset = cell_ids[i:i+5]
        if len(subset) == 0:
            continue
        pixel_traces[pixel_id] = np.mean([avg_traces[cid] for cid in subset if cid in avg_traces], axis=0)

    # gc cleanup traces to save memory
    del traces
    del avg_traces
    gc.collect()

    return pixel_traces

def extract_features(traces, tvec):
    """Return amplitude, latency, half-width."""
    int_pts = tvec[1] - tvec[0]  # integration points (sampling interval)
    features = []
    processed_traces = []
    for tr in traces:
        # flatten the trace from start_time to end
        # i.e. draw a line from the value at start_time to the value at end_time
        # and subtract that line from the trace
        x_ = np.array([start_time, tr.shape[0]-1])
        y = np.array([tr[start_time], tr[-1]])
        m, b = np.polyfit(x_, y, 1)
        trend = np.polyval([m, b], np.arange(tr.shape[0]))
        tr = tr - trend
        tr = tr[start_time:]  # only analyze from start_time onward, and invert
        tr -= np.min(tr)  # baseline to 0
        tp = TraceProperties(tr, start=0, width=400, int_pts=int_pts)
        tp.measure_properties()
        features.append([tp.get_max_amp(), tp.get_half_amp_latency(), tp.get_half_width()])
        processed_traces.append(tr)
    return np.array(features), processed_traces

def myObjective(simData):
    try:
        return myObjectiveInner(simData)
    except Exception as e:
        import traceback
        traceback.print_exc()   # will print full traceback to your terminal
        raise

def load_cell_id_to_me_type_map(file_path):
    cell_id_to_me_type_map = {}
    target_dir_net = '../data/optuna_tuning/gen_0/trial_0_data.pkl'

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
    return cell_id_to_me_type_map

def average_voltage_traces_into_hVOS_pixels(simData, cells, me_type_morphology_map,
                                            target_hVOS_populations = ("L4_SS", "L4_PC")):
    target_population_cells = [
    cells[cell_id] for cell_id in cells 
        if any([t_pop in cells[cell_id].get_me_type() for 
                    t_pop in target_hVOS_populations ]) 
    ]
    hvos_readout = hVOSReadout(target_hVOS_populations, 
                                {cell.get_cell_id(): cell for cell in target_population_cells}, 
                                me_type_morphology_map,
                                force_overwrite=True)
    
    curr_trial = max([int(d.split("gen_")[-1]) for d in os.listdir(save_folder) if (os.path.isdir(os.path.join(save_folder, d)) and 'gen_' in d)])
    save_folder = '../data/optuna_tuning/gen_' + str(curr_trial)
    hvos_readout.compute_optical_signal(save_folder)

    ####################################
    # Create PSF 
    ####################################
    psf = PSF(
        radial_lim=(0, 100.0),  # radial limits, keep < image width 
        axial_lim=(-100.0, 100.0),  # axial limits
    )
    # build 3D PSF 
    psf = psf.build_3D_PSF()


    ######################################
    # determine which morphology to use for each cell
    ######################################
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
            for cell in target_population_cells])) + " target cells are missing structure data:")
        # report which cells are missing morphology
        for cell in target_population_cells:
            if cell.get_morphology() is None:
                raise("Cell", cell.get_cell_id(), "is missing morphology for me_type", cell.get_me_type())

    #######################################
    # Draw cells with PSF
    #######################################
    model_rec_out_dir = save_folder + '/'
    os.makedirs(model_rec_out_dir + 'psf/', exist_ok=True)
    time = np.array(simData['t'])  # time vector is the same for both conditions

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

    cam_width = cam_params['cam_width']
    cam_height = cam_params['cam_height']
    camera_resolution = cam_params['cam_resolution']
    soma_dend_hVOS_ratio = 0.5
    comparts = ['axon', 'dend', 'soma', 'apic']
    all_cells_rec = None
    print("location of soma of cell to center on:", soma_position)
    for target_cell in target_population_cells:
        cell_model_rec_out_dir = model_rec_out_dir + 'psf/' + target_cell.get_cell_id() + '/'
        os.makedirs(cell_model_rec_out_dir, exist_ok=True)
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
                    soma_dend_hVOS_ratio=soma_dend_hVOS_ratio
                    )
        cam._draw_cell(target_cell)

        recording = cam.get_cell_recording()  # returns a CellRecording object
        recording = recording.get_combined_recording()


        # store recording in all_cells_rec, superimposed
        if all_cells_rec is None:
            all_cells_rec = recording
        else:
            for compart in comparts:
                all_cells_rec[:] += recording[:]

        psf_nonzero_files = cam.get_cell_recording().get_non_zero_file_list()
        print("PSF non-zero files:", psf_nonzero_files)
        cam.close_memmaps()

        # delete all zero files to save disk space
        for file in os.listdir(cell_model_rec_out_dir):
            if file not in psf_nonzero_files and (file.endswith('.mm') or file.endswith('.npy')):
                os.remove(os.path.join(cell_model_rec_out_dir, file))
        del cam
        gc.collect()
    return all_cells_rec

def load_morphologies(simData, cell_id_to_me_type_map, 
                        target_hVOS_populations = ("L4_SS", "L4_PC")):
    
    # simData is keyed by cell_id and compartment name
    morphology_data_dir = '../../NMC_model/NMC.NeuronML2/'

    #######################################
    # for each soma, get a cell id and aggregate its axons, apics, dends
    #######################################
    cells = {}
    me_type_morphology_map = {}
    for cell_id in simData:
        axons, apics, dends, soma = {}, {}, {}, None
        for compart in simData[cell_id].keys():
            data = simData[cell_id][compart]
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
            m_type, e_type = me_type[:6], me_type[7:]
            me_type_files = [f for f in os.listdir(morphology_data_dir) if (m_type in f and e_type in f and f.endswith('.cell.nml'))]
            if len(me_type_files) == 0:
                assert me_type not in target_hVOS_populations  # we only care if we cannot load a target population
            
            # load all morphology file matches
            me_type_morphology_map[me_type] = [Morphology(me_type, morphology_data_dir + me_type_file) for me_type_file in me_type_files]

    return cells, me_type_morphology_map


def myObjectiveInner(simData):
    # simData['acsf'] and simData['nbqx'] are the two conditions
    # each is a dict with keys like 'Vsoma', 'Vdend_32', etc
    """
    Custom objective for NetPyNE Batch optimization.
    Called automatically after each simulation.
    
    simData: dictionary with recorded traces
    cfg: simulation configuration object
    netParams: network parameters
    """
    
    simData = simData['simData']
    
    simData_acsf = simData['acsf']
    simData_nbqx = simData['nbqx']

    print("simData_acsf keys:", simData_acsf.keys())
    if 'simConfig' in simData_acsf:
        cfg = simData_acsf['simConfig']
        print("simConfig.experiment_NBQX_global for acsf:", cfg.experiment_NBQX_global)
    print("simData_nbqx keys:", simData_nbqx.keys())
    if 'simConfig' in simData_nbqx:
        cfg = simData_nbqx['simConfig']
        print("simConfig.experiment_NBQX_global for nbqx:", cfg.experiment_NBQX_global)

    cell_id_to_me_type_map = load_cell_id_to_me_type_map('../data/cell_id_to_me_type_map.json')
    cells_acsf, me_type_morphology_map = load_morphologies(simData_acsf, cell_id_to_me_type_map)
    cells_nbqx, _ = load_morphologies(simData_nbqx, cell_id_to_me_type_map)
    # TO DO: put hVOS/optical processing here instead of just using voltage traces
    simData_traces_acsf = average_voltage_traces_into_hVOS_pixels(simData_acsf, cells_acsf, 
                                                                  me_type_morphology_map)
    simData_traces_nbqx = average_voltage_traces_into_hVOS_pixels(simData_nbqx, cells_nbqx, 
                                                                  me_type_morphology_map)
    tvec = np.array(simData_acsf['t'])  # time vector is the same for both conditions

    # Compare ACSF vs NBQX
    acsf_features, acsf_processed_traces = extract_features(list(simData_traces_acsf.values()), tvec)
    nbqx_features, nbqx_processed_traces = extract_features(list(simData_traces_nbqx.values()), tvec)

    nbqx_features[:, 0] = nbqx_features[:, 0] / acsf_features[:, 0]  # first col is ratios

    # now we have 3 features: ratio, latency, half-width stored in 3 columns
    # of nbqx_features
    # let's generate a Gaussian distribution of experimental values
    # with the same number of samples as we have simulated cells
    # Then we compute the mean squared error between simulated and experimental
    num_cells = nbqx_features.shape[0]
    # random seed 
    np.random.seed(4322)
    exp_ratio = np.random.normal(loc=exp_data['nbqx_acsf_ratio_mean'], scale=exp_data['nbqx_acsf_ratio_std'], size=num_cells)
    exp_latency = np.random.normal(loc=exp_data['nbqx_latency_mean'], scale=exp_data['nbqx_latency_std'], size=num_cells)
    exp_hw = np.random.normal(loc=exp_data['nbqx_halfwidth_mean'], scale=exp_data['nbqx_halfwidth_std'], size=num_cells)

    sim_ratio = nbqx_features[:, 0]
    sim_latency = nbqx_features[:, 1]
    sim_hw = nbqx_features[:, 2]

    # return mean squared error cost, normalized to target (experimental) value
    err_ratio = mse_weights['ratio'] * (np.mean(sim_ratio)-np.mean(exp_ratio))**2 / np.mean(exp_ratio)
    err_latency = mse_weights['latency'] * (np.mean(sim_latency)-np.mean(exp_latency))**2 / np.mean(exp_latency)
    err_hw = mse_weights['halfwidth'] * (np.mean(sim_hw)-np.mean(exp_hw))**2 / np.mean(exp_hw)
    print(
          f"ratio err: {err_ratio}, "
          f"latency err: {err_latency}, "
          f"halfwidth err: {err_hw}")
    
    # save raw components for analysis
    save_folder = '../data/optuna_tuning'
    curr_trial = max([int(d.split("gen_")[-1]) for d in os.listdir(save_folder) if (os.path.isdir(os.path.join(save_folder, d)) and 'gen_' in d)])
    with open(os.path.join(save_folder, f"fitness_components_trial{curr_trial}.json"), 'w') as f:
        json.dump({
            'err_ratio': err_ratio,
            'err_latency': err_latency,
            'err_hw': err_hw,
            'sim_ratio_mean': np.mean(sim_ratio),
            'sim_latency_mean': np.mean(sim_latency),
            'sim_hw_mean': np.mean(sim_hw),
            'exp_ratio_mean': np.mean(exp_ratio),
            'exp_latency_mean': np.mean(exp_latency),
            'exp_hw_mean': np.mean(exp_hw),
        }, f, indent=4)
    print("Files in ", save_folder, ": ", os.listdir(save_folder))

    # save the simulated traces of this trial to file
    # pickle simData_traces_acsf and simData_traces_nbqx
    with open(os.path.join(save_folder, f"simData_traces_acsf_trial{curr_trial}.pkl"), 'wb') as f:
        pickle.dump(simData_traces_acsf, f)
    with open(os.path.join(save_folder, f"simData_traces_nbqx_trial{curr_trial}.pkl"), 'wb') as f:
        pickle.dump(simData_traces_nbqx, f)

    # plot processed traces for this trial
    plt.figure(figsize=(12, 6))
    for tr in acsf_processed_traces:
        plt.plot(tr, color='blue', alpha=0.1)
    for tr in nbqx_processed_traces:
        plt.plot(tr, color='red', alpha=0.1)
    plt.title(f"Processed traces for trial {curr_trial} (blue=ACSF, red=NBQX)")
    plt.xlabel("Time (ms)")
    plt.ylabel("ΔF/F") 
    plt.savefig(os.path.join(save_folder, f"processed_traces_trial{curr_trial}.png"))
    plt.close()

    

    return err_ratio + err_latency + err_hw
