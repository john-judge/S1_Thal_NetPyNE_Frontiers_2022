import numpy as np
import gc
from measure_properties import TraceProperties
from netpyne import sim 
import os
import json
import matplotlib.pyplot as plt
import pickle


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

    # TO DO: put hVOS/optical processing here instead of just using voltage traces
    simData_traces_acsf = extract_traces(simData_acsf)
    simData_traces_nbqx = extract_traces(simData_nbqx)
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
