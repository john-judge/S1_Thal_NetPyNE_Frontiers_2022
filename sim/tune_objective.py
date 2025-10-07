import numpy as np
import gc
from measure_properties import TraceProperties
from netpyne import sim 

# from recording 10/23/2024 slice 1 L2/3_Stim
start_time = 500

# --- cache dict, keyed by propVelocity ---
_acsf_cache = {}

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

    # Average over each cell and call it a pixel
    # this at least roughly preserves the soma:neurite ratio
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

    # gc cleanup traces to save memory
    del traces
    gc.collect()

    return avg_traces

def extract_features(traces, tvec):
    """Return amplitude, latency, half-width."""
    int_pts = tvec[1] - tvec[0]  # integration points (sampling interval)
    features = []
    for tr in traces:
        # flatten the trace from start_time to end
        # i.e. draw a line from the value at start_time to the value at end_time
        # and subtract that line from the trace
        x_ = np.array([start_time, tr.shape[0]-1])
        y = np.array([tr[start_time], tr[-1]])
        m, b = np.polyfit(x_, y, 1)
        trend = np.polyval([m, b], np.arange(tr.shape[0]))
        tr = tr - trend
        tr = -tr[start_time:]  # only analyze from start_time onward, and invert
        tr -= np.min(tr)  # baseline to 0
        tp = TraceProperties(tr, start=0, width=400, int_pts=int_pts)
        tp.measure_properties()
        features.append([tp.get_max_amp(), tp.get_half_amp_latency(), tp.get_half_width()])
    return np.array(features)

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
    simData_traces_acsf = extract_traces(simData_acsf)
    simData_traces_nbqx = extract_traces(simData_nbqx)
    tvec = np.array(simData_acsf['t'])  # time vector is the same for both conditions

    # Compare ACSF vs NBQX
    acsf_features = extract_features(list(simData_traces_acsf.values()), tvec)
    nbqx_features = extract_features(list(simData_traces_nbqx.values()), tvec)

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

    # return mean squared error cost
    err_ratio = mse_weights['ratio'] * (np.mean(sim_ratio)-np.mean(exp_ratio))**2
    err_latency = mse_weights['latency'] * (np.mean(sim_latency)-np.mean(exp_latency))**2
    err_hw = mse_weights['halfwidth'] * (np.mean(sim_hw)-np.mean(exp_hw))**2
    print(
          f"ratio err: {np.sum(err_ratio)}, "
          f"latency err: {np.sum(err_latency)}, "
          f"halfwidth err: {np.sum(err_hw)}")

    return err_ratio, err_latency, err_hw
