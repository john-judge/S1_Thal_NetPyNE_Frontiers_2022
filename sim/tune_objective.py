import numpy as np
import gc

# from recording 10/23/2024 slice 1 L2/3_Stim
exp_data = {
    'nbqx_halfwidth': 4.3623,
    'nbqx_latency': 3.5624,
    'nbqx_amplitude': 0.0672,
    'acsf_amplitude': 0.0547,
    'nbqx_acsf_ratio': 1.2354
}

def extract_traces(simData):
    # return a dict populated by traces[cell_id][compartment] = trace
    traces = {}
    for k in simData:
        for compart in ['soma', 'dend', 'apic', 'axon']:
            if compart in k:
                traces[k] = {}
                for cell_id in simData[k]:
                    traces[cell_id][k] = np.array(simData[k][cell_id])
                    print(f"Extracted trace for cell {cell_id} compartment {compart} with shape {traces[k][cell_id].shape}")
    
    return traces

def extract_features(trace, tvec):
    """Return amplitude, latency, half-width."""
    baseline = np.mean(trace[:int(0.05*len(trace))])
    peak_val = np.min(trace)  # assuming downward deflection
    peak_idx = np.argmin(trace)
    amp = baseline - peak_val
    latency = tvec[peak_idx]
    
    half_amp = baseline - amp/2
    # indices where trace crosses half amplitude
    crossings = np.where(trace < half_amp)[0]
    if len(crossings) >= 2:
        hw = (crossings[-1] - crossings[0]) * (tvec[1]-tvec[0])
    else:
        hw = np.nan
    return amp, latency, hw

def myObjective(simData, cfg, netParams):
    """
    Custom objective for NetPyNE Batch optimization.
    Called automatically after each simulation.
    
    simData: dictionary with recorded traces
    cfg: simulation configuration object
    netParams: network parameters
    """

    traces = extract_traces(simData)

    # Average over each cell and call it a pixel
    # this at least roughly preserves the soma:neurite ratio
    avg_traces = {}
    for cell_id in traces:
        traces = np.average(np.array([
            traces[cell_id][k] for k in traces[cell_id]
        ]), axis=0)
        avg_traces[cell_id] = traces

    # gc cleanup traces to save memory
    del traces
    gc.collect()
    
    tvec = simData['t']  # ms
    features = [extract_features(tr, tvec) for tr in avg_traces.values()]

    # Average features across pixels
    amps, lats, hws = zip(*features)
    amp_mean = np.nanmean(amps)
    lat_mean = np.nanmean(lats)
    hw_mean  = np.nanmean(hws)

    # --------------------------
    # 3. Compare ACSF vs NBQX
    # --------------------------
    # This depends on whether we’re in ACSF or NBQX run
    if cfg.experiment_NBQX_global:  # ACSF run
        # store NBQX results
        result = {'amp': amp_mean, 'lat': lat_mean, 'hw': hw_mean}
        np.save('nbqx_results.npy', result)
        return None
    else:  # NBQX run
        # load NBQX results (assumes NBQX run already done)
        nbqx = np.load('nbqx_results.npy', allow_pickle=True).item()
        acsf = {'amp': amp_mean, 'lat': lat_mean, 'hw': hw_mean}

        sim_ratio   = nbqx['amp'] / acsf['amp'] if acsf['amp']>0 else np.nan
        sim_latency = nbqx['lat']
        sim_hw      = nbqx['hw']

        # squared error cost
        err = (sim_ratio-exp_data['nbqx_acsf_ratio'])**2 + \
              (sim_latency-exp_data['nbqx_latency'])**2 + \
              (sim_hw-exp_data['nbqx_halfwidth'])**2

        return err
