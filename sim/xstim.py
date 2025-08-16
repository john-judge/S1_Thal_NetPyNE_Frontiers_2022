""" This scripy explicitly attaches xstims to cells 
    after the network and sim have been built (after running sim.createSim()).
    It is used to add xstim functionality to the simulation, called from init.py
    
    Alternative to relying on netParams x-stim assignment, which may cause segfault
    when spatial bounding box is used."""

from neuron import h
import numpy as np


def attach_xstim_to_segments(sim, field, waveform, decay='1/r2', stim_radius=100):
    """
    Attach a single NetStim/XStim source to all segments within a cubic region.

    This is equivalent to mod_based=False, and does not require
         xtra.mod file compiled in NEURON.

    Parameters
    ----------
    sim : NetPyNE Sim object
    field : dict
        type: 'pointSource' or 'uniform'
        location: [x, y, z] for pointSource
        direction: 3-vector for uniform field
        sigma: conductivity in mS/mm (default 0.276)
    stim_params : dict
        {'delay':..., 'dur':...}
    decay : str
        '1/r' or '1/r2' distance-based decay
    stim_radius : float or None
        Maximum distance (microns) from electrode to apply stimulation. None = no cutoff.
    """

    seg_coords = []     # (cell_gid, sec, seg) tuples
    seg_positions = []  # corresponding x,y,z

    # Collect all segment positions

    for cell in sim.net.cells:  # local cells only, avoids MPI abort
        gid = cell.gid 
        for sec_name, sec_dict in cell.secs.items():
            sec = sec_dict['hSec']
            for seg in sec:
                try:
                    x = seg.x3d(seg.x)
                    y = seg.y3d(seg.x)
                    z = seg.z3d(seg.x)
                except Exception:
                    continue
                seg_coords.append((gid, sec, seg))
                seg_positions.append([x, y, z])

    seg_positions = np.array(seg_positions)  # shape (N,3)

    # Compute distances for pointSource stim
    if field['type'] == 'pointSource':
        dx = seg_positions[:,0] - field['location'][0]
        dy = seg_positions[:,1] - field['location'][1]
        dz = seg_positions[:,2] - field['location'][2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r[r < 1e-9] = 1e-9  # avoid divide by zero

        # Apply stim radius cutoff
        if stim_radius is not None:
            mask = r <= stim_radius
            seg_coords = [c for c, m in zip(seg_coords, mask) if m]
            seg_positions = seg_positions[mask]
            r = r[mask]

        # Compute extracellular potential with distance-based decay
        r_m = r * 1e-6 # convert to meters. Sigma is in mS/mm = S/m
        sigma = field.get('sigma', 0.276) # default conductivity in mS/mm
        I_A = waveform['amp'] * 1e-3 # convert mA to A
        if decay == '1/r':
            Vext = I_A / (4 * np.pi * sigma * r_m) * 1e3  # convert to mV
        elif decay == '1/r2':
            Vext = I_A / (4 * np.pi * sigma * r_m**2) * 1e3 # convert to mV
        elif decay == '1/r3':
            Vext = I_A / (4 * np.pi * sigma * r_m**3) * 1e3 # convert to mV
        else:
            raise ValueError("decay must be '1/r' or '1/r2'")

    elif field['type'] == 'uniform':
        raise Exception("Uniform field not yet implemented: fix the units before using")
        direction = np.array(field['direction'])
        Vext = waveform['amp'] * np.dot(seg_positions, direction)
    else:
        raise ValueError("Unsupported field type")

    # Attach IClamp to segments and set amplitude
    for (gid, sec, seg), v in zip(seg_coords, Vext):
        stim = h.IClamp(seg(0.5))
        stim.delay = waveform.get('delay', 0)
        stim.dur = waveform.get('dur', 1e9)
        stim.amp = v

    print(f"Applied extracellular stimulation to {len(seg_coords)} segments.")
