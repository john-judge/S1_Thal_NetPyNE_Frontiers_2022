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
    for gid, cell in sim.net.allCells.items():
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
        if decay == '1/r':
            Vext = waveform['amp'] / (4 * np.pi * r)
        elif decay == '1/r2':
            Vext = waveform['amp'] / (4 * np.pi * r**2)
        elif decay == '1/r3':
            Vext = waveform['amp'] / (4 * np.pi * r**3)
        else:
            raise ValueError("decay must be '1/r' or '1/r2'")

    elif field['type'] == 'uniform':
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
