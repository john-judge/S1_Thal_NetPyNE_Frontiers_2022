""" This scripy explicitly attaches xstims to cells 
    after the network and sim have been built (after running sim.createSim()).
    It is used to add xstim functionality to the simulation, called from init.py
    
    Alternative to relying on netParams x-stim assignment, which may cause segfault
    when spatial bounding box is used."""

from neuron import h
import numpy as np


def attach_xstim_to_segments(sim, field, waveform, decay='1/r', stim_radius=100):
    """
    Attach a single NetStim/XStim source to all segments within a cubic region.

    This is equivalent to mod_based=False, and does not require
         xtra.mod file compiled in NEURON.

    Parameters
    ----------
    sim : NetPyNE Sim object
    field : dict
        class: 'pointSource' or 'uniform'
        location: [x, y, z] for pointSource
        sigma: conductivity in mS/mm (default 0.276)
    stim_params : dict
        {'delay':..., 'dur':...}
    decay : str
        '1/r' or '1/r2' distance-based decay
    stim_radius : float or None
        Maximum distance (microns) from electrode to apply stimulation. None = no cutoff.
    """
    pc = h.ParallelContext()
    seg_coords = []     # (cell_gid, sec, seg) tuples
    seg_positions = []  # corresponding x,y,z

    # Collect all segment positions

    #missing_3d = 0
    #not_missing_3d = 0
    #types_missing_3d = {}
    for cell in sim.net.cells:  # local cells only, avoids MPI abort
        gid = cell.gid 
        if not pc.gid_exists(gid):
            continue  # skip non-local cells
        for sec_name, sec_dict in cell.secs.items():
            sec = sec_dict['hObj']
            sec.insert('extracellular')   # make sure mechanism is present
            sec.push()
            for seg in sec:

                if int(h.n3d()) > 0:
                    idx = int(seg.x * (h.n3d()-1))
                    x = h.x3d(idx)
                    y = h.y3d(idx)
                    z = h.z3d(idx)
                    #not_missing_3d += 1
                    
                else:
                    x = cell.tags.get('x', 0)
                    y = cell.tags.get('y', 0)
                    z = cell.tags.get('z', 0) 
                    print("no 3d points for cell gid %d sec %s; " \
                        "using cell center" % (gid, sec_name))
                    #missing_3d += 1
                    #types_missing_3d[sec_name] = True

                seg_coords.append((gid, sec, seg))
                seg_positions.append([x, y, z])
            h.pop_section()

    seg_positions = np.array(seg_positions)  # shape (N,3)

    # only axon segments were missing 3d points. Ok to default to cell center
    # just for those.
    #print("Number of segments approximated at cell centers:", missing_3d, 
    #      "out of total", missing_3d + not_missing_3d)
    #if len(types_missing_3d) > 0:
    #    print("  (types missing 3d points:", list(types_missing_3d.keys()), ")")

    # Compute distances for pointSource stim
    if field['class'] == 'pointSource':
        try:
            dx = seg_positions[:,0] - field['location'][0]
            dy = seg_positions[:,1] - field['location'][1]
            dz = seg_positions[:,2] - field['location'][2]
        except IndexError as e:
            print("shape of seg_positions: ", )
            raise KeyError(f"{seg_positions.shape}. Ensure 'location' is provided for pointSource.")
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r[r < 1e-9] = 1e-9  # avoid divide by zero

        # Apply stim radius cutoff
        if stim_radius is not None:
            mask = r <= stim_radius
            seg_coords = [c for c, m in zip(seg_coords, mask) if m]
            seg_positions = seg_positions[mask]
            r = r[mask]

        # Compute extracellular potential with distance-based decay
        r_m = np.maximum(1e-9, r * 1e-6) # convert to meters. Sigma is in mS/mm = S/m
        sigma = field.get('sigma', 0.276) # default conductivity in mS/mm
        I_A = waveform['amp'] * 1e-3 # convert mA to A
        if decay == '1/r':
            Vext_base = I_A / (4 * np.pi * sigma * r_m) * 1e3  # convert to mV
        elif decay == '1/r2':
            Vext_base = I_A / (4 * np.pi * sigma * r_m**2) * 1e3 # convert to mV
        elif decay == '1/r3':
            Vext_base = I_A / (4 * np.pi * sigma * r_m**3) * 1e3 # convert to mV
        else:
            raise ValueError("decay must be '1/r' or '1/r2'")

    elif field['class'] == 'uniform':
        raise Exception("Uniform field not yet implemented: fix the units before using")
        direction = np.array(field['direction'])
        Vext_base = waveform['amp'] * np.dot(seg_positions, direction)
    else:
        raise ValueError("Unsupported field class")

    ## Attach IClamp to segments and set amplitude
    #for (gid, sec, seg), v in zip(seg_coords, Vext):
    #    stim = h.IClamp(seg)
    #    stim.delay = waveform.get('delay', 0)
    #    stim.dur = waveform.get('dur', 1e9)
    #    stim.amp = v

    tstop = sim.cfg.duration
    dt = sim.cfg.dt

    # Build time waveform (one template vector for all segments)
    times = np.arange(0, tstop+dt, dt)
    wave = np.zeros_like(times)
    start_idx = int(waveform.get('delay',0)/dt)
    end_idx   = int((waveform.get('delay',0)+waveform.get('dur',1e9))/dt)
    wave[start_idx:end_idx] = 1.0  # normalized, scaled later

    tvec = h.Vector(times)

    # Apply to each segment, scaled by Vext_base, as the driving potential
    for (gid, sec, seg), vscale in zip(seg_coords, Vext_base):
        wvec = h.Vector(wave * vscale)
        wvec.play(seg._ref_vext[0], tvec, 1)

    print(f"Applied extracellular stimulation to {len(seg_coords)} segments.")
