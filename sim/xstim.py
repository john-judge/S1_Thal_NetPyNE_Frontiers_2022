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



"""
MPI-safe Vext precompute + attach (Approach 3).

Usage:
    attach_vext_precompute_and_attach(sim, field, waveform,
                                      decay='1/r', stim_radius=100,
                                      tmpfile='/tmp/vext_precompute.pkl')

- Rank 0 computes per-segment scaling factors (Vext_base) and a single time/wave template,
  writes them to `tmpfile`.
- All ranks wait at a barrier, then load `tmpfile`.
- Each rank attaches a Vector.play to seg._ref_vext[0] only for segments whose cell GID
  is local on that rank (pc.gid_exists(cell.gid) == True).

Requirements:
- Shared filesystem accessible by all MPI ranks.
- sim is NetPyNE sim object with sim.net.cells available.
- waveform: dict with keys 'amp' (mA), optional 'delay' (ms), optional 'dur' (ms).
"""

import os
import pickle
from neuron import h
import numpy as np

def _seg_key(gid, sec_name, seg_index):
    return (int(gid), str(sec_name), int(seg_index))

def attach_xstim_to_segments_mpi_safe(sim, field, waveform,
                                      decay='1/r', stim_radius=100,
                                      tmpfile='/tmp/vext_precompute_v1.pkl',
                                      require_shared_fs=True):
    """
    MPI-safe Vext precompute on node 0 + attach in parallel

    - Rank 0 computes per-segment scaling factors (Vext_base) and a single time/wave template,
    writes them to `tmpfile`.
    - All ranks wait at a barrier, then load `tmpfile`.
    - Each rank attaches a Vector.play to seg._ref_vext[0] only for segments whose cell GID
    is local on that rank (pc.gid_exists(cell.gid) == True).

    Requirements:
    - Shared filesystem accessible by all MPI ranks.
    - sim is NetPyNE sim object with sim.net.cells available.
    - waveform: dict with keys 'amp' (mA), optional 'delay' (ms), optional 'dur' (ms).
    """
    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = int(pc.nhost())

    # --- Rank 0: precompute everything and write to disk ---
    if rank == 0:
        seg_keys = []        # parallel list of keys for ordering
        seg_positions = []   # shape (N,3)

        # collect positions for ALL cells/sections/segments
        for cell in sim.net.cells:           # sim.net.cells is iterable of cell objects
            gid = cell.gid
            for sec_name, sec_dict in cell.secs.items():
                sec = sec_dict['hObj']
                # Ensure extracellular mechanism exists so later attach works
                try:
                    sec.insert('extracellular')
                except Exception:
                    # ignore insertion errors; remains best-effort
                    pass

                # for segment coordinate calculation we rely on sec.nseg
                nseg = getattr(sec, 'nseg', 1)
                sec.push()
                for seg in sec:
                    # seg.x in [0,1], compute integer seg_index in [0, nseg-1]
                    seg_index = int(round(seg.x * (max(nseg,1)-1)))
                    # get 3d coordinates if available for this section
                    if int(h.n3d()) > 0:
                        # compute index into 3d points for section (approximation)
                        idx = int(seg.x * (h.n3d()-1))
                        # If the section contains 3D points, x3d() refers to the currently pushed section
                        try:
                            x = float(h.x3d(idx))
                            y = float(h.y3d(idx))
                            z = float(h.z3d(idx))
                        except Exception:
                            # fallback to cell-level tags if x3d fails
                            x = float(cell.tags.get('x', 0.0))
                            y = float(cell.tags.get('y', 0.0))
                            z = float(cell.tags.get('z', 0.0))
                    else:
                        x = float(cell.tags.get('x', 0.0))
                        y = float(cell.tags.get('y', 0.0))
                        z = float(cell.tags.get('z', 0.0))

                    key = _seg_key(gid, sec_name, seg_index)
                    seg_keys.append(key)
                    seg_positions.append([x, y, z])
                h.pop_section()

        seg_positions = np.array(seg_positions) if len(seg_positions) > 0 else np.zeros((0,3))

        # If no segments found, still create an empty payload
        if seg_positions.shape[0] == 0:
            payload = {
                'times': np.array([0.0]),
                'wave_template': np.array([0.0]),
                'scales': {},    # empty
                'seg_keys': []
            }
            # write file and barrier then return (no attachments)
            with open(tmpfile, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("[Rank 0] No segments found. Wrote empty payload.")
            pc.barrier()
            return

        # compute distances to pointSource
        if field.get('class', 'pointSource') != 'pointSource':
            raise NotImplementedError("Only pointSource implemented in this helper.")

        loc = field.get('location', [0.0, 0.0, 0.0])
        dx = seg_positions[:,0] - loc[0]
        dy = seg_positions[:,1] - loc[1]
        dz = seg_positions[:,2] - loc[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r[r < 1e-9] = 1e-9

        # apply stim radius cutoff if requested
        if stim_radius is not None:
            mask = r <= stim_radius
            # filter lists
            seg_keys = [k for k, m in zip(seg_keys, mask) if m]
            r = r[mask]
            seg_positions = seg_positions[mask]

        # compute Vext scale per segment (mV amplitude for unit waveform)
        r_m = r * 1e-6  # microns -> meters
        sigma = field.get('sigma', 0.276)  # mS/mm as in your code
        I_A = float(waveform.get('amp', 0.0)) * 1e-3  # mA -> A

        if decay == '1/r':
            Vext_base_vals = (I_A / (4.0 * np.pi * sigma * r_m)) * 1e3  # in mV
        elif decay == '1/r2':
            Vext_base_vals = (I_A / (4.0 * np.pi * sigma * (r_m**2))) * 1e3
        elif decay == '1/r3':
            Vext_base_vals = (I_A / (4.0 * np.pi * sigma * (r_m**3))) * 1e3
        else:
            raise ValueError("decay must be one of '1/r','1/r2','1/r3'")

        # Build time vector and normalized wave template (0/1)
        tstop = float(sim.cfg.duration)
        dt = float(sim.cfg.dt)
        times = np.arange(0.0, tstop + dt, dt)
        wave = np.zeros_like(times)
        start_idx = int(round(float(waveform.get('delay', 0.0)) / dt))
        end_idx = int(round((float(waveform.get('delay', 0.0)) + float(waveform.get('dur', 1e9))) / dt))
        # clamp
        start_idx = max(0, min(start_idx, len(wave)-1))
        end_idx = max(0, min(end_idx, len(wave)))
        wave[start_idx:end_idx] = 1.0

        # Build scales dict keyed by seg_keys order
        scales = {}
        for key, vscale in zip(seg_keys, Vext_base_vals):
            scales[key] = float(vscale)

        payload = {
            'times': times,
            'wave_template': wave,
            'scales': scales,
            'seg_keys': seg_keys
        }

        # write to tmpfile for other ranks to read
        # Ensure directory exists
        tmpdir = os.path.dirname(tmpfile) or '/tmp'
        if not os.path.isdir(tmpdir):
            os.makedirs(tmpdir, exist_ok=True)

        with open(tmpfile, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[Rank 0] Precomputed waveforms for {len(scales)} segments and wrote to {tmpfile}.")

    # --- All ranks wait until payload is ready on disk ---
    pc.barrier()

    # --- All ranks load payload from disk ---
    if not os.path.exists(tmpfile):
        # Could not find file; abort gracefully
        raise RuntimeError(f"[Rank {rank}] Expected precompute file {tmpfile} not found after barrier.")

    with open(tmpfile, 'rb') as f:
        payload = pickle.load(f)

    times = np.array(payload['times'])
    wave_template = np.array(payload['wave_template'])
    scales = payload.get('scales', {})

    # Create NEURON Vectors for times and reuse a single tvec across plays
    tvec = h.Vector(list(times))

    attached_count = 0
    missing_local = 0

    # Each rank loops over its local cells and their sections; attach if key present
    for cell in sim.net.cells:
        if not pc.gid_exists(cell.gid):
            continue  # skip non-local cells entirely

        gid = cell.gid
        for sec_name, sec_dict in cell.secs.items():
            sec = sec_dict['hObj']
            # ensure extracellular exists
            try:
                sec.insert('extracellular')
            except Exception:
                pass

            nseg = getattr(sec, 'nseg', 1)
            sec.push()
            for seg in sec:
                seg_index = int(round(seg.x * (max(nseg,1)-1)))
                key = _seg_key(gid, sec_name, seg_index)
                if key in scales:
                    vscale = scales[key]
                    # build waveform vector for this segment
                    # multiply wave_template by vscale (mV)
                    wdata = (wave_template * float(vscale)).tolist()
                    wvec = h.Vector(wdata)
                    # Play into seg._ref_vext[0] (should exist because we inserted extracellular above)
                    try:
                        wvec.play(seg._ref_vext[0], tvec, 1)
                        attached_count += 1
                    except Exception as e:
                        # Could not attach (seg may not actually be local) — count and continue
                        missing_local += 1
                # else: no precomputed scale for this seg (outside stim radius etc.)
            h.pop_section()

    print(f"[Rank {rank}] Attached waveforms to {attached_count} local segments "
          f"(failed attachments {missing_local}).")

    # final barrier to sync before running simulation
    pc.barrier()

    # Optionally: cleanup on rank 0 (keep or remove file)
    if rank == 0:
        # if user wants to keep file for debugging, comment this out
        try:
            os.remove(tmpfile)
        except Exception:
            pass

    return attached_count
