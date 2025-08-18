""" This scripy explicitly attaches xstims to cells 
    after the network and sim have been built (after running sim.createSim()).
    It is used to add xstim functionality to the simulation, called from init.py
    
    Alternative to relying on netParams x-stim assignment, which may cause segfault
    when spatial bounding box is used."""

from neuron import h
import numpy as np


def attach_xstim_to_segments(sim, field, waveform, decay='1/r', stim_radius=100,
                             stim_mech='IClamp'):
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
    stim_mech : str
        Mechanism to use for stimulation ('IClamp' or 'Vext').
    """

    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = int(pc.nhost())

    seg_coords = []     # (cell_gid, sec, seg) tuples
    seg_positions = []  # corresponding x,y,z

    # small helper to test if a section object is local to this process
    def section_is_local(sec):
        # iterate over allsec on this process and compare object identity
        for s in h.allsec():
            if s == sec:
                return True
        return False
    # Ensure every LOCAL section on each rank has extracellular inserted ---
    # iterate h.allsec() (these are the sections instantiated on this rank)
    local_secs = list(h.allsec())
    if stim_mech == 'Vext':
        for sec in local_secs:
            try:
                sec.insert('extracellular')
            except Exception:
                # ignore if already present or insertion fails, but keep going
                pass

    # collect only segments that are local to THIS rank
    for cell in sim.net.cells:
        gid = cell.gid
        # require that the cell gid is local to this rank (fast guard)
        if not pc.gid_exists(gid):
            continue
        for sec_name, sec_dict in cell.secs.items():
            sec = sec_dict['hObj']
            if not section_is_local(sec):
                # section object referenced by this cell is not present on this rank
                continue

            # Ensure extracellular mechanism exists on this local section
            try:
                sec.insert('extracellular')
            except Exception:
                # insertion sometimes fails if mechanism 
                #  already present or other issues;
                # below catches missing _ref_vext anyway
                pass

            sec.push()
            nseg = max(1, int(getattr(sec, 'nseg', 1)))
            for seg in sec:
                # compute approximate seg index and coordinates
                seg_index = int(round(seg.x * (nseg - 1)))
                if int(h.n3d()) > 0:
                    idx = int(seg.x * (h.n3d()-1))
                    try:
                        x = float(h.x3d(idx)); y = float(h.y3d(idx)); z = float(h.z3d(idx))
                    except Exception:
                        x = float(cell.tags.get('x', 0.0)); y = float(cell.tags.get('y', 0.0)); z = float(cell.tags.get('z', 0.0))
                else:
                    x = float(cell.tags.get('x', 0.0)); y = float(cell.tags.get('y', 0.0)); z = float(cell.tags.get('z', 0.0))

                # final safety check: does this segment expose _ref_vext on this process?
                # getattr will return something or None; do not dereference here (that can be done later)
                if not hasattr(seg, '_ref_vext'):
                    # _ref_vext not present — skip this seg
                    continue

                seg_coords.append((gid, sec, seg))
                seg_positions.append([x, y, z])
            h.pop_section()

    seg_positions = np.array(seg_positions) if len(seg_positions) > 0 else np.zeros((0,3))

    if seg_positions.shape[0] == 0:
        print(f"[Rank {rank}/{nhost}] No local segments to attach stim to.")
        return 0

    # Compute distance-based scaling (local subset only)
    if field.get('class','pointSource') != 'pointSource':
        raise NotImplementedError("Only pointSource implemented.")

    dx = seg_positions[:,0] - field['location'][0]
    dy = seg_positions[:,1] - field['location'][1]
    dz = seg_positions[:,2] - field['location'][2]
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    r[r < 1e-9] = 1e-9

    if stim_radius is not None:
        mask = r <= stim_radius
        seg_coords = [c for c,m in zip(seg_coords, mask) if m]
        r = r[mask]

    if len(r) == 0:
        print(f"[Rank {rank}/{nhost}] No local segments within stim_radius.")
        return 0

    r_m = r * 1e-6
    sigma = field.get('sigma', 0.276)
    I_A = float(waveform.get('amp', 0.0)) * 1e-3

    if decay == '1/r':
        Vext_base = (I_A / (4.0 * np.pi * sigma * r_m)) * 1e3
    elif decay == '1/r2':
        Vext_base = (I_A / (4.0 * np.pi * sigma * (r_m**2))) * 1e3
    elif decay == '1/r3':
        Vext_base = (I_A / (4.0 * np.pi * sigma * (r_m**3))) * 1e3
    else:
        raise ValueError("decay must be '1/r','1/r2','1/r3'")
    
    if stim_mech == 'Vext':

        # Build single time vector per rank and normalized template
        tstop = float(sim.cfg.duration)
        dt = float(sim.cfg.dt)
        times = np.arange(0.0, tstop + dt, dt)
        wave_template = np.zeros_like(times)
        start_idx = int(round(float(waveform.get('delay',0.0)) / dt))
        end_idx   = int(round((float(waveform.get('delay',0.0)) + float(waveform.get('dur', 1e9))) / dt))
        start_idx = max(0, min(start_idx, len(wave_template)-1))
        end_idx = max(0, min(end_idx, len(wave_template)))
        wave_template[start_idx:end_idx] = 1.0
        tvec = h.Vector(list(times))

        # store wvecs to avoid GC if requested
        if not hasattr(sim.net, '_xstim_wvecs'):
            sim.net._xstim_wvecs = []

        attached = 0
        failed = 0

        # Attach local wvecs only — wrap play in try/except to skip any runtime oddities
        for (gid, sec, seg), vscale in zip(seg_coords, Vext_base):
            try:
                # Double-check segment still local via comparing to allsec (defensive)
                found_local = False
                for s in h.allsec():
                    if s == sec:
                        found_local = True
                        break
                if not found_local:
                    # section lost from local list for whatever reason — skip
                    failed += 1
                    continue

                # dereference seg._ref_vext now (safe because sec is local)
                ref = seg._ref_vext[0]  # this should not segfault for truly-local seg
                wvec = h.Vector((wave_template * float(vscale)).tolist())
                wvec.play(seg._ref_vext[0], tvec, 1)

                sim.net._xstim_wvecs.append(wvec)
                attached += 1
            except Exception as e:
                # log and continue
                failed += 1
                # Optionally print debug for first few failures
                if failed <= 5:
                    print(f"[Rank {rank}] Failed to attach to gid {gid} sec {sec.name()} seg.x={seg.x}: {e}")
                continue
    elif stim_mech == 'IClamp':
        # Convert Vext (mV) -> Iamp (nA) using coupling resistance (MΩ)
        coupling = float(coupling_resist_Mohm)
        # I_nA = V_mV / R_MOhm
        Iamps_nA = (Vext_base / coupling).astype(float)

        # Prepare storage to keep references if requested
        if not hasattr(sim.net, '_xstim_iclamps'):
            sim.net._xstim_iclamps = []

        attached = 0
        failed = 0

        # Attach IClamp to each local segment with computed amplitude (nA)
        for (gid, sec, seg), iamp in zip(seg_coords, Iamps_nA):
            try:
                # attach IClamp. IClamp accepts a segment object argument.
                stim = h.IClamp(seg)
                stim.delay = float(waveform.get('delay', 0.0))  # delay in ms
                stim.dur = float(waveform.get('dur', 1e9))  # duration in ms
                stim.amp = float(iamp)  # NEURON expects nA
                sim.net._xstim_iclamps.append(stim)
                attached += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"[Rank {rank}] Failed attaching IClamp to gid {gid} sec {sec.name()} seg.x={seg.x}: {e}")
                continue

    print(f"[Rank {rank}/{nhost}] Attached {attached} local segments (failed {failed}).")
    return attached
