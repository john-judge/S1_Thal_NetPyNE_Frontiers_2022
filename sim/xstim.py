""" This scripy explicitly attaches xstims to cells 
    after the network and sim have been built (after running sim.createSim()).
    It is used to add xstim functionality to the simulation, called from init.py
    
    Alternative to relying on netParams x-stim assignment, which may cause segfault
    when spatial bounding box is used."""

from neuron import h
import numpy as np
from collections import defaultdict
import os, json, glob


def export_xstim_targets(sim, field, waveform, decay='1/r', stim_radius=1000,
                         out_file='xstim_targets_rank.json'):
    """
    Discover and export extracellular stimulation targets (local to this MPI rank).

    Each rank writes its own JSON file with tuples of:
        gid, sec_name, seg_index, x, y, z, Vext_mV

    Parameters
    ----------
    sim : NetPyNE Sim object (after sim.createSim())
    field : dict, must include {'class': 'pointSource', 'location': [x,y,z], 'sigma': ...}
    waveform : dict, must include {'amp': ...} (mA injected current)
    decay : '1/r','1/r2','1/r3'
    stim_radius : cutoff radius in µm
    out_file : str, filename template; 'rank' will be replaced with rank ID
    """

    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = int(pc.nhost())

    out_file = out_file.replace('rank', f"rank{rank}")

    seg_coords = []     # (gid, sec, seg)
    seg_positions = []  # [[x,y,z],...]

    # collect local segments like attach_xstim_to_segments_mpi_safe
    for cell in sim.net.cells:
        gid = cell.gid
        if not pc.gid_exists(gid):
            continue
        for sec_name, sec_dict in cell.secs.items():
            sec = sec_dict['hObj']
            # local only
            is_local = any(sec == s for s in h.allsec())
            if not is_local:
                continue

            sec.push()
            nseg = max(1, int(getattr(sec, 'nseg', 1)))
            for seg in sec:
                seg_index = int(round(seg.x * (nseg - 1)))
                if int(h.n3d()) > 0:
                    idx = int(seg.x * (h.n3d()-1))
                    try:
                        x = float(h.x3d(idx)); y = float(h.y3d(idx)); z = float(h.z3d(idx))
                    except Exception:
                        x = float(cell.tags.get('x',0.0)); y = float(cell.tags.get('y',0.0)); z = float(cell.tags.get('z',0.0))
                else:
                    x = float(cell.tags.get('x',0.0)); y = float(cell.tags.get('y',0.0)); z = float(cell.tags.get('z',0.0))

                seg_coords.append((gid, sec_name, seg_index))
                seg_positions.append([x,y,z])
            h.pop_section()

    seg_positions = np.array(seg_positions) if len(seg_positions) > 0 else np.zeros((0,3))

    if seg_positions.shape[0] == 0:
        print(f"[Rank {rank}/{nhost}] No local segs for export")
        return []

    # distance to electrode
    ex, ey, ez = field['location']
    dx = seg_positions[:,0] - ex
    dy = seg_positions[:,1] - ey
    dz = seg_positions[:,2] - ez
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    r[r < 1e-9] = 1e-9

    mask = (r <= stim_radius)
    seg_coords = [c for c,m in zip(seg_coords,mask) if m]
    seg_positions = seg_positions[mask]
    r = r[mask]

    if len(r) == 0:
        print(f"[Rank {rank}/{nhost}] No local segs within stim_radius")
        return []

    # extracellular potential calc
    r_m = r * 1e-6
    sigma = field.get('sigma',0.276)
    I_A = float(waveform.get('amp',0.0)) * 1e-3

    if decay == '1/r':
        Vext = (I_A / (4*np.pi*sigma*r_m)) * 1e3
    elif decay == '1/r2':
        Vext = (I_A / (4*np.pi*sigma*r_m**2)) * 1e3
    elif decay == '1/r3':
        Vext = (I_A / (4*np.pi*sigma*r_m**3)) * 1e3
    else:
        raise ValueError("decay must be '1/r','1/r2','1/r3'")

    # build records
    results = []
    for (gid, sec_name, seg_index), (x,y,z), v in zip(seg_coords, seg_positions, Vext):
        results.append(dict(
            gid=int(gid),
            sec=sec_name,
            seg_index=int(seg_index),
            x=float(x), y=float(y), z=float(z),
            Vext=float(v)
        ))

    # write file
    with open(out_file,'w') as f:
        json.dump(results,f,indent=2)

    print(f"[Rank {rank}/{nhost}] Exported {len(results)} targets -> {out_file}")
    return results


def load_xstim_targets_and_add_stims(netParams, stim_dir='xstim/', 
                                     stim_pattern='rank*_xstim_targets.json',
                                     stim_delay=75, stim_dur=4, stim_amp_factor=1.0):
    """
    Load extracellular target files from multiple MPI ranks and add IClamp stims in netParams.

    Parameters
    ----------
    netParams : NetParams object (NetPyNE)
    stim_dir : str, directory containing the exported JSON files
    stim_pattern : str, filename glob pattern
    stim_delay : float, ms
    stim_dur : float, ms
    stim_amp_factor : scaling factor to convert Vext (mV) into IClamp.amp (nA)

    Returns
    -------
    all_targets : list of dicts with gid, sec, seg_index, Vext, etc.
    """

    all_targets = []

    files = glob.glob(os.path.join(stim_dir, stim_pattern))
    if len(files) == 0:
        print(f"[Loader] No stim files found in {stim_dir} matching {stim_pattern}")
        return []

    for fpath in files:
        with open(fpath, 'r') as f:
            data = json.load(f)
        all_targets.extend(data)
        print(f"[Loader] Loaded {len(data)} targets from {os.path.basename(fpath)}")

    # Add stims to netParams
    stim_count = 0
    for tgt in all_targets:
        gid = tgt['gid']
        sec = tgt['sec']
        seg_index = tgt['seg_index']
        Vext = tgt['Vext']  # mV

        stim_name = f"xstim_{gid}_{sec}_{seg_index}_{stim_count}"
        stim_dict = {
            'source': 'IClamp',
            'sec': sec,
            'loc': seg_index,  # segment index treated as loc (approx mapping)
            'delay': stim_delay,
            'dur': stim_dur,
            # convert Vext -> current; factor allows tuning
            'amp': Vext * stim_amp_factor
        }

        # attach stim to correct population via gid
        netParams.stims[stim_name] = stim_dict
        stim_count += 1

    print(f"[Loader] Added {stim_count} IClamp stims into netParams")
    return all_targets


# default simple mapper
def _default_type_map(sec_name):
    s = sec_name.lower()
    if 'soma' in s:
        return 'soma'
    if 'axon' in s:
        return 'axon'
    if 'apic' in s or 'apical' in s:
        return 'apic'
    return 'dend'

def attach_xstim_to_segments_mpi_safe(sim, field, waveform, decay='1/r', stim_radius=1000):
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

    seg_coords = []     # (cell_gid, sec, seg) tuples
    seg_positions = []  # corresponding x,y,z

    #missing_3d = 0
    #not_missing_3d = 0
    #types_missing_3d = {}
    for cell in sim.net.cells:  # local cells only, avoids MPI abort
        gid = cell.gid 
        for sec_name, sec_dict in cell.secs.items():
            sec = sec_dict['hObj']
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
        
        res = estimate_coupling_resistance_by_segt(sim, default_Rm_ohm_cm2=20000.0)
        # res is a dict like {'soma': {...}, 'axon': {...}, ...}
        coupling_by_type = {t: res[t]['median_MOhm_approx'] or res[t]['mean_MOhm'] for t in res}

    elif field['class'] == 'uniform':
        raise Exception("Uniform field not yet implemented: fix the units before using")
        direction = np.array(field['direction'])
        Vext = waveform['amp'] * np.dot(seg_positions, direction)
    else:
        raise ValueError("Unsupported field class")

    

    # ======= Diagnostics by section type ==========

    amps_by_type = defaultdict(list)
    counts_by_type = defaultdict(int)

    for (gid, sec, seg), V_ in zip(seg_coords, Vext):
        sec_type = _default_type_map(sec.name())
        I_nA = float(V_ / coupling_by_type[sec_type])
        amps_by_type[sec_type].append(I_nA)
        counts_by_type[sec_type] += 1

    print("=== XStim diagnostics ===")
    print(f"Electrode @ {field['location']} stim_radius={stim_radius} µm, "
          f"delay={waveform.get('del', 50)} ms, dur={waveform.get('dur', 1e9)} ms")
    for t in ['soma','dend','apic','axon']:
        if counts_by_type[t]:
            arr = np.array(amps_by_type[t])
            print(f"{t:5s}: n={counts_by_type[t]:4d}, "
                f"median={np.median(arr):.3g} nA, "
                f"min={arr.min():.3g}, max={arr.max():.3g}")
        else:
            print(f"{t:5s}: n=0")


    # Attach IClamp to segments and set amplitude
    for (gid, sec, seg), V_ in zip(seg_coords, Vext):
        # I_nA = V_mV / R_MOhm
        sec_type = _default_type_map(sec.name())
        #print("mapped section type:", sec_type, "for gid", gid, "sec", sec.name())
        Iamps_nA = (V_ / coupling_by_type[sec_type]).astype(float)
        stim = h.IClamp(seg)
        stim.delay = waveform.get('del', 50)
        stim.dur = waveform.get('dur', 1e9)
        stim.amp = float(Iamps_nA)
        #print(f"Attached IClamp to gid {gid}, sec {sec.name()}, "
        #      f"amp={Iamps_nA} nA")

    print(f"Applied extracellular stimulation to {len(seg_coords)} segments.")


def estimate_coupling_resistance_by_segt(sim,
                                         default_Rm_ohm_cm2=20000.0,
                                         type_map=None,
                                         return_per_segment=False):
    """
    Estimate effective input/coupling resistance (MΩ) per segment-type (soma, axon, dend, apic).

    Parameters
    ----------
    sim : NetPyNE sim object
    default_Rm_ohm_cm2 : float
        Fallback specific membrane resistance (Ohm * cm^2) when g_pas not present.
        Typical order: 1e4 - 1e5 Ohm*cm^2 (tune for your model).
    type_map : callable(sec_name: str) -> str
        Optional function to map a section name to a type string in {'soma','axon','dend','apic'}.
        If None a heuristic is used: 'soma' in name -> soma; 'axon' -> axon; 'apic'/'apical' -> apic;
        otherwise 'dend'.
    return_per_segment : bool
        If True, returns per-segment list of 
        (key, area_cm2, Rm_ohm_cm2, R_input_Mohm).
    Returns
    -------
    type_stats : dict
        keys are segment types; values are dicts with keys:
           'count', 'mean_MOhm', 'median_MOhm', 'values_MOhm' (if return_per_segment True)
    """
    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = int(pc.nhost())

    map_fn = type_map or _default_type_map

    # iterate only over local sections (h.allsec())
    local_secs = list(h.allsec())

    seg_records = []  # list of (type, gid, sec_name, seg_index, area_cm2, Rm_ohm_cm2, R_input_MOhm)

    for sec in local_secs:
        sec_name = sec.name() if hasattr(sec, 'name') else str(sec)
        # try to find any associated cell gid via sim.net.cells (cheap-ish mapping)
        # build reverse map once
    # Build a mapping from sec object id to gid/section name as reported in sim.net.cells
    sec_to_gidname = {}
    for cell in sim.net.cells:
        gid = int(cell.gid)
        for sname, sdict in cell.secs.items():
            sobj = sdict.get('hObj')
            if sobj is not None:
                sec_to_gidname[id(sobj)] = (gid, sname)

    for sec in local_secs:
        sec_name = sec.name() if hasattr(sec, 'name') else str(sec)
        gid, reported_secname = sec_to_gidname.get(id(sec), (None, sec_name))
        # geometry
        nseg = max(1, int(getattr(sec, 'nseg', 1)))
        L = float(getattr(sec, 'L', 0.0))   # µm
        # segment length = L / nseg
        seg_len_um = max(1e-12, L / float(nseg))
        # section diameter: try seg.diam if available, else sec.diam
        # we'll compute for each segment individually below
        sec_diam_um = float(getattr(sec, 'diam', 0.0))

        sec.push()
        for seg in sec:
            try:
                seg_diam_um = float(getattr(seg, 'diam', sec_diam_um))
            except Exception:
                seg_diam_um = sec_diam_um
            # fallback for zero diam sections: skip (area 0)
            if seg_diam_um <= 0 or seg_len_um <= 0:
                area_cm2 = 1e-20  # tiny non-zero to avoid div-by-zero (will produce very large R)
            else:
                # cylindrical lateral surface area for small segment approx:
                # area_um2 = pi * diam_um * seg_len_um   (approx, neglecting end caps)
                area_um2 = np.pi * seg_diam_um * seg_len_um
                area_cm2 = float(area_um2) * 1e-8  # convert µm^2 -> cm^2

            # Determine specific membrane resistance Rm (Ohm * cm^2)
            # prefer using g_pas if present on the section (units S/cm^2)
            Rm_ohm_cm2 = None
            # attempt to read sec.g_pas (works if pas inserted)
            try:
                g_pas = getattr(sec, 'g_pas', None)
                if g_pas is not None and float(g_pas) > 0.0:
                    Rm_ohm_cm2 = 1.0 / float(g_pas)
            except Exception:
                Rm_ohm_cm2 = None

            # fallback to default
            if Rm_ohm_cm2 is None or not np.isfinite(Rm_ohm_cm2):
                Rm_ohm_cm2 = float(default_Rm_ohm_cm2)

            # estimate input resistance of this segment (Ohm)
            R_input_ohm = Rm_ohm_cm2 / area_cm2  # Ohm
            R_input_MOhm = R_input_ohm / 1e6

            # determine type from section name (user-supplied map_fn can override)
            sec_type = map_fn(reported_secname if reported_secname is not None else sec_name)

            seg_index = int(round(seg.x * (nseg - 1)))
            seg_key = (gid, reported_secname, seg_index)

            seg_records.append((sec_type, seg_key, area_cm2, Rm_ohm_cm2, R_input_MOhm))
        h.pop_section()

    # Gather records across ranks to rank 0
    # We'll serialize minimal tuples to Python objects via pickle-friendly lists and use allreduce/bcast.
    # Simpler: collect per-rank stats and then use pc.allreduce to gather counts and means.
    # But to get full per-seg lists, we will use a simple approach: send lengths and then each rank writes to a temp file.
    # For simplicity and robustness here we'll aggregate only numeric statistics globally using allreduce.

    # local aggregation by type
    stats_local = {}
    for rec in seg_records:
        sec_type, seg_key, area_cm2, Rm_ohm_cm2, R_input_MOhm = rec
        if sec_type not in stats_local:
            stats_local[sec_type] = {'vals': []}
        stats_local[sec_type]['vals'].append(R_input_MOhm)

    # Compute local counts and sums for all types encountered across ranks
    all_types = list(stats_local.keys())
    # Prepare vectors for reduction: for simplicity, consider canonical types
    canonical_types = ['soma', 'axon', 'dend', 'apic']
    local_counts = [len(stats_local.get(t, {'vals': []})['vals']) for t in canonical_types]
    local_sums = [float(np.sum(stats_local.get(t, {'vals': []})['vals'])) for t in canonical_types]

    # reduce to global
    global_counts = [int(pc.allreduce(c, 1)) for c in local_counts]
    global_sums = [float(pc.allreduce(s, 1.0)) for s in local_sums]

    # For medians, we'll collect per-rank medians and approximate global median by pooling small lists:
    # Here we gather per-rank value arrays using a simple rank-0 gather via pc.py_alltoall? To keep dependency-free,
    # we will gather medians per rank and compute weighted median approx. For strict correctness, user can run with rank 0 collecting lists.

    # compute local medians
    local_medians = [float(np.median(stats_local.get(t, {'vals': [np.nan]})['vals'])) if len(stats_local.get(t, {'vals': []})['vals'])>0 else float('nan') for t in canonical_types]
    # reduce medians by averaging medians (approx)
    global_medians_approx = [float(pc.allreduce(m, 1.0) / float(nhost)) for m in local_medians]

    # Build result dict (rank 0 will print)
    result = {}
    for i, t in enumerate(canonical_types):
        cnt = global_counts[i]
        s = global_sums[i]
        mean = (s / cnt) if cnt > 0 else float('nan')
        median_approx = global_medians_approx[i]
        result[t] = {'count': int(cnt), 'mean_MOhm': float(mean), 'median_MOhm_approx': float(median_approx)}

    if rank == 0:
        print("Estimated coupling/input resistance per segment type (MΩ) — APPROXIMATE")
        for t in canonical_types:
            r = result[t]
            print(f"  {t:5s}: count={r['count']:4d}, mean={r['mean_MOhm']:.3g} MΩ, median_approx={r['median_MOhm_approx']:.3g} MΩ")

    # optionally return per-segment list (local only) if requested
    if return_per_segment:
        return seg_records  # local-only list of tuples
    return result


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
        coupling_resist_Mohm = 100.0
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
