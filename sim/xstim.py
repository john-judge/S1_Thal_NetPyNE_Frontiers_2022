""" This scripy explicitly attaches xstims to cells 
    after the network and sim have been built (after running sim.createSim()).
    It is used to add xstim functionality to the simulation, called from init.py
    
    Alternative to relying on netParams x-stim assignment, which may cause segfault
    when spatial bounding box is used."""

from neuron import h

def attach_xstim_to_segments(sim, stim_name, x0, y0, z0, stim_radius, stim_params):
    """
    Attach a single NetStim/XStim source to all segments within a cubic region.

    Args:
        sim: NetPyNE sim object (after sim.createSim())
        stim_name: name of the stim (used for reference)
        x0, y0, z0: center coordinates of the target cube
        stim_radius: half-width of cube around center
        stim_params: dict of stim parameters from netParams.stimSourceParams
    """
    # Create the NetStim or XStim source
    stim_type = stim_params.get('type', 'NetStim')
    if stim_type == 'NetStim':
        stim = h.NetStim()
        for param, value in stim_params.items():
            if param != 'type':
                setattr(stim, param, value)
    else:
        # Replace with your custom XStim mechanism
        stim = getattr(h, stim_type)()

    target_segs = []

    # Collect all segments within bounds
    for gid, cell in sim.net.allCells.items():
        if not hasattr(cell, 'secs'):
            continue
        for sec_name, sec_dict in cell.secs.items():
            sec = sec_dict['hSec']
            for seg in sec:
                try:
                    x = seg.x3d(seg.x)
                    y = seg.y3d(seg.x)
                    z = seg.z3d(seg.x)
                    """if (x0 - stim_radius <= x <= x0 + stim_radius and
                        y0 - stim_radius <= y <= y0 + stim_radius and
                        z0 - stim_radius <= z <= z0 + stim_radius):"""
                    # sphere of radius stim_radius
                    if ((x - x0)**2 + (y - y0)**2 + (z - z0)**2) <= stim_radius**2:
                        # This segment is within the target region
                        target_segs.append(seg)
                except Exception:
                    continue

    # Connect the stim to all target segments
    for seg in target_segs:
        nc = h.NetCon(stim, seg(0.5))  # midpoint
        nc.weight[0] = 0.1  # adjust as needed
        nc.delay = 0

    print(f"[{stim_name}] Attached to {len(target_segs)} segments.")

