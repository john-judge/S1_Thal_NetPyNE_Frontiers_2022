
try:
    from src.hVOS.camera import Camera
except ModuleNotFoundError:
    from camera import Camera  # for testing in this directory


''' Precompute morphology -> hVOS geometry mapping 
    for faster simulation runtime
    Builds a mapping W[cell_id, compart, pixel(x, y)] = Amplitude
    This is pre-PSF
'''

import numpy as np
import os

class PrecomputedCellImage:

    """ Class to hold precomputed geometry mapping for a cell 
    Input: Cell object with morphology attribute populated
    Writes a file cell_{cell_id}_pre_geometry.pkl"""

