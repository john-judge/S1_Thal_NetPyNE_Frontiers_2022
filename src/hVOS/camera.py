import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.measure import block_reduce


class Camera:
    """ A class that draws a camera view of each cell in the network,
     showing the cell's morphology and the intensity of the hVOS signal,
     Then renders frames at every time step to create a video of the 
     network activity seen through the optical traces. """
    
    def __init__(self, target_cells, morphologies, time, 
                 fov_center=(0, 0, 0),  # um
                 camera_width=80,  # pixels
                 camera_height=80,  # pixels
                 camera_resolution=6.0,  # um/pixel
                 render_factor=10,  # render this many times more resolution before downscaling
                 camera_angle='coronal', 
                 psf=None,
                 psf_resolution=1.0):  # um/pixel
        self.target_cells = target_cells
        self.morphologies = morphologies
        self.time = time
        self.fov_center = fov_center
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_resolution = camera_resolution
        self.render_factor = render_factor
        self.camera_angle = camera_angle  # coronal: see all layers in y-z plane

        self.recording = np.zeros((len(self.time), self.camera_width, self.camera_height))
        self.psf = psf
        self.psf_resolution = psf_resolution
        self.rescale_psf()
        self.orient_psf_to_camera()
        
    def rescale_psf(self):
        ''' 
        The PSF is a 3D array, with the center of the PSF at the center of the array.
        Its units are in um, while the camera view is in pixels,
        which is (self.camera_resolution) um per pixel by default.
        So the PSF needs to be rescaled to match the camera view's resolution.
        This is done by downsampling the PSF to the camera view's resolution.
        '''
        if self.psf is None:
            return
        if self.camera_resolution == self.psf_resolution:
            return
        if self.camera_resolution < self.psf_resolution:
            raise ValueError("Camera resolution must be greater than PSF resolution.")
        # downsample the PSF to the camera view's resolution
        # downsample at a factor of (camera_resolution / psf_resolution)
        downsample_factor = int(self.camera_resolution / self.psf_resolution)
        # downsample the PSF along all three dimensions. Use binning to downsample
        # along each dimension.
        self.psf = block_reduce(self.psf, (downsample_factor, downsample_factor, downsample_factor), np.mean)
        # normalize the PSF so that its sum is 1
        self.psf /= np.sum(self.psf)

    def orient_psf_to_camera(self):
        """ Orient the PSF to the camera view. """
        if self.psf is None:
            return
        if self.camera_angle == 'coronal':
            # the camera's x-axis is the same as the PSF's z-axis
            # the camera's y-axis is the same as the PSF's x-axis
            # the camera's z-axis is the same as the PSF's y-axis
            # rotate the PSF by 90 degrees around the y-axis
            self.psf = np.rot90(self.psf, axes=(0, 2))
            return
        elif self.camera_angle == 'sagittal':
            raise NotImplementedError("Sagittal view is not implemented.")
        else:
            raise NotImplementedError("Only coronal view is implemented.")

    def get_recording(self):
        return self.recording

    def draw_single_frame(self, time_step):
        """ Draw the camera view of the network at a single time step. 
            Loop over all segments of compartments of all cells in the target population.
            use the optical trace and PSF to determine which pixels receive 
            illumination from that segment. 
            """
        for cell in self.target_cells:
            self._draw_cell(cell, time_step)
        plt.show()

    def _draw_cell(self, cell, time_step):
        """ Draw the camera view of a single cell at a single time step. """
        x_soma, y_soma, z_soma = cell.get_soma_position()
        structure = cell.get_morphology().get_structure()
        
        # follow the morphology of the cell, pulling optical trace from each compartment
        # and drawing it on the camera view
        for compartment in structure:
            intensity_value = cell.get_optical_trace("V" + compartment)[time_step]
            for segment_id in structure[compartment]:
                print(structure[compartment])
                
                # draw the optical trace on the camera view
                self._draw_segment(structure[compartment][segment_id], 
                                   intensity_value,
                                   x_soma, y_soma, z_soma, time_step)
            # get the morphology of the cell

    def _draw_segment(self, segment, intensity_value, x_soma, y_soma, z_soma, t):
        """ draw the segment of the cell on the camera view. """
        x_seg_prox = float(segment['proximal']['x']) + x_soma
        y_seg_prox = float(segment['proximal']['y']) + y_soma
        z_seg_prox = float(segment['proximal']['z']) + z_soma
        diam_seg_prox = float(segment['proximal']['diameter'])
        x_seg_dist = float(segment['distal']['x']) + x_soma
        y_seg_dist = float(segment['distal']['y']) + y_soma
        z_seg_dist = float(segment['distal']['z']) + z_soma
        diam_seg_dist = float(segment['distal']['diameter'])

        # calculate lateral membrane surface area (um^2) of the segment
        # assuming the segment is a conical frustum. 
        # LA = pi * (r1 + r2) * √((r1 - r2)2 + h2)
        height = np.sqrt((x_seg_dist - x_seg_prox) ** 2 +
                         (y_seg_dist - y_seg_prox) ** 2 +
                         (z_seg_dist - z_seg_prox) ** 2)
        area_lateral = np.pi * (diam_seg_prox + diam_seg_dist) * \
                       np.sqrt((diam_seg_prox - diam_seg_dist) ** 2 + height ** 2)
        
        x_center = (x_seg_prox + x_seg_dist) / 2
        y_center = (y_seg_prox + y_seg_dist) / 2
        z_center = (z_seg_prox + z_seg_dist) / 2

        # so this is a point at (x_center, y_center, z_center) with intensity_value
        # and weighted by area_lateral. Seen from the y-z plane, draw its projection
        # onto the camera view as described by the PSF.
        self._draw_weighted_point(x_center, y_center, z_center, intensity_value * area_lateral, t)

    def map_point_to_pixel(self, x, y, z):
        """ Given the 3D coordinates of a point, return the pixel location
            in the camera view. 
            The camera is pointed at self.fov_center, and extends 
                    self.camera_width / 2 * self.camera_resolution
            to the left and right of the center, and 
                    self.camera_height / 2 * self.camera_resolution 
            above and below the center.
            (0, 0) is the top-left corner of the camera view

            """
        if self.camera_angle == 'coronal':
            y_dist_to_center = y - self.fov_center[1]
            z_dist_to_center = z - self.fov_center[2]
            # convert to pixels
            y_dist_to_center /= self.camera_resolution
            z_dist_to_center /= self.camera_resolution
            # invert y because the camera view has y increasing downwards
            y = -y_dist_to_center
            z = z_dist_to_center
            # pixels are centered at (self.camera_width / 2, self.camera_height / 2)
            # i.e. 0, 0 -> 40, 40
            i = int(y + self.camera_width / 2)
            j = int(z + self.camera_height / 2)
            
        elif self.camera_angle == 'sagittal':
            raise NotImplementedError("Sagittal view is not implemented.")
        else:
            raise NotImplementedError("Only coronal view is implemented.")
        return i, j

    def _draw_weighted_point(self, x, y, z, weight, t):
        """ Draw a point on the camera view of this weight,
            convolved with the PSF. """
        i, j = self.map_point_to_pixel(x, y, z)
        if self.psf is None:
            if 0 <= i < self.camera_width and 0 <= j < self.camera_height:
                self.recording[t, i, j] += weight
        else:
            # paste the PSF in the recording array centered at the point
            # (it was already rescaled to match the camera resolution)
            # make sure the PSF is centered at the point, and that the PSF's depth dimension
            # is the same as the camera view's depth dimension
            pass
            


            

        

