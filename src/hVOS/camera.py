import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Camera:
    """ A class that draws a camera view of each cell in the network,
     showing the cell's morphology and the intensity of the hVOS signal,
     Then renders frames at every time step to create a video of the 
     network activity seen through the optical traces. """
    
    def __init__(self, target_cells, morphologies, time, 
                 camera_width=80,  # pixels
                 camera_height=80,  # pixels
                 camera_resolution=6.0,  # um/pixel
                 render_factor=10,  # render this many times more resolution before downscaling
                 camera_angle='coronal', psf=None):
        self.target_cells = target_cells
        self.morphologies = morphologies
        self.time = time
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_resolution = camera_resolution
        self.render_factor = render_factor
        self.camera_angle = camera_angle  # coronal: see all layers in y-z plane

        self.recording = np.zeros((len(self.time), self.camera_width, self.camera_height))
        self.render_canvas = np.zeros((self.camera_width * self.render_factor, 
                                   self.camera_height * self.render_factor))
        self.psf = psf
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def draw_single_frame(self, time_step):
        """ Draw the camera view of the network at a single time step. """
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
                                   x_soma, y_soma, z_soma)
            # get the morphology of the cell

    def _draw_segment(self, segment, intensity_value, x_soma, y_soma, z_soma):
        """ draw the segment of the cell on the camera view. """
        x_seg_prox = float(segment['proximal']['x']) + x_soma
        y_seg_prox = float(segment['proximal']['y']) + y_soma
        z_seg_prox = float(segment['proximal']['z']) + z_soma
        diam_seg_prox = float(segment['proximal']['diameter'])
        x_seg_dist = float(segment['distal']['x']) + x_soma
        y_seg_dist = float(segment['distal']['y']) + y_soma
        z_seg_dist = float(segment['distal']['z']) + z_soma
        diam_seg_dist = float(segment['distal']['diameter'])

        print("Drawing segment from", x_seg_prox, y_seg_prox, z_seg_prox, "to", x_seg_dist, y_seg_dist, z_seg_dist)

        # assume the segment is a tube with a circular cross section
        # stretching from x_seg_prox, y_seg_prox, z_seg_prox 
        #              to x_seg_dist, y_seg_dist, z_seg_dist
        # it linearly interpolates the diameter from proximal to distal
        # the outside of the tube is evenly illuminated with intensity_value
        # the inside and faces of the tube are dark
        # draw the segment on the camera view seen in the y-z plane
        # use mpl_toolkits.mplot3d for 3D rendering
        avg_diam = (diam_seg_prox + diam_seg_dist) / 2
        x = np.linspace(x_seg_prox, x_seg_dist, 100)
        y = np.linspace(y_seg_prox, y_seg_dist, 100)
        z = np.linspace(z_seg_prox, z_seg_dist, 100)
        X, Y, Z = np.meshgrid(x, y, z)
        R = np.sqrt((X - x_soma) ** 2 + (Y - y_soma) ** 2 + (Z - z_soma) ** 2)
        intensity = np.zeros_like(R)
        r_membrane = 1
        intensity[avg_diam / 2 - r_membrane <= R 
                  <= avg_diam / 2 + r_membrane] = intensity_value
        intensity[R < avg_diam / 2 - r_membrane] = 0
        intensity[R > avg_diam / 2 + r_membrane] = 0
        self.ax.plot_surface(X, Y, Z, facecolors=intensity)
        plt.show()
