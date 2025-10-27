import matplotlib.pyplot as plt
import numpy as np
import gc
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import block_reduce
import os
import time
from scipy import sparse
try:
    from src.hVOS.cell_recording import CellRecording
except ModuleNotFoundError:
    from cell_recording import CellRecording  # for testing in this directory


class Camera:
    """ A class that draws a camera view of each cell in the network,
     showing the cell's morphology and the intensity of the hVOS signal,
     Then renders frames at every time step to create a video of the 
     network activity seen through the optical traces. """
    
    def __init__(self, target_cells, morphologies, time, 
                 fov_center=(0, 0, 0),  # um, including axial focus
                 camera_width=80,  # pixels
                 camera_height=80,  # pixels
                 camera_resolution=6.0,  # um/pixel
                 camera_angle='coronal', 
                 psf=None,
                 psf_resolution=1.0, # um/pixel
                 data_dir='',  # for storing memory-mapped numpy arrays
                 use_2d_psf=False,  # flatten entire image to z=0
                 spike_thresh=0.1674, # optial a.u., found in data['net']['params']['defaultThreshold'], then 0.196 + 0.00286 * -10 = 0.1674
                 init_dummy=False,
                 draw_synapses=None,
                 soma_dend_hVOS_ratio=1.0,
                 compartment_include_prob=1.0,  # probability of including each compartment in the camera view; reduce for speed (e.g. for tuning)
                 precompute_geometry=False,
                 geometry_cache_filename=None
                 ):  
        # seed random
        np.random.seed(4332)

        self.target_cells = target_cells
        self.morphologies = morphologies
        self.time = time
        self.fov_center = fov_center  # in um, including axial focus
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_resolution = camera_resolution
        self.camera_angle = camera_angle  # coronal: see all layers in y-z plane
        self.data_dir = data_dir
        self.use_2d_psf = use_2d_psf
        self.spike_thresh = spike_thresh  # optical units
        # if draw synapses is not None but instead a 
        # SubConnMap, also create a 2D mask
        # showing the location of the synapses (the presynaptic cell id is recorded in the mask)
        self.draw_synapses = draw_synapses
        self.synapse_mask = np.zeros((camera_width, camera_height), dtype=np.uint8)

        # optical tuning
        self.soma_dend_hVOS_ratio = soma_dend_hVOS_ratio
        self.compartment_include_prob = compartment_include_prob

        # make memory-mapped numpy arrays.
        self.cell_recording = None
        if not init_dummy:
            self.cell_recording = CellRecording(self.data_dir,
                                                target_cells[0].get_cell_id(),
                                                self.time,
                                                camera_width=camera_width,
                                                camera_height=camera_height)
        self.psf = psf
        self.psf_resolution = psf_resolution  # treat this as the axial resolution as well, which is realistically ~1 um
        if not init_dummy:
            self.rescale_psf()
            #self.orient_psf_to_camera()

        # store precomputed geometry mappings for each cell
        self.precompute_geometry = precompute_geometry
        self.geometry_map = {}  # cell_id → list of (pixel_idx, weight)
        self.geometry_filename = geometry_cache_filename

        if self.precompute_geometry:
            # turn off psf, store as post_psf, and apply PSF only on the final image
            self.post_psf = np.average(self.psf, axis=2) if self.psf is not None else None
            self.psf = None

        if self.geometry_filename and os.path.exists(self.geometry_filename):
            print("Loading precomputed geometry from", self.geometry_filename)
            self.load_geometry(self.geometry_filename)

    def save_geometry(self, filename=None):
        """Save precomputed geometry map to disk."""
        import pickle, os
        filename = filename or self.geometry_filename or (self.data_dir + "geometry_cache.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.geometry_map, f)
        total_entries = 0
        for key in self.geometry_map:
            total_entries += len(self.geometry_map[key]) if key != 'N/A' else 0
        print(f"Saved geometry map with {total_entries} entries to {filename}")

    def load_geometry(self, filename):
        """Load precomputed geometry map from disk."""
        import pickle
        with open(filename, "rb") as f:
            self.geometry_map = pickle.load(f)
        print(f"Loaded geometry map from {filename} with {len(self.geometry_map)} cells")

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
            # psf has x-y symmetry
            # so, just swap the z and y axes of the PSF
            self.psf = np.swapaxes(self.psf, 0, 2)
            return
        elif self.camera_angle == 'sagittal':
            raise NotImplementedError("Sagittal view is not implemented.")
        else:
            raise NotImplementedError("Only coronal view is implemented.")

    def get_cell_recording(self, decomp_type=None):
        """ Get the recording of the network activity. Returns a CellRecording object"""
        return self.cell_recording 
    
    def flush_memmaps(self):
        """ Flush the memory-mapped numpy arrays to disk. """
        self.cell_recording.flush_memmaps()

    def close_memmaps(self):
        """ Close the memory-mapped numpy arrays. """
        self.cell_recording.close_memmaps()

    def draw_single_frame(self, time_step):
        """ Draw the camera view of the network at a single time step. 
            Loop over all segments of compartments of all cells in the target population.
            use the optical trace and PSF to determine which pixels receive 
            illumination from that segment. 
            """
        for cell in self.target_cells:
            self._draw_cell(cell, time_step=time_step)
            self.flush_memmaps()

    def draw_all_frames(self):
        """ Draw the camera view of the network at all time steps. 
        Do this efficiently, by rendering all frames at once cell-by-cell,
        so that optical trace only needs to be loaded once per segment,
        and segment position only needs to be calculated once per segment. """
        for cell in self.target_cells:
            self._draw_cell(cell, time_step=None)
            self.flush_memmaps()

    def add_time_annotations(self, frame_step_size, img_filenames):
        # add time annotations
        import imageio
        final_images = []
        t_frame = 0


        for filename in img_filenames:
            # Open the image
            img = Image.open(filename).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Define font and text
            try:
                font = ImageFont.truetype("arial.ttf", 20)  # Use a system font
            except IOError:
                font = ImageFont.load_default()  # Fallback to default font if arial.ttf is unavailable

            text = f"{round(t_frame, 1)} ms"
            text_position = (5, 5)  # Position for the text
            text_color = (0, 0, 0)  # Black color

            # Add text to the image
            draw.text(text_position, text, fill=text_color, font=font)

            # Save the annotated image
            annotated_filename = filename + "annotate.png"
            img.save(annotated_filename)

            # Append the annotated image to the final images list
            final_images.append(imageio.imread(annotated_filename))

            # Increment the frame time
            t_frame += frame_step_size

            final_images.append(imageio.imread(filename +"annotate.png"))

        return final_images

    def animate_frames_to_video(self, recording, filename='camera_view.gif',
                                frames=(0,999), time_step_size=0.1, frame_stride=10,
                                vmin=0, vmax=0.01, autoscale_minmax=True, flip=True):
        """ Animate the frames to a video. """
        import imageio
        # each timestep to a frame in a list of images
        sign = -1 if flip else 1
        time_step_size *= frame_stride
        frames = [frames[0], min(frames[1], len(self.time))]
        images = [sign * recording[i, :, :].copy() for i in range(frames[0], frames[1], frame_stride)]

        # if autoscale_minmax is True, set vmin and vmax to the min and max of the images
        if autoscale_minmax:
            vmin = sign * np.min(recording)
            vmax = sign * np.max(recording)
            vmin, vmax = (vmin, vmax) if vmin < vmax else (vmax, vmin)

        # write each image to a file and keep the filenames in a list
        # then use imageio to create a video from the images
        image_filenames = []
        for i, img in enumerate(images):
            plt.clf()
            plt.imshow(img, vmin=vmin, vmax=vmax)
            image_filename = 'frame_' + str(i) + '.png'
            plt.savefig(image_filename)
            
            image_filenames.append(image_filename)

        image_filenames = self.add_time_annotations(time_step_size, image_filenames)
        try:
            imageio.mimsave(self.data_dir + filename, image_filenames)
            print("CREATED MOVIE:", filename)

        except Exception as e:
            print("Not creating movie for " + filename)
            print(e)

    def classify_compartment(self, compartment):
        """ Classify the compartment of the cell. 
        This is used to determine which compartment to draw in the camera view. """
        if 'soma' in compartment:
            return 'soma'
        elif 'axon' in compartment:
            return 'axon'
        elif 'dend' in compartment:
            return 'dend'
        elif 'apic' in compartment:
            return 'apic'
        else:
            raise ValueError("Unknown compartment type: " + compartment)

    def _draw_cell(self, cell, time_step=None):
        """ Draw the camera view of a single cell at a single time step. 
        Returns true if the cell is within the camera view, false otherwise."""
        print("Drawing", cell)

        geom_timer = time.time()

        if self.precompute_geometry and cell.get_cell_id() in self.geometry_map:
            print("Using precomputed mapping")
            geom = self.geometry_map[cell.get_cell_id()]
            for compart in geom:
                for (pixel_i, pixel_j, area_lateral, compartment) in geom[compart]:
                    # directly record the optical intensity without geometric computation
                    intensity_value = np.array(cell.get_optical_trace("V" + compartment))
                    self.record_point_intensity(pixel_i, pixel_j, intensity_value, area_lateral, time_step, None, 
                                            decomp_type=compart,
                                            spike_mask=None)
                    self.cell_recording.record_activity(pixel_i, pixel_j, intensity_value, area_lateral, time_step, compart=compart)
            if self.post_psf is not None:
                print("Applying post-PSF")
                print(self.post_psf.shape, self.post_psf.sum(), self.post_psf)
                self.cell_recording.apply_psf(self.post_psf, time_step=time_step)
            end_timer = time.time()
            print(f"Used precomputed geometry for cell {cell.get_cell_id()} in {end_timer - geom_timer:.2f} seconds")
            return True

        x_soma, y_soma, z_soma = cell.get_soma_position()
        structure = cell.get_morphology().get_structure()
        is_cell_in_bounds = False
        
        # follow the morphology of the cell, pulling optical trace from each compartment
        # and drawing it on the camera view
        for compartment in structure:
            if np.random.rand() > self.compartment_include_prob:
                continue
            intensity_value = np.array(cell.get_optical_trace("V" + compartment))
            spike_mask = intensity_value > self.spike_thresh
            if time_step is not None:
                intensity_value = intensity_value[time_step]
            for segment_id in structure[compartment]:

                optical_tuning_weight = 1.0
                if 'soma' in compartment:
                    optical_tuning_weight = self.soma_dend_hVOS_ratio

                decomp_type = self.classify_compartment(compartment)
                # draw the optical trace on the camera view
                is_cell_in_bounds = \
                    self._draw_segment(structure[compartment][segment_id], 
                                   intensity_value * optical_tuning_weight,
                                   x_soma, y_soma, z_soma, time_step, 
                                   decomp_type=decomp_type, compartment=compartment, spike_mask=spike_mask) \
                    or is_cell_in_bounds
            del intensity_value
            gc.collect()
        del structure
        gc.collect()

        if self.precompute_geometry and cell.get_cell_id() not in self.geometry_map:
            # record geometry that was traversed during drawing
            self.geometry_map[cell.get_cell_id()] = self._capture_last_geometry()

        if self.post_psf is not None:
            print("Applying post-PSF")
            print(self.post_psf.shape, self.post_psf.sum(), self.post_psf)
            self.cell_recording.apply_psf(self.post_psf, time_step=time_step)
        end_timer = time.time()
        print(f"Fully computed geometry for cell {cell.get_cell_id()} in {end_timer - geom_timer:.2f} seconds")
        print(f"Was cell {cell.get_cell_id()} in bounds? {is_cell_in_bounds}")
        return is_cell_in_bounds

    def _draw_segment(self, segment, intensity_value, x_soma, y_soma, z_soma, t, decomp_type=None, spike_mask=None, compartment=None):
        """ draw the segment of the cell on the camera view. 
        Returns true if the segment is within the camera view, false otherwise."""
        x_seg_prox = float(segment['proximal']['x']) + x_soma
        y_seg_prox = float(segment['proximal']['y']) + y_soma
        z_seg_prox = float(segment['proximal']['z']) + z_soma
        diam_seg_prox = float(segment['proximal']['diameter'])
        x_seg_dist = float(segment['distal']['x']) + x_soma
        y_seg_dist = float(segment['distal']['y']) + y_soma
        z_seg_dist = float(segment['distal']['z']) + z_soma
        diam_seg_dist = float(segment['distal']['diameter'])

        if self.draw_synapses is not None:
            if segment in self.draw_synapses.subconn_map:
                for pre_cell_id in self.draw_synapses.subconn_map[segment]:
                    for synapse_loc in self.draw_synapses.subconn_map[segment][pre_cell_id]:
                        # draw the synapse location on the camera view
                        x_synapse = x_seg_prox + (x_seg_dist - x_seg_prox) * synapse_loc
                        y_synapse = y_seg_prox + (y_seg_dist - y_seg_prox) * synapse_loc
                        z_synapse = z_seg_prox + (z_seg_dist - z_seg_prox) * synapse_loc
                        i, j = self.map_point_to_pixel(x_synapse, y_synapse, z_synapse)
                        if 0 <= i < self.camera_width and 0 <= j < self.camera_height:
                            self.synapse_mask[i, j] = pre_cell_id

        # calculate lateral membrane surface area (um^2) of the segment
        # assuming the segment is a conical frustum. 
        # LA = pi * (r1 + r2) * √((r1 - r2)2 + h2)
        height = np.sqrt((x_seg_dist - x_seg_prox) ** 2 +
                         (y_seg_dist - y_seg_prox) ** 2 +
                         (z_seg_dist - z_seg_prox) ** 2)
        area_lateral = np.pi * (diam_seg_prox + diam_seg_dist) * \
                       np.sqrt((diam_seg_prox - diam_seg_dist) ** 2 + height ** 2)

        # if the segment diameters are smaller than the camera resolution (x-y and axial),
        # treat this as a line of light
        resolution = min(self.camera_resolution, self.psf_resolution)
        if diam_seg_prox < resolution and diam_seg_dist < resolution:

            # figure out what fraction of the segment falls into each pixel
            return self._draw_weighted_line(x_seg_prox, y_seg_prox, z_seg_prox,
                                            x_seg_dist, y_seg_dist, z_seg_dist,
                                            intensity_value, area_lateral, t,
                                            decomp_type=decomp_type,
                                            spike_mask=spike_mask,
                                            compartment=compartment)
        elif x_seg_dist == x_seg_prox and y_seg_dist == y_seg_prox and z_seg_dist == z_seg_prox:
            return self._draw_weighted_sphere(x_seg_prox, y_seg_prox, z_seg_prox, 
                                              max(diam_seg_prox / 2, diam_seg_dist / 2),
                                              intensity_value, area_lateral, t,
                                              decomp_type=decomp_type,
                                              spike_mask=spike_mask,
                                              compartment=compartment)
            
        else:   
            return self._draw_weighted_frustum(x_seg_prox, y_seg_prox, z_seg_prox,
                                            x_seg_dist, y_seg_dist, z_seg_dist,
                                            diam_seg_prox, diam_seg_dist,
                                            intensity_value, area_lateral, t,
                                            decomp_type=decomp_type,
                                            spike_mask=spike_mask,
                                            compartment=compartment)
        
    def _draw_weighted_sphere(self, x, y, z, r, intensity_value, area_lateral, t, decomp_type=None, spike_mask=None, compartment=None):
        """ Draw a sphere on the camera view of this weight, weighted by weight.
            Determine what fraction of the sphere falls into each pixel.
            """
        weight = intensity_value * area_lateral
        # calculate the number of steps needed to approximate the sphere
        step_size = 0.5 # in um
        n_steps = int(np.pi * r * 2 / step_size)

        # parametrize by theta and phi and draw a point at each theta, phi on the sphere surface
        theta = np.linspace(0, 2 * np.pi, n_steps)
        phi = np.linspace(0, np.pi, n_steps)
        is_cell_in_bounds = False

        for th in theta:
            for ph in phi:
                # spherical coordinates
                x1 = x + r * np.sin(ph) * np.cos(th)
                y1 = y + r * np.sin(ph) * np.sin(th)
                z1 = z + r * np.cos(ph)
                is_cell_in_bounds = \
                    self._draw_weighted_point(x1, y1, z1, weight / (n_steps ** 2), t,
                                              decomp_type=decomp_type,
                                              spike_mask=spike_mask,
                                              compartment=compartment) \
                    or is_cell_in_bounds
        return is_cell_in_bounds

    def get_points_of_circle(self, center, radius, num_points, u, v):
        """ Get the points of a circle in 3D space.
            Args:
                center: the center of the circle
                u, v: the vectors defining the plane of the circle
                radius: the radius of the circle
                num_points: the number of points to sample on the circle
            Returns:
                A list of points on the circle.
        """
        # Generate points
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + radius * (np.cos(theta) * u[0] + np.sin(theta) * v[0])
        y = center[1] + radius * (np.cos(theta) * u[1] + np.sin(theta) * v[1])
        z = center[2] + radius * (np.cos(theta) * u[2] + np.sin(theta) * v[2])
        
        points = np.stack([x, y, z], axis=-1)
        return points
        
    def _draw_weighted_frustum(self, x1, y1, z1, x2, y2, z2, d1, d2, 
                               intensity_value, area_lateral, t, decomp_type=None, spike_mask=None, compartment=None):
        """ Draw a frustum on the camera view of this weight, weighted by weight.
            Determine what fraction of the frustum falls into each pixel.
            
            Just break this up into multiple lines connecting two circles of diameter d1 and d2
            Step around the circumference of the circles and draw lines between the points.
            The step size is 1/3 the camera resolution, so not many lines needed.
            """
        weight = intensity_value * area_lateral
        # calculate the number of steps needed to approximate the frustum
        step_size = 0.5  # in um
        vector_between_circles = np.array([x2 - x1, y2 - y1, z2 - z1])
        n_points = int(max(d1, d2) / step_size)

        # Normalize the normal vector
        norm = np.linalg.norm(vector_between_circles) 
        if norm == 0:  # should not happen because this would have been a sphere
            return False
        normal = vector_between_circles / norm

        # Create two orthogonal vectors to the normal
        if normal[0] != 0 or normal[1] != 0:
            temp_vector = np.array([-normal[1], normal[0], 0])
        else:
            temp_vector = np.array([1, 0, 0])

        u = temp_vector / np.linalg.norm(temp_vector)
        v = np.cross(normal, u)

        # the plane that contains the two circles is perpendicular to the vector between them
        circle1_points = self.get_points_of_circle([x1, y1, z1], 
                                                   d1 / 2, 
                                                   n_points,
                                                   u, v)
        circle2_points = self.get_points_of_circle([x2, y2, z2],
                                                   d2 / 2, 
                                                   n_points,
                                                   u, v)
        is_cell_in_bounds = False
        for i_circle in range(len(circle1_points)):
            is_cell_in_bounds = self._draw_weighted_line(circle1_points[i_circle][0], 
                                                         circle1_points[i_circle][1], 
                                                         circle1_points[i_circle][2],
                                                         circle2_points[i_circle][0],
                                                         circle2_points[i_circle][1],
                                                         circle2_points[i_circle][2],
                                                         intensity_value, area_lateral / n_points, t,
                                                         decomp_type=decomp_type,
                                                         spike_mask=spike_mask,
                                                         compartment=compartment) \
                or is_cell_in_bounds
            
        del circle1_points, circle2_points
        gc.collect()
        return is_cell_in_bounds

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
    
    def bresenham_line(self, x1, y1, x2, y2):
        """
        Draws a line between two points using Bresenham's algorithm.
        Args:
            x1: x-coordinate of the starting point.
            y1: y-coordinate of the starting point.
            x2: x-coordinate of the ending point.
            y2: y-coordinate of the ending point.
        Returns:
            A list of tuples representing the coordinates of the points on the line.
            Includes both the start and end points.
        """
        m_new = 2 * (y2 - y1) 
        slope_error_new = m_new - (x2 - x1) 
        points = []
    
        y = y1 
        for x in range(x1, x2+1): 
    
            points.append((x, y))
  
            # Add slope to increment angle formed 
            slope_error_new = slope_error_new + m_new 
    
            # Slope error reached limit, time to 
            # increment y and update slope error. 
            if (slope_error_new >= 0): 
                y = y+1
                slope_error_new = slope_error_new - 2 * (x2 - x1) 
        return points
  
    def _draw_weighted_line(self, x1, y1, z1, x2, y2, z2, 
                            intensity_value, area_lateral, t, decomp_type=None, spike_mask=None, compartment=None):
        """ Draw a line on the camera view of this weight, weighted by weight.
            Determine what fraction of the line falls into each pixel."""
        weight = intensity_value * area_lateral
        # Bresenham's line algorithm
        is_cell_in_bounds = False
        i_start, j_start = self.map_point_to_pixel(x1, y1, z1)
        i_end, j_end = self.map_point_to_pixel(x2, y2, z2)

        # if the line does not cross any pixels...
        if i_start == i_end and j_start == j_end:
            step_size = 0.5  # in um
            z_dist = np.abs(z2 - z1)
            zv = np.linspace(z1, z2, int(z_dist / step_size)) # ...it might still cross in/out of z-focus
            if len(zv) <= 2:  # ... if it a very short line, treat as one point
                del zv
                gc.collect()
                return self._draw_weighted_point(x1, y1, z1, intensity_value, area_lateral, t, 
                                                 i=i_start, j=j_start, 
                                                 decomp_type=decomp_type,
                                                 spike_mask=spike_mask, 
                                                 compartment=compartment)
            for z_inter in zv:  # ... if it is a longer line, chunk into many points
                is_cell_in_bounds = self._draw_weighted_point(x1, y1, z_inter, 
                                                              intensity_value, area_lateral / len(zv), t,
                                                              decomp_type=decomp_type,
                                                              spike_mask=spike_mask, 
                                                              compartment=compartment) \
                    or is_cell_in_bounds
            return is_cell_in_bounds

        # list the pixels that the line passes through (from i_start, j_start to i_end, j_end)
        # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        pixels = self.bresenham_line(i_start, j_start, i_end, j_end)

        # make an approximation and say that the start and end contain half the weight
        # and the rest of the weight is distributed evenly among the pixels in between
        if len(pixels) <= 2:
            del pixels
            gc.collect()
            r1 = self._draw_weighted_point(x1, y1, z1, intensity_value, area_lateral / 2, t, 
                                           i=i_start, j=j_start, 
                                           decomp_type=decomp_type,
                                           spike_mask=spike_mask, 
                                           compartment=compartment)
            r2 = self._draw_weighted_point(x2, y2, z2, intensity_value, area_lateral / 2, t, 
                                           i=i_end, j=j_end, 
                                           decomp_type=decomp_type,
                                           spike_mask=spike_mask, 
                                           compartment=compartment)
            return r1 or r2
        else:
            area_per_pixel = area_lateral / (len(pixels) - 1)
            for i_px, pt in enumerate(pixels):
                if i_px == 0 or i_px == len(pixels) - 1:
                    continue
                is_cell_in_bounds = self._draw_weighted_point(x1, y1, z1, intensity_value, area_per_pixel, 
                                                              t, i=pt[0], j=pt[1],
                                                              decomp_type=decomp_type,
                                                              spike_mask=spike_mask,
                                                                compartment=compartment) \
                    or is_cell_in_bounds
                
            del pixels
            gc.collect()
                
            r1 = self._draw_weighted_point(x1, y1, z1, intensity_value, area_per_pixel / 2, t, 
                                           i=i_start, j=j_start, 
                                           decomp_type=decomp_type,
                                           spike_mask=spike_mask,
                                           compartment=compartment)
            r2 = self._draw_weighted_point(x2, y2, z2, intensity_value, area_per_pixel / 2, t, 
                                           i=i_end, j=j_end, 
                                           decomp_type=decomp_type,
                                           spike_mask=spike_mask,
                                           compartment=compartment)
            return r1 or r2 or is_cell_in_bounds

    def _draw_weighted_point(self, x, y, z, intensity_value, area_lateral, t, i=None, j=None, 
                             decomp_type=None, spike_mask=None, compartment=None):
        """ Draw a point on the camera view of this weight,
            convolved with the PSF. 
            Returns true if the point is within the camera view, false otherwise."""
        weight = intensity_value * area_lateral
        
        if i is None or j is None:
            i, j = self.map_point_to_pixel(x, y, z)
        # print("Drawing point at", i, j, "with weight", weight)
        if self.psf is None:
            if 0 <= i < self.camera_width and 0 <= j < self.camera_height:
                if self.precompute_geometry:
                    if not hasattr(self, "_geometry_buffer"):
                        self._geometry_buffer = {}
                    if decomp_type not in self._geometry_buffer:
                        self._geometry_buffer[decomp_type] = []
                    self._geometry_buffer[decomp_type].append((i, j, area_lateral, compartment))
                self.record_point_intensity(i, j, intensity_value, area_lateral, t, None, 
                                            decomp_type=decomp_type,
                                            spike_mask=spike_mask)
            return (0 <= i < self.camera_width and 0 <= j < self.camera_height)
        else:
            # paste the PSF in the recording array centered at the point
            # (it was already rescaled to match the camera resolution)
            # make sure the PSF is centered at the point, and that the PSF's depth dimension
            # is the same as the camera view's depth dimension

            # IF PSF size is bigger than camera view,
            #   then the PSF is too big to fit in the camera view
            if self.psf.shape[0] > self.camera_width or \
                self.psf.shape[1] > self.camera_height:
                raise ValueError("PSF is too big to fit in the camera view. "
                "Just make the PSF smaller or the camera view bigger.")
            
            x_psf_shape, y_psf_shape, z_psf_shape = self.psf.shape
            x_psf_lim = [-x_psf_shape // 2, -x_psf_shape // 2 + x_psf_shape]
            y_psf_lim = [-y_psf_shape // 2, -y_psf_shape // 2 + y_psf_shape]
            z_psf_lim = [-z_psf_shape // 2, -z_psf_shape // 2 + z_psf_shape]

            if i + x_psf_lim[1] < 0:  # the PSF is entirely to the left of the camera view
                return False 
            if i + x_psf_lim[0] > self.camera_width: # the PSF is entirely to the right of the camera view
                return False
            if j + y_psf_lim[1] < 0: # the PSF is entirely above the camera view
                return False
            if j + y_psf_lim[0] > self.camera_height: # the PSF is entirely below the camera view
                return False

            z_fov = self.fov_center[2] / self.camera_resolution

            # if the point z is outside the PSF's z range, return False
            xy_psf_weighted = None
            if not self.use_2d_psf:
                if z < z_fov + z_psf_lim[0] or z > z_fov + z_psf_lim[1]:
                    return False
            z_overlap = 0
            if not self.use_2d_psf:
                z_overlap = int(round(z - z_fov))

            if z_overlap > self.psf[:, :, :].shape[2] // 2 - 1:
                return False
            if -z_overlap > self.psf[:, :, :].shape[2] // 2 - 1:
                return False
            
            if t is None:
                z_center_psf = self.psf.shape[2] // 2
                z_overlap = z_center_psf + z_overlap
                psf_slice = self.psf[:, :, z_overlap].copy()
                
                # If psf_slice shape is 18x18, and intensity_value is 2000x1x1,
                #   tile psf_slice to match the intensity_value shape (2000 x 18 x 18)
                psf_slice = np.tile(psf_slice, (intensity_value.shape[0], 1, 1))
                # element-wise multiplication of the PSF slice with the weight
                intensity_value = intensity_value.reshape(-1, 1, 1)
                xy_psf_weighted = psf_slice * np.ones(intensity_value.shape)
            else:
                xy_psf_weighted = self.psf[:, :, z_overlap] # * intensity_value

            # actual bounds
            i_bounds = [max(0, i + x_psf_lim[0]), min(self.camera_width, i + x_psf_lim[1])]
            j_bounds = [max(0, j + y_psf_lim[0]), min(self.camera_height, j + y_psf_lim[1])]

            # fit xy_psf to match the actual bounds
            if i + x_psf_lim[0] < 0:
                x_psf_lim = [-i, x_psf_lim[1]]
            elif i + x_psf_lim[1] > self.camera_width:
                x_psf_lim = [x_psf_lim[0], self.camera_width - i]
            if j + y_psf_lim[0] < 0:
                y_psf_lim = [-j, y_psf_lim[1]]
            if j + y_psf_lim[1] > self.camera_height:
                y_psf_lim = [y_psf_lim[0], self.camera_height - j]

            x_psf_weighted_bounded = None
            if t is None:
                x_psf_weighted_bounded = \
                    xy_psf_weighted[:, x_psf_lim[0]-x_psf_lim[0]:x_psf_lim[1]-x_psf_lim[0],
                                    y_psf_lim[0]-y_psf_lim[0]:y_psf_lim[1]-y_psf_lim[0]]
            else:
                x_psf_weighted_bounded = \
                    xy_psf_weighted[x_psf_lim[0]-x_psf_lim[0]:x_psf_lim[1]-x_psf_lim[0],
                                    y_psf_lim[0]-y_psf_lim[0]:y_psf_lim[1]-y_psf_lim[0]]

            self.record_point_intensity(None, None, x_psf_weighted_bounded, area_lateral,
                                        t, i_bounds=i_bounds, j_bounds=j_bounds,
                                        decomp_type=decomp_type,
                                        spike_mask=spike_mask)
            del xy_psf_weighted, x_psf_weighted_bounded, psf_slice
            gc.collect()
            return True            
        
    def record_point_intensity(self, i, j, intensity_value, area_lateral, t, i_bounds=None, j_bounds=None,
                               decomp_type=None, spike_mask=None):
        """ Record the intensity of a point on the camera view. """
        self.cell_recording.record_activity(i, j, intensity_value, area_lateral, t, i_bounds=i_bounds, j_bounds=j_bounds,
                                            compart=decomp_type, spike_mask=spike_mask)

    def _capture_last_geometry(self):
        geom = getattr(self, "_geometry_buffer", [])
        self._geometry_buffer = []
        return geom

'''
    def get_sparse_W(self, cell_id):
        geom = np.array(self.geometry_map[cell_id])
        rows, cols, data = geom[:,0], geom[:,1], geom[:,2]
        P = self.camera_width * self.camera_height
        C = len(self.target_cells[cell_id].get_morphology().get_structure())
        # flatten pixel indices
        flat_rows = rows * self.camera_height + cols
        W = sparse.csr_matrix((data, (flat_rows, np.arange(len(data)))), shape=(P, C))
        return W
    
    def fast_draw_cell(self, W, Vm):
        """ Fast draw a cell using precomputed sparse matrix W and Vm. """
        # Vm is (C, T)
        # W is (P, C)
        # result is (P, T)
        P = W @ Vm
        '''
