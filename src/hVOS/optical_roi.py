import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError


class BaselineCorrection:
    """ Class for trace baseline correction. Module version of code in baseline_proof.ipynb """
    def __init__(self, exclusion_windows=((100,300), (357,552)), 
                 polyorder=8):
        self.exclusion_windows = exclusion_windows
        self.polyorder = polyorder
        self.corrected_trace = None
        self.baseline = None

    # fit a polynomial to the trace and subtract it
    def fit_polynomial(self, x, y, t_complete):
        coefs = polynomial.polyfit(x, y, self.polyorder)
        y_fit = polynomial.polyval(t_complete, coefs)
        return y_fit

    def fit_baseline(self, t, trace):
        ''' Fit a polynomial to the trace and subtract it, excluding the points at
         the exclusion windows, respecting the original times of the included points '''
        
        # Create a mask for the exclusion windows
        mask = np.ones(len(t), dtype=bool)
        for start, end in self.exclusion_windows:
            mask[start:end] = False

        # Get the x and y values for the polynomial fit
        x = t[mask]
        y = trace[mask]

        # Fit the polynomial to the trace
        try:
            y_fit = self.fit_polynomial(x, y, t)
        except LinAlgError:
            print("LinAlgError: Unable to fit polynomial. Returning original trace.")
            return trace
        
        # Subtract the fitted polynomial from the trace
        subtracted = trace - y_fit
        return subtracted


class ROI:
    """Represents a single Region of Interest (ROI)."""
    def __init__(self, center, diameter, mask=None):
        """
        Initialize an ROI.

        Args:
            center (tuple): (x, y) coordinates of the ROI center.
            diameter (int): Diameter of the ROI.
            mask (np.ndarray): Optional mask for the ROI.
        """
        self.center = center
        self.diameter = diameter
        self.mask = mask

    def generate_mask(self, shape):
        """
        Generate a circular mask for the ROI.

        Args:
            shape (tuple): Shape of the image (height, width).

        Returns:
            np.ndarray: A boolean mask for the ROI.
        """
        mask = np.zeros(shape, dtype=bool)
        x_center, y_center = self.center
        radius = self.diameter // 2

        for i in range(max(0, x_center - radius), min(shape[0], x_center + radius)):
            for j in range(max(0, y_center - radius), min(shape[1], y_center + radius)):
                if np.sqrt((i - x_center) ** 2 + (j - y_center) ** 2) <= radius:
                    mask[i, j] = True

        self.mask = mask
        return mask

    def apply_mask(self, image):
        """
        Apply the ROI mask to an image.

        Args:
            image (np.ndarray): The image to apply the mask to.

        Returns:
            np.ndarray: The masked image.
        """
        if self.mask is None:
            raise ValueError("Mask has not been generated. Call generate_mask() first.")
        return image * self.mask


class ROIGenerator:
    """Generates and manages multiple ROIs."""
    def __init__(self, n_rois, diameter_range, image_shape):
        """
        Initialize the ROIGenerator.

        Args:
            n_rois (int): Number of ROIs to generate.
            diameter_range (tuple): Min and max diameter for ROIs.
            image_shape (tuple): Shape of the image (height, width).
        """
        self.n_rois = n_rois
        self.diameter_range = diameter_range
        self.image_shape = image_shape
        self.rois = []

    def generate_rois(self):
        """
        Generate random ROIs.

        Returns:
            list: A list of ROI objects.
        """
        for _ in range(self.n_rois):
            diameter = random.randint(self.diameter_range[0], self.diameter_range[1])
            x = random.randint(diameter // 2, self.image_shape[0] - diameter // 2)
            y = random.randint(diameter // 2, self.image_shape[1] - diameter // 2)
            roi = ROI(center=(x, y), diameter=diameter)
            roi.generate_mask(self.image_shape)
            self.rois.append(roi)
        return self.rois

    def plot_rois(self, image, colors=None):
        """
        Plot the ROIs on an image.

        Args:
            image (np.ndarray): The image to plot the ROIs on.
            colors (list): List of colors for the ROIs.
        """
        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']

        plt.figure(figsize=(10, 6))
        plt.imshow(image, cmap='gray', interpolation='nearest')

        for i, roi in enumerate(self.rois):
            mask = roi.mask
            y, x = np.where(mask)
            plt.scatter(x, y, color=colors[i % len(colors)], s=5, alpha=0.25)

        plt.title("ROIs")
        plt.show()

    def extract_traces(self, image_stack):
        """
        Extract optical traces for each ROI from an image stack.

        Args:
            image_stack (np.ndarray): A 3D array (time, height, width).

        Returns:
            list: A list of optical traces for each ROI.
        """
        traces = []
        for roi in self.rois:
            trace = np.sum(image_stack[:, roi.mask], axis=1)
            traces.append(trace)
        return traces