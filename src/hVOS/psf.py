import inspect
import numpy
import matplotlib.pyplot as pyplot

import src.hVOS.microscPSF as msPSF


class PSF:
    """ https://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/
    build a 3D point spread function (PSF) for widefield microscopy.
    """
    def __init__(self, radial_lim=(0, 5,0), # in um
                    axial_lim=(-2.0, 2.0), # in um
                 psf_resolution=1.0,
                  NA=1.02, magnification=20, wavelength=0.5, particle_z=0, plot=True):
        self.params = msPSF.m_params
        self.params['NA'] = NA
        self.params['M'] = magnification
        self.params['wavelength'] = wavelength
        self.params['particle_z'] = particle_z
        self.psf_resolution = psf_resolution
        self.radial_lim = radial_lim
        self.axial_lim = axial_lim
        self.plot = plot

    def build_radial_psf(self):
        # Radial PSF
        mp = msPSF.m_params
        pixel_size = 0.05
        rv = numpy.arange(self.radial_lim[0], self.radial_lim[1], self.psf_resolution)
        zv = numpy.arange(self.axial_lim[0], self.axial_lim[1], self.psf_resolution)

        psf_zr = msPSF.gLZRFocalScan(mp, rv, zv, 
                                    pz = self.params['particle_z'], # Particle # um above the surface.
                                    wvl = self.params['wavelength'],      # Detection wavelength.
                                    zd = self.params["zd0"]) # Detector exactly at the tube length of the microscope.
        if self.plot:
            fig, ax = pyplot.subplots()

            ax.imshow(numpy.sqrt(psf_zr),
                    extent=(rv.min(), rv.max(), zv.max(), zv.min()),
                    cmap = 'gray')
            ax.set_xlabel(r'r, $\mu m$')
            ax.set_ylabel(r'z, $\mu m$')
            pyplot.show()
        return psf_zr

    def build_3D_PSF(self):
        """ Build the 3D PSF. """
        psf_radial = self.build_radial_psf()

        # 3D PSF by rotating the 2D PSF around the z-axis
        # (i.e. the optical axis of the microscope)
        theta = numpy.linspace(0, 2*numpy.pi, 100)
        rv = numpy.arange(self.radial_lim[0], self.radial_lim[1], self.psf_resolution)
        zv = numpy.arange(self.axial_lim[0], self.axial_lim[1], self.psf_resolution)
        psf_3d = numpy.zeros((len(rv), len(rv), len(zv)))
        for x in range(len(rv)):
            for y in range(len(rv)):
                # calculate radial distance from the center of the PSF
                r = numpy.sqrt(rv[x]**2 + rv[y]**2)
                # interpolate the 2D PSF to the radial distance
                psf_3d[x, y, :] = numpy.interp(zv, rv, psf_radial[r, :])
        return psf_3d