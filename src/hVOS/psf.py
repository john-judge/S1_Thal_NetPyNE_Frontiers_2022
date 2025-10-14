from scipy.signal import convolve2d
import numpy
import matplotlib.pyplot as pyplot

try:
    import src.hVOS.microscPSF as msPSF
except ModuleNotFoundError:
    import microscPSF as msPSF  # for testing in this directory


class PSF:
    """ https://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/
    build a 3D point spread function (PSF) for widefield microscopy.

    tissue is much more turbid than the medium in which the PSF was measured,
    the PSF will be distorted by scattering and absorption in the tissue.
    
    """
    def __init__(self, radial_lim=(0, 18.0), # in um
                    axial_lim=(-12.0, 12.0), # in um
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

        self.radial_psf = self.build_radial_psf()

    def build_radial_psf(self):
        # Radial PSF, axial (z) x radial (r)
        mp = msPSF.m_params
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
    
    def get_radial_psf(self):
        """ Get the radial PSF. """
        return self.radial_psf
    
    def convolve_radial_psf(self, add_psf):
        """ Convolve the radial PSF with another PSF of same resolution """
        psf_radial = self.get_radial_psf()
        psf_radial = convolve2d(psf_radial, add_psf, mode='same')  # preserve the size


        # normalize the PSF to sum to 1
        psf_radial = psf_radial / numpy.sum(psf_radial)
        self.radial_psf = psf_radial
        return psf_radial

    def build_3D_PSF(self):
        """ Build the 3D PSF. """
        psf_radial = self.get_radial_psf()

        # 3D PSF by rotating the 2D PSF around the z-axis
        # (i.e. the optical axis of the microscope)
        rv = numpy.arange(self.radial_lim[0], self.radial_lim[1], self.psf_resolution)
        zv = numpy.arange(self.axial_lim[0], self.axial_lim[1], self.psf_resolution)
        psf_3d = numpy.zeros((len(rv), len(rv), len(zv)))
        xy_center = len(rv) // 2
        for x in range(len(rv)):
            for y in range(len(rv)):
                # calculate radial distance from the center of the PSF
                r = numpy.sqrt((rv[x] - xy_center)**2 + (rv[y] - xy_center)**2)
                # interpolate the 2D PSF to the radial distance r
                for z in range(len(zv)):
                    r_interp_psf = numpy.interp(r, rv, psf_radial[z, :])
                    psf_3d[x, y, z] = r_interp_psf
        return psf_3d