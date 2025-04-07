import numpy as np


class mcPSF:
    ''' Read in and parse monte carlo PSF data.
     
    
    Ch. 6 in Handbook of Biomedical Fluorescence,
      M.A. Mycek, B.W. Pogue, publ. Marcel-Dekker,
        New York, NY, 2003. 

    https://omlc.org/software/mc/mcfluor/index.html

    ../sim/mc_psf/mcOUT102.dat has 21 lines of header
    followed by 200x200 float arrray representing the PSF
    in r and z coordinates.
    '''

    def __init__(self):
        self.psf = self.read_mc_psf()

    def read_mc_psf(self, filename='sim/mc_psf/mcOUT102.dat'):
        ''' Read in the mc psf data. 
        The file is a 200x200 float array representing the PSF
        in r and z coordinates.
        '''
        df = np.loadtxt(filename, skiprows=21, usecols=range(1, 201),
                        dtype=float, delimiter='\t')
        # convert to a 200x200 array
        psf = df.reshape((200, 200))

        # Not needed: scattering only in the x-y plane towards the camera
        # add the negative z values
        #psf = np.vstack((np.flipud(psf), psf))

        # normalize the PSF to sum to 1
        psf = psf / np.sum(psf)

        return psf
    
    def get_mc_psf(self):
        ''' Get the mc psf data. '''
        return self.psf

        
        