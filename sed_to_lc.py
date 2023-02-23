import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
from astropy import units as u

from interpolate_bulla_sed import BullaSEDInterpolator

# Code to convert the SED's to lightcurves in different filters
class SEDDerviedLC(): 

    phases = np.arange(start=0.1, stop=19.9, step=0.1) # phase in the LC. Proxy for time

    def __init__(self, mej, phi, cos_theta):

        # Setting paramters for interpolating SED
        self.mej = mej
        self.phi = phi
        self.cos_theta = cos_theta
        self.sed_interpolator = BullaSEDInterpolator()
    
    def getAbsMagsInPB(self, lmbd, t):
        
        abs_mags = []

        for phase in self.phases:
                
            # Create mesh of points for interpolator
            mesh_grid = np.meshgrid(self.cos_theta, self.mej, self.phi, phase, lmbd)
            points = np.array(mesh_grid).T.reshape(-1, 5)

            # Interpolate spectral luminosity at all integer wavelengths in the filter
            spectral_lum = self.sed_interpolator.interpolator(points)

            # @TODO: Make sure the math checks out
        
            # Integrating the spectral luminosity times transmision over all wavelengths in the filter
            v1 = simps(x = lmbd, y= spectral_lum * t * lmbd)
            
            # Integrating the transmision over all wavelengths in the filter
            v2 = simps(x=lmbd, y=t * lmbd)

            # Find the flux and mag at current phase
            flux = (v1/v2) * (u.erg/(u.s * u.cm**2))
            lum = flux * 4 * np.pi * (10 * u.pc)**2
            abs_mag = lum.to(u.M_bol)

            abs_mags.append(abs_mag.value)

        return abs_mags, self.phases

    def buildLsstLC(self, bands = ['g','i','r','u','y','z']):

        lc = {}
        for band in bands:
            with open(f'filters/LSST_LSST.{band}.dat') as fh:
                
                # Transmission at different wavelenghts
                lmbd, t = np.genfromtxt(fh, unpack=True)
                mag, phase = self.getAbsMagsInPB(lmbd, t)
                lc[band] = mag

        return lc, self.phases
    
    # @TODO: Add functions for HST, JWST, etc



if __name__ == '__main__':

    mej = 0.03
    phi = 45
    cos_theta = 0.1

    temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta)
    lcs, phase = temp.buildLsstLC()

    for band in lcs:
        plt.plot(phase, lcs[band], label = band)

    plt.xlabel('Phase')
    plt.ylabel('Absolute Mag')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'Interpolated Data: mej = {mej} phi = {phi} cos theta = {cos_theta}')
    plt.show()