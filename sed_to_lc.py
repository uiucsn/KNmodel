import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps, simpson
from astropy import units as u
from astropy import constants as const

from interpolate_bulla_sed import BullaSEDInterpolator

# Code to convert the SED's to lightcurves in different filters
class SEDDerviedLC(): 

    phases = np.arange(start=0.1, stop=7.6, step=0.1) # phase in the LC. Proxy for time

    def __init__(self, mej, phi, cos_theta):

        # Setting paramters for interpolating SED
        self.mej = mej
        self.phi = phi
        self.cos_theta = cos_theta
        self.sed_interpolator = BullaSEDInterpolator(from_source=False)
    
    def getAbsMagsInPB(self, lmbd, t, phases):
        
        abs_mags = []

        for phase in phases:
                
            # Create mesh of points for interpolator
            mesh_grid = np.meshgrid(self.cos_theta, self.mej, self.phi, phase, lmbd)
            points = np.array(mesh_grid).T.reshape(-1, 5)

            # Interpolate spectral luminosity at all integer wavelengths in the filter
            log_spectral_lum = self.sed_interpolator.interpolator(points)

            # Converting log spectral luminosity (from extrapolation) to spectral luminosity
            spectral_lum = 10**log_spectral_lum * (u.erg/(u.s * u.cm**2 * u.AA))

            wave_lengths = lmbd * u.AA
            nu = const.c / wave_lengths

            Fnu = spectral_lum * const.c/ nu**2
            Fnu = Fnu.to(u.erg/(u.s * u.cm**2 * u.Hz))

            sed_integral = simpson(x=nu, y=Fnu * t / nu, axis=-1)
            transmission_integral = simpson(x=nu, y=t / nu)
        
            # # Integrating the spectral luminosity times transmision over all wavelengths in the filter
            # v1 = simps(x = lmbd, y= spectral_lum * t * lmbd)
            
            # # Integrating the transmision over all wavelengths in the filter
            # v2 = simps(x=lmbd, y=t * lmbd)

            # Find the flux and mag at current phase
            flux = (sed_integral/transmission_integral) * (u.erg/(u.s * u.cm**2 * u.Hz))
            abs_mag = flux.to(u.ABmag)

            abs_mags.append(abs_mag.value)

        return abs_mags, phases

    def buildLsstLC(self, bands = None, phases = None):
        
        # Set band and phase values to all possible values if None are explicity provided
        if bands == None:
            bands = ['g','i','r','u','y','z']

        if phases == None:
            phases = self.phases

        lc = {}
        for band in bands:
            with open(f'filters/LSST_LSST.{band}.dat') as fh:
                
                # Transmission at different wavelenghts
                lmbd, t = np.genfromtxt(fh, unpack=True)
                mag, phase = self.getAbsMagsInPB(lmbd, t, phases)
                lc[band] = mag

        return lc, phase
    
    # @TODO: Add functions for HST, JWST, etc



if __name__ == '__main__':

    mej = 0.001
    phi = 45
    cos_theta = 0.1

    temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta)
    lcs, phase = temp.buildLsstLC()

    for band in lcs:
        plt.plot(phase, lcs[band], label = band, marker='.')

    plt.xlabel('Phase')
    plt.ylabel('Absolute Mag')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'Interpolated Data: mej = {mej} phi = {phi} cos theta = {cos_theta}')
    plt.show()