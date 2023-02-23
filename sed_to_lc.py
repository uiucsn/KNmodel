import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
from astropy import units as u

from interpolate_bulla_sed import BullaSEDInterpolator

# Code to convert the SED's to lightcurves in different filters
class SEDDerviedLC(): 

    phases = np.arange(start=0.1, stop=19.9, step=0.1) # phase in the LC. Proxy for time

    def __init__(self, mej, phi, cos_theta):

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


temp = SEDDerviedLC(mej = 0.08, phi = 15, cos_theta = 0.5)
lcs, phase = temp.buildLsstLC()

for band in lcs:
    plt.plot(phase, lcs[band], label = band)

plt.xlabel('Phase')
plt.ylabel('Absolute Mag')
plt.gca().invert_yaxis()
plt.legend()
plt.show()