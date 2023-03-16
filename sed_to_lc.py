import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps, simpson
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import Distance

from interpolate_bulla_sed import BullaSEDInterpolator

lsst_bands = ['g','i','r','u','y','z']
phases = np.arange(start=0.1, stop=7.6, step=0.1)

# Code to convert the SED's to lightcurves in different filters
class SEDDerviedLC(): 

    def __init__(self, mej, phi, cos_theta, dist):

        # Setting paramters for interpolating SED
        self.mej = mej
        self.phi = phi
        self.cos_theta = cos_theta
        self.distance = Distance(dist)
        self.sed_interpolator = BullaSEDInterpolator(from_source=False)
    
    def getAbsMagsInPB(self, lmbd, t, phases):
        """
        Find the absolute mag (AB) vs phase time series. 

        Args:
            lmbd (numpy array): List of wavelenghts to interpolate SED at.
            t (numpy array): Transmission from the survey filter.
            phases (numpy array): Times to interpolate SED at.

        Returns:
            abs_mag, phases: The absolute magnitudes computed from the SED's for the given
                            survey filter. Phases is just the input arg.
        """
        
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

            # This is done to integrate over frequency rather than the wavelength
            Fnu = spectral_lum * const.c/ nu**2
            Fnu = Fnu.to(u.erg/(u.s * u.cm**2 * u.Hz))
            
            # Integrating over the passband
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

    def buildLsstLC(self, bands = lsst_bands, phases = phases):
        """
        Build absolute (AB mag) vs Phase LC's for the interpolated SED model in LSST passband.

        Args:
            bands (list, optional): List of LSST passband in which the LC must be computed. Defaults to None.
                                    If value is None, LC will be created in all 6 passbands.
            phases (_type_, optional): List of phase values where the absolute mag must be interpolated. Defaults 
                                    to None. If value is None, mags for the first 7.5 days will be computed.

        Returns:
            lc, phase: lc contains LC for the phases and passbands mentioned in args. phase just returns the input
                        phases arg. 
        """

        lc = {}
        for band in bands:
            with open(f'filters/LSST_LSST.{band}.dat') as fh:
                
                # Transmission at different wavelenghts
                lmbd, t = np.genfromtxt(fh, unpack=True)
                mag, phase = self.getAbsMagsInPB(lmbd, t, phases)
                lc[band] = mag

        return lc, phase
    
    def detectionPhasesLSST(self, bands = lsst_bands, phases = phases):

        threshold = {
            'u': 22,
            'g': 22,
            'r': 22,
            'i': 22,
            'z': 22,
            'y': 22,
        }

        lc, p = self.buildLsstLC(bands, phases)

        # Convert from absolute mag to apparent mag based on the distance value
        for band in lc:
            lc[band] += self.distance.distmod.value

        phases_below_cutoff = {}

        # Find the phases in each passband where magnitude are below (ie exceed the brightness) the thresholds
        for band in lc:
            
            idx = (lc[band] <= threshold[band])
            phases_below_cutoff[band] = phases[idx]

        return phases_below_cutoff

    def detectionBoolLSST(self, bands = lsst_bands, phases = phases):

        # Find the phases where the mags exceed the threshold values.
        phases_below_cutoff = self.detectionPhasesLSST(bands=bands, phases=phases)

        detection_bool = {}

        # If a passband has 1 or more detection below threshold mag, then mark as true
        for band in phases_below_cutoff:
            
            detection_bool[band] = (len(phases_below_cutoff[band]) >= 1)
        
        return detection_bool


    
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