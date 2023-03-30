import numpy as np
import matplotlib.pyplot as plt
import sncosmo

from scipy.integrate import simps, simpson
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import Distance

from interpolate_bulla_sed import BullaSEDInterpolator

# Passbands for different surveys
lsst_bands = ['lsstg','lssti','lsstr','lsstu','lssty','lsstz']
jwst_NIRcam_bands = ['f200w']


phases = np.arange(start=0.1, stop=7.6, step=0.1)
lmbd = np.arange(start=100, stop = 99900, step=200)

# Code to convert the SED's to lightcurves in different filters
class SEDDerviedLC(): 

    def __init__(self, mej, phi, cos_theta, dist):

        # Setting paramters for interpolating SED
        self.mej = mej
        self.phi = phi
        self.cos_theta = cos_theta
        self.distance = Distance(dist)
        self.sed_interpolator = BullaSEDInterpolator(from_source=False)
    
    def getAbsMagsInPB(self, passband, phases):
        """
        Find the absolute mag (AB) vs phase time series. 

        Args:
            lmbd (numpy array): List of wavelenghts to interpolate SED at.
            t (numpy array): Transmission from the survey filter.
            phases (numpy array): Times to interpolate SED at.

        Returns:
            abs_mag: The absolute magnitudes computed from the SED's for the given
                            survey filter. 
        """
        
        abs_mags = []

        for phase in phases:
                
            # Create mesh of points for interpolator
            mesh_grid = np.meshgrid(self.cos_theta, self.mej, self.phi, phase, lmbd)
            points = np.array(mesh_grid).T.reshape(-1, 5)

            # Interpolate spectral luminosity at all wavelengths
            log_spectral_flux_density = self.sed_interpolator.interpolator(points)

            # Converting log spectral flux density (from extrapolation) to  spectral flux density
            spectral_flux_density  = 10**log_spectral_flux_density 

            # Generate magnitude from synthetic SED
            spectrum = sncosmo.Spectrum(flux = spectral_flux_density, wave=lmbd)
            ab_mag = spectrum.bandmag(band=passband, magsys="ab")
            abs_mags.append(ab_mag)
            

        return abs_mags

    def buildLsstLC(self, bands = lsst_bands, phases = phases):
        """
        Build apparent (AB mag) vs Phase LC's for the interpolated SED model in LSST passband.

        Args:
            bands (list, optional): List of LSST passband in which the LC must be computed. Defaults to None.
                                    If value is None, LC will be created in all 6 passbands.
            phases (_type_, optional): List of phase values where the absolute mag must be interpolated. Defaults 
                                    to None. If value is None, mags for the first 7.5 days will be computed.

        Returns:
            lc: lc contains LC for the phases and passbands mentioned in args. 
        """

        lc = {}
        for band in bands:
                
                mag = self.getAbsMagsInPB(passband=band, phases=phases)

                # Convert from absolute mag to apparent mag based on the distance value
                lc[band] = mag + self.distance.distmod.value

        return lc
    
    def detectionPhasesLsst(self, bands = lsst_bands, phases = phases):

        threshold = {
            'u': 22,
            'g': 22,
            'r': 22,
            'i': 22,
            'z': 22,
            'y': 22,
        }

        lc = self.buildLsstLC(bands, phases)

        phases_below_cutoff = {}

        # Find the phases in each passband where magnitude are below (ie exceed the brightness) the thresholds
        for band in lc:
            
            idx = (lc[band] <= threshold[band])
            phases_below_cutoff[band] = phases[idx]

        return phases_below_cutoff

    def detectionBoolLSST(self, bands = lsst_bands, phases = phases):

        # Find the phases where the mags exceed the threshold values.
        phases_below_cutoff = self.detectionPhasesLsst(bands=bands, phases=phases)

        detection_bool = {}

        # If a passband has 1 or more detection below threshold mag, then mark as true
        for band in phases_below_cutoff:
            
            detection_bool[band] = (len(phases_below_cutoff[band]) >= 1)
        
        return detection_bool

    
    def buildJwstNircamLC(self, bands = jwst_NIRcam_bands, phases = phases):
        """
        Build apparent (AB mag) vs Phase LC's for the interpolated SED model in LSST passband.

        Args:
            bands (list, optional): List of LSST passband in which the LC must be computed. Defaults to None.
                                    If value is None, LC will be created in all 6 passbands.
            phases (_type_, optional): List of phase values where the absolute mag must be interpolated. Defaults 
                                    to None. If value is None, mags for the first 7.5 days will be computed.

        Returns:
            lc: lc contains LC for the phases and passbands mentioned in args. 
        """

        lc = {}
        for band in bands:
                
                mag = self.getAbsMagsInPB(passband=band, phases=phases)

                # Convert from absolute mag to apparent mag based on the distance value
                lc[band] = mag + self.distance.distmod.value

        return lc
    
    def detectionPhasesJwstNircam(self, bands = jwst_NIRcam_bands, phases = phases):

        threshold = {
            'F200W': 22,
        }

        lc = self.buildJwstNircamLC(bands, phases)

        phases_below_cutoff = {}

        # Find the phases in each passband where magnitude are below (ie exceed the brightness) the thresholds
        for band in lc:
            
            idx = (lc[band] <= threshold[band])
            phases_below_cutoff[band] = phases[idx]

        return phases_below_cutoff

    def detectionBoolJwstNircam(self, bands = jwst_NIRcam_bands, phases = phases):

        # Find the phases where the mags exceed the threshold values.
        phases_below_cutoff = self.detectionBoolJwstNircam(bands=bands, phases=phases)

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

    

    temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=10*u.pc)
    lcs = temp.buildLsstLC(phases=phases)

    print(lcs)

    for band in lcs:
        plt.plot(phases, lcs[band], label = band)

    plt.xlabel('Phase')
    plt.ylabel('Absolute Mag')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'Interpolated Data: mej = {mej} phi = {phi} cos theta = {cos_theta}')
    plt.show()