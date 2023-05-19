import numpy as np
import matplotlib.pyplot as plt
import sncosmo
import re
import pandas as pd


from scipy.integrate import simps, simpson
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import Distance
from astropy.io import ascii

from interpolate_bulla_sed import BullaSEDInterpolator

# Passbands for different surveys
lsst_bands = ['lsstg','lssti','lsstr','lsstu','lssty','lsstz']
jwst_NIRcam_bands = ['f200w']
hst_bands = ['uvf625w']


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
            spectral_flux_density = self.sed_interpolator.interpolator(points)

            # Converting log spectral flux density (from extrapolation) to  spectral flux density
            spectral_flux_density[spectral_flux_density < 0]  = 0 

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
    
    def buildHstLC(self, bands = hst_bands, phases = phases):
        """
        Build apparent (AB mag) vs Phase LC's for the interpolated SED model in HST passband.

        Args:
            bands (list, optional): List of HST passband in which the LC must be computed. Defaults to None.
                                    If value is None, LC will be created in all passbands.
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
    
    def detectionPhasesHst(self, bands = hst_bands, phases = phases):

        threshold = {
            'u': 22,
            'g': 22,
            'r': 22,
            'i': 22,
            'z': 22,
            'y': 22,
        }

        lc = self.buildHstLC(bands, phases)

        phases_below_cutoff = {}

        # Find the phases in each passband where magnitude are below (ie exceed the brightness) the thresholds
        for band in lc:
            
            idx = (lc[band] <= threshold[band])
            phases_below_cutoff[band] = phases[idx]

        return phases_below_cutoff

    def detectionBoolHst(self, bands = hst_bands, phases = phases):

        # Find the phases where the mags exceed the threshold values.
        phases_below_cutoff = self.detectionPhasesHst(bands=bands, phases=phases)

        detection_bool = {}

        # If a passband has 1 or more detection below threshold mag, then mark as true
        for band in phases_below_cutoff:
            
            detection_bool[band] = (len(phases_below_cutoff[band]) >= 1)
        
        return detection_bool
    
    # @TODO: Add functions for HST, JWST, etc



if __name__ == '__main__':

    # Pass band stuff
    bands = ['u','g','r','i','z','y']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Best fit parameters for GW 170817
    mej = 0.05
    phi = 30
    cos_theta = 0.9

    # LC from sed
    temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=43*u.Mpc)
    lcs = temp.buildLsstLC(phases=phases)

    # table from https://iopscience.iop.org/article/10.3847/2041-8213/aa8fc7#apjlaa8fc7t2
    data = pd.read_csv('gw170817photometry.csv', delimiter='\t' )  
    data['mag'] = [float(re.findall("\d+\.\d+", i)[0]) for i in data['Mag [AB]']]

    # Plotting the interpolated lc and scattering the plots
    for i, band in enumerate(bands):
        plt.plot(phases, lcs[f'lsst{band}'], label = f'lsst{band}', c=colors[i])
        plt.scatter(data[data['Filter'] == band]['MJD'], data[data['Filter'] == band]['mag'], label=band, c=colors[i])

    plt.xlabel('Phase')
    plt.ylabel('Apparent Mag')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f'Interpolated Data: mej = {mej} phi = {phi} cos theta = {cos_theta}')
    plt.show()


