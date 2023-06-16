import numpy as np
import matplotlib.pyplot as plt
import sncosmo
import re
import pandas as pd
import sfdmap


from scipy.integrate import simps, simpson
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import Distance, SkyCoord
from astropy.io import ascii
from matplotlib import cm

from interpolate_bulla_sed import BullaSEDInterpolator

# Passbands for different surveys
lsst_bands = ['lsstg','lssti','lsstr','lsstu','lssty','lsstz']
jwst_NIRcam_bands = ['f200w']
hst_bands = ['uvf625w']


phases = np.arange(start=0.3, stop=7.6, step=0.2)
lmbd = np.arange(start=100, stop = 99901, step=200)

mej_grid_low = 0.01
mej_grid_high = 0.1

# Code to convert the SED's to lightcurves in different filters
class SEDDerviedLC(): 

    def __init__(self, mej, phi, cos_theta, dist, coord, av , rv = 3.1):

        # Setting paramters for interpolating SED
        self.mej = mej
        self.phi = phi
        self.cos_theta = cos_theta
        self.distance = Distance(dist)
        self.coord = coord
        self.sed_interpolator = BullaSEDInterpolator(from_source=False)

        # For mw extinction
        self.dust_map = sfdmap.SFDMap('./sfddata-master')
        self.mw_ebv = self.dust_map.ebv(self.coord.ra, self.coord.dec)

        # For host extinction
        self.av = av
        self.rv = rv
        self.host_ebv = self.av/self.rv
    

    def getInterpolatedSed(self, phases=phases, remove_negative = True, extrapolate = True):

        interpolated_sed = np.zeros((len(phases), len(lmbd)))

        for i, phase in enumerate(phases):
                
            # Create mesh of points for interpolator
            if extrapolate:
                # Extrapolate off the grid
                mesh_grid = np.meshgrid(self.cos_theta, self.mej, self.phi, phase, lmbd)

            else:
                # Find the closest sed on the grid.
                if self.mej < mej_grid_low:
                    # if mej is too low, get the spectra for the lowest mej on the grid.
                    mesh_grid = np.meshgrid(self.cos_theta, mej_grid_low, self.phi, phase, lmbd)

                elif self.mej > mej_grid_high:
                    # if mej is too high, get the spectra for the highest mej on the grid.
                    mesh_grid = np.meshgrid(self.cos_theta, mej_grid_high, self.phi, phase, lmbd)

                else:
                    # if mej is on the grid, just interpolate.                 
                    mesh_grid = np.meshgrid(self.cos_theta, self.mej, self.phi, phase, lmbd)

            points = np.array(mesh_grid).T.reshape(-1, 5)

            # Interpolate spectral luminosity at all wavelengths
            spectral_flux_density = self.sed_interpolator.interpolator(points)

            if remove_negative:
                spectral_flux_density[spectral_flux_density < 0]  = 0 

            interpolated_sed[i, :] = spectral_flux_density

        return interpolated_sed

  
    def getAvgScalingFactor(self):
        
        df = pd.read_csv('data/scaling_laws.csv')

        # Find entries with the same cos theta and phi, for mej = mej_grid_high at all phase values
        closest_df = df[(df['cos_theta'] == self.cos_theta) & (df['phi'] == self.phi)]

        # Curve parameters
        a_avgs = closest_df['a_avg']
        n_avgs = closest_df['n_avg']

        if self.mej > mej_grid_high:

            # Find the avg flux relative to mej = mej_grid_high at all phase values for current mej
            scaling_values = (a_avgs * (self.mej)**n_avgs) / ((a_avgs * (mej_grid_high)**n_avgs))

        elif self.mej < mej_grid_low:

            # Find the avg flux relative to mej = mej_grid_high at all phase values for current mej
            scaling_values = (a_avgs * (self.mej**n_avgs)) / (a_avgs * (mej_grid_low**n_avgs))

        return scaling_values.to_numpy()[0]
    
    def getMaxScalingFactor(self):

        df = pd.read_csv('data/scaling_laws.csv')

        # Find entries with the same cos theta and phi at all phase values
        closest_df = df[(df['cos_theta'] == self.cos_theta) & (df['phi'] == self.phi)]

        # Curve parameters
        a_maxs = closest_df['a_max']
        n_maxs = closest_df['n_max']

        if self.mej > mej_grid_high:

            # Find the avg flux relative to mej = mej_grid_high at all phase values for current mej
            scaling_values = (a_maxs * (self.mej)**n_maxs) / ((a_maxs * (mej_grid_high)**n_maxs))

        elif self.mej < mej_grid_low:

            # Find the avg flux relative to mej = mej_grid_high at all phase values for current mej
            scaling_values = (a_maxs * (self.mej)**n_maxs) / ((a_maxs * (mej_grid_low)**n_maxs))

        return scaling_values.to_numpy()[0]

    def getSed(self, phases=phases, scaling='avg'):

        scaling_options = ['avg', 'max']

        if scaling not in scaling_options:
            raise ValueError(f"Invalid scaling method. Expected one of: {scaling_options}, got {scaling}" )
        
        # If the sed is off the grid, pick the closes one and scale.
        if self.mej > mej_grid_high or self.mej < mej_grid_low:

            # Get the closest sed on the grid.
            closest_grid_sed = self.getInterpolatedSed(phases=phases, remove_negative=True, extrapolate=False)

            # Scale the SED
            if scaling == 'avg':
                self.scaling_factor = self.getAvgScalingFactor()
            elif scaling == 'max':
                self.scaling_factor = self.getMaxScalingFactor()

            sed = self.scaling_factor * closest_grid_sed

        else:
            
            self.scaling_factor = 1
            # Get interpolated sed.
            sed = self.getInterpolatedSed(phases=phases, remove_negative=True)

        return sed

    def makeSedPlot(self, phases=phases):

        interpolated_sed = self.getSed(phases=phases)
        source_name = f"Interpolated Object\ncos_theta: {self.cos_theta}, mej: {self.mej}, phi: {self.phi}"

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X, Y = np.meshgrid(lmbd, phases)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, interpolated_sed, cmap=cm.plasma)
        ax.set_xlabel('Wavelength (A)')
        ax.set_ylabel('Phase (days)')
        ax.set_zlabel('Spectral flux density (erg / s / cm^2 / A)')
        ax.set_title(source_name)
        return ax

    
    def getAbsMagsInPB(self, passband, phases, apply_extinction = True, scaling='avg'):
        
        source_name = f"Interpolated Object\ncos_theta: {self.cos_theta}, mej: {self.mej}, phi: {self.phi}"

        interpolated_sed = self.getSed(phases=phases, scaling=scaling)

        source = sncosmo.TimeSeriesSource(phase=phases, wave=lmbd, flux = interpolated_sed, name=source_name, zero_before=True)

        self.model = sncosmo.Model(source)

        if apply_extinction:

            # add host galaxy extinction E(B-V)
            self.model.add_effect(sncosmo.CCM89Dust(), 'host', 'rest')
            self.model.set(hostebv = self.host_ebv)

            # add MW extinction to observing frame
            self.model.add_effect(sncosmo.F99Dust(), 'mw', 'obs')
            self.model.set(mwebv=self.mw_ebv)


        abs_mags = self.model.bandmag(band=passband, time = phases, magsys="ab")
    
        return abs_mags

    def buildLsstLC(self, bands = lsst_bands, phases = phases, scaling='avg'):
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
                mag = self.getAbsMagsInPB(passband=band, phases=phases, scaling=scaling)

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
    bands = ['g','r','i']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
    mej = 0.05
    phi = 30
    cos_theta = 0.9

    # coordinates for GW170817
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 43*u.Mpc

    def plot_GW170817_lc_and_spectra():

        av = 0.0

        # LC from sed
        temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
        lcs = temp.buildLsstLC(phases=phases)
        print(temp.model)
        temp.makeSedPlot()
        plt.show()


        # table from https://iopscience.iop.org/article/10.3847/2041-8213/aa8fc7#apjlaa8fc7t2
        data = pd.read_csv('gw170817photometry.csv', delimiter='\t' )  
        #data = data[data['Instrument'] == 'DECam']
        data['mag'] = [float(re.findall("\d+\.\d+", i)[0]) for i in data['Mag [AB]']]

        # Plotting the interpolated lc and scattering the plots
        # for i, band in enumerate(bands):
        #     plt.plot(phases, lcs[f'lsst{band}'], label = f'lsst{band}', c=colors[i])
        #     plt.scatter(data[data['Filter'] == band]['MJD'], data[data['Filter'] == band]['mag'], label=band, c=colors[i])

        plt.scatter(data[data['Filter'] == 'g']['MJD'], data[data['Filter'] == 'g']['mag'] + 2, label='g + 2',c=colors[0])
        plt.scatter(data[data['Filter'] == 'r']['MJD'], data[data['Filter'] == 'r']['mag'], label='r',c=colors[1]) 
        plt.scatter(data[data['Filter'] == 'i']['MJD'], data[data['Filter'] == 'i']['mag'] - 2, label='i - 2', c=colors[2])

        plt.plot(phases, lcs[f'lsstg'] + 2, label = f'lsstg + 2', c=colors[0])
        plt.plot(phases, lcs[f'lsstr'], label = f'lsstr', c=colors[1])
        plt.plot(phases, lcs[f'lssti'] - 2, label = f'lssti - 2', c=colors[2])

        plt.xlabel('Phase')
        plt.ylabel('Apparent Mag')

        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(linestyle="--")

        plt.title(f'Interpolated Data: mej = {mej} phi = {phi} cos theta = {cos_theta}')
        plt.show()
    
    def plot_mag_vs_mej():


        k = np.array([1,2,3,4,5,6,7,8,9])
        mej_vals = np.concatenate((k*0.001, k*0.01, k*0.1))
        mej_mag = {}
        discovery_mag = {}

        for band in lsst_bands:
            mej_mag[band] = []
            discovery_mag[band] = []

        for mej in mej_vals:
            temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = 1)
            lcs = temp.buildLsstLC(scaling='avg')
            
            for band in lcs:
                mej_mag[band].append(min(lcs[band]))

                min_mag = min(lcs[band])
                
                idx = lcs[band] < 23
                
                if min_mag < 23:

                    discovery_mag[band].append((lcs[band][idx])[0])
                else:
                    discovery_mag[band].append(np.nan)

        for band in lcs:
            plt.plot(mej_vals, mej_mag[band], label = band)

        plt.axvspan(xmin=0.01, xmax=0.09, color='r', alpha=0.5)
        plt.xlabel('mej')
        plt.ylabel('min mag')
        #plt.xscale('log')
        plt.gca().invert_yaxis()
        plt.legend()

        plt.show()

        for band in lcs:
            plt.plot(mej_vals, discovery_mag[band], label = band)

        plt.axvspan(xmin=0.01, xmax=0.09, color='r', alpha=0.5)
        plt.xlabel('mej')
        plt.ylabel('min mag')
        plt.xscale('log')
        plt.legend()

        plt.show()

    def plot_spectra_at_mej():
        mej_vals = [0.001, 0.002, 0.005, 0.007, 0.01, 0.025, 0.05, 0.075, 0.11, 0.3, 0.7, 0.9]

        # fig, ax = plt.subplots(3, 4,subplot_kw={"projection": "3d"})
        fig, ax = plt.subplots(3, 4)

        for i, mej in enumerate(mej_vals):
            print(f"mej {mej}")
            temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = 1)
            #temp.makeSedPlot([1,1.1117677,1.2])
            spectra = temp.getSed([1], scaling='max').reshape((len(lmbd)))
            ax[int(i/4)][i%4].plot(lmbd, spectra)
            ax[int(i/4)][i%4].set_title(f'mej: {mej}, flux scaling factor: {temp.scaling_factor:4f}')
            # ax[int(i/4),i%4].set_xlabel('Wavelength (A)')
            # ax[int(i/4),i%4].set_ylabel('Spectral flux density (erg / s / cm^2 / A)')
            #plt.savefig(f'SED_mej_{mej}.png')
        plt.show()

    def plot_spectra_at_mej_3d(scaling='avg'):
        mej_vals = [0.001, 0.002, 0.005, 0.007, 0.01, 0.025, 0.05, 0.075, 0.11, 0.3, 0.7, 0.9]

        fig, ax = plt.subplots(3, 4,subplot_kw={"projection": "3d"})

        fig.suptitle(f'Flux scaling used: {scaling}')


        for i, mej in enumerate(mej_vals):
            print(f"mej {mej}")
            temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = 1)
            spectra = temp.getSed(phases, scaling=scaling)

            X, Y = np.meshgrid(lmbd, phases)

            # Plot the surface.
            ax[int(i/4)][i%4].plot_surface(X, Y, spectra, cmap=cm.plasma)
            ax[int(i/4)][i%4].set_title(f'mej: {mej}, flux scaling factor: {temp.scaling_factor:4f}')
            ax[int(i/4)][i%4].set_xlabel('Wavelength (A)', fontsize = 5)
            ax[int(i/4)][i%4].set_ylabel('Phase (days)', fontsize = 5)
            ax[int(i/4)][i%4].set_zlabel('Spectral flux density (erg / s / cm^2 / A)', fontsize = 5)
            # ax[int(i/4),i%4].set_xlabel('Wavelength (A)')
            # ax[int(i/4),i%4].set_ylabel('Spectral flux density (erg / s / cm^2 / A)')
            #plt.savefig(f'SED_mej_{mej}.png')
        plt.show()
    
    def plot_plot_mej_power_fits():
    
        # mej_vals = np.arange(start=0.001, stop=0.9, step=0.001)
        # mej_vals = np.concatenate((k*0.001, k*0.01, k*0.1))
        mej_vals = np.arange(start=0.01, stop=0.1, step=0.005)

        peak_flux = []
        peak_flux_wavelength = []

        for i, mej in enumerate(mej_vals):
            temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = 1)
            spectra = temp.getInterpolatedSed(phases).reshape((len(lmbd)))
            idx = np.argmax(spectra)
            peak_flux.append(np.sum(spectra))
            peak_flux_wavelength.append(lmbd[idx])
        
        fit_mej = np.arange(start=0.001, stop=0.9, step=0.001)

        deg = 1
        p = np.polyfit(np.log10(mej_vals), np.log10(peak_flux), deg=deg)
        n = p[0] 
        log_a = p[1]
        a = 10**log_a
        fit = a * (fit_mej**n) #+ p[1] #* (fit_mej) + p[2]
        plt.scatter(mej_vals, peak_flux, c=peak_flux_wavelength, cmap='plasma')
        plt.plot(fit_mej, fit, label='Best power law fit')
        plt.xlabel('mej')
        plt.ylabel('avg flux')
        plt.colorbar()
        plt.legend()
        plt.title(rf'$y = {a:2f} \cdot x^{n:2f}$')
        plt.xscale('log')
        plt.show()

    def plot_plot_mej_poly_fits():
    
        # mej_vals = np.arange(start=0.001, stop=0.9, step=0.001)
        # mej_vals = np.concatenate((k*0.001, k*0.01, k*0.1))
        mej_vals = np.arange(start=0.01, stop=0.1, step=0.005)

        peak_flux = []
        peak_flux_wavelength = []

        for i, mej in enumerate(mej_vals):
            temp = SEDDerviedLC(mej = mej, phi = 75, cos_theta = 0.9, dist=d, coord=c, av = 1)
            spectra = temp.getInterpolatedSed([1]).reshape((len(lmbd)))
            idx = np.argmax(spectra)
            peak_flux.append(np.mean(spectra))
            peak_flux_wavelength.append(lmbd[idx])
        
        fit_mej = np.arange(start=0.001, stop=0.9, step=0.001)

        deg = 2
        p = np.polyfit(mej_vals, peak_flux, deg=deg)

        fit = p[0] * (fit_mej**2) + p[1] * (fit_mej) + p[2]
        plt.scatter(mej_vals, peak_flux, c=peak_flux_wavelength, cmap='plasma')
        plt.plot(fit_mej, fit, label='Best power law fit')
        plt.xlabel('mej')
        plt.ylabel('avg flux')
        plt.colorbar()
        plt.legend()
        plt.loglog
        plt.show()

    def plotBulla19Plot():
        # https://watermark.silverchair.com/stz2495.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsYwggLCBgkqhkiG9w0BBwagggKzMIICrwIBADCCAqgGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMEuJWAI9tbfCgTjd2AgEQgIICefaJ5S02v8ICGEVAC-wHLh4LI0lNpWecSmLD807vmp2vMro9_uAlcH7kSj3w3_WXlkdCnw3XN6rl3U0atnCCWLon9nIW826JFjkxrPImGJ1opI4pQ3F-bjC3Oqg-M-A07lVLf9OIvIU7xESH_1jcMdfvatVEn1bSGI0AZNK92omwMAW5Kg6ena30gUbtrl6hFC3qlrk5EIFVBSSeXqz0MRxV0b51wJK1fI5Nu1m6kqeLyTFyJfDvKZzRbo51K14XpXrl4iVhI7QYRowpLtS1LSe2fUAe5m3ng9kdMhkrFnaLUTKbYH2hpKWM4AS8eNmGN2thAzP_hiXIq-DEWH0qvj9ZBy0QaXr8PGJL2c5Msc0urIxRDj6WtpE7Qw7sSy-bOTVUYDycX1g_EcBn2GFdxsBcsjaek3rdiUa6MzKcyiiClKxMKAP4ZAXARhOTZPFZT_G9moE47wYtgJAQu9zaOPlk-hq5pYRRe8xmcUx_CaCUbp2bSESlJKNUodF_KHOhnvpC3TgRJmt7_oLnkgtIXV_41btDh1wPsqpYV7PBrVxQ7nXybqOYH9bO6lz9dh2HJ9qQdmiPiSyNYIqqdx2MWWTklPBEehIfRvZw6TaDTegKFTQ_gyUFZVcYlS-mZJ_qQ7-ZMdwB8CxSUQjaiMqWU9rfV48KqM3RTROE7GyIhikxeh5ux0rcqGiF8ufSfNRG-o626pKcDP0k8DwW84-q9XnB__EuH2C-YiO7dDnicEy5mmVksYV_Rduum6lzF5Ja-rSDvX5EjzEhVw_i-xTcWwXEXscVE4IXSMhqoS6F-wyzJAJjJ40ZptBy6J5nA7kIlZbJe2BTXQufXw
        cos_theta = 1
        phi = 30
        mej_vals = [0.02, 0.04, 0.06, 0.08, 0.1]
        d = 40 * u.Mpc

        phases = np.arange(start=0.5, stop=10, step=0.5)
        bands = ['lsstu','lsstg','lsstr','lssti','lsstz','lssty']


        fig, ax = plt.subplots(4, 4)

        for i, band in enumerate(bands):

            lcs = {}

            for mej in mej_vals:
                temp = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = 0.1)
                lc = temp.getAbsMagsInPB([band], phases) + Distance(d).distmod.value
                
                lcs[mej] = lc

                x = int(i/4) * 2
                y = i%4
                ax[x][y].plot(phases, lc, label=f'{mej}')
                ax[x][y].invert_yaxis()

                ax[x][y].set_ylabel('Mag')
                ax[x][y].set_title(band)
                ax[x][y].legend()



            for mej in mej_vals:

                x = int(i/4) * 2 + 1
                y = i%4

                lc = lcs[0.04] - lcs[mej] 
                ax[x][y].plot(phases, lc)
                ax[x][y].set_ylabel(r'$\Delta$')

                ax[x][y].set_aspect(0.8)

                if x == 3:
                    ax[x][y].set_xlabel('Phase (days)')

        plt.show()








    #plot_GW170817_lc_and_spectra()
    #plot_mag_vs_mej()
    #plot_spectra_at_mej()
    #plot_plot_mej_power_fits()
    #plot_spectra_at_mej_3d(scaling='avg')
    #plot_plot_mej_poly_fits()
    plotBulla19Plot()