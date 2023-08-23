import numpy as np
import matplotlib.pyplot as plt
import sncosmo
import re
import pandas as pd
import sfdmap

from scipy.interpolate import SmoothBivariateSpline
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import Distance, SkyCoord
from astropy.io import ascii
from matplotlib import cm

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_mej_dyn, uniq_mej_wind, phases, lmbd


mej_dyn_grid_low = np.min(uniq_mej_dyn)
mej_dyn_grid_high = np.max(uniq_mej_dyn)

mej_wind_grid_low = np.min(uniq_mej_wind)
mej_wind_grid_high = np.max(uniq_mej_wind)


# Passbands for different surveys
lsst_bands = ['lsstg','lssti','lsstr','lsstu','lssty','lsstz']
jwst_NIRcam_bands = ['f200w']
hst_bands = ['uvf625w']

# Code to convert the SED's to lightcurves in different filters
class SEDDerviedLC(): 

    def __init__(self, mej_dyn, mej_wind, phi, cos_theta, dist, coord, av, rv = 3.1):

        # Setting paramters for interpolating SED
        self.mej_dyn = mej_dyn
        self.mej_wind = mej_wind
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

        # Call the important functions
        self._setNearestGridMejPoints()
        self._setLinearScalingFactor()
        self._setPowerScalingFactor()

        # Compute the SED
        self.sed = self.getSed(phases, lmbd)

    def _setNearestGridMejPoints(self):

        # Find nearest grid neighbors for mej_wind
        if self.mej_wind > mej_wind_grid_high:
            self.nearest_grid_mej_wind = mej_wind_grid_high 
            self.requires_mej_wind_extrapolation = True

        elif self.mej_wind < mej_wind_grid_low:
            self.nearest_grid_mej_wind = mej_wind_grid_low 
            self.requires_mej_wind_extrapolation = True

        else: 
            self.nearest_grid_mej_wind = self.mej_wind
            self.requires_mej_wind_extrapolation = False

        # Find nearest grid neighbors for mej_dyn
        if self.mej_dyn > mej_dyn_grid_high:
            self.nearest_grid_mej_dyn = mej_dyn_grid_high 
            self.requires_mej_dyn_extrapolation = True

        elif self.mej_wind < mej_wind_grid_low:
            self.nearest_grid_mej_dyn = mej_dyn_grid_low 
            self.requires_mej_dyn_extrapolation = True

        else: 
            self.nearest_grid_mej_dyn = self.mej_dyn
            self.requires_mej_dyn_extrapolation = False

    def _getInterpolatedSed(self, phases = phases, wavelengths = lmbd, remove_negative = True):

        interpolated_sed = np.zeros((len(phases), len(wavelengths)))

        # Get points mesh
        mesh_grid = np.meshgrid(self.cos_theta, self.nearest_grid_mej_dyn, self.nearest_grid_mej_wind, self.phi, phases, wavelengths)

        points = np.array(mesh_grid).T.reshape(-1, 6)

        # Interpolate spectral luminosity at all wavelengths
        spectral_flux_density = self.sed_interpolator.interpolator(points)

        if remove_negative:
            spectral_flux_density[spectral_flux_density < 0]  = 0 

        #interpolated_sed[i, :] = spectral_flux_density
        spectral_flux_density = spectral_flux_density.reshape((len(wavelengths), len(phases))).T
        return spectral_flux_density

    def _getLinearLawParameters(self):

        df_linear = pd.read_csv("data/m3_linear_scaling_laws.csv")

        phi = df_linear['phi']
        cos_theta = df_linear['cos_theta']
        slopes = df_linear['slope']
        intercepts = df_linear['intercept']

        m_func = SmoothBivariateSpline(phi, cos_theta, slopes) 
        c_func = SmoothBivariateSpline(phi, cos_theta, intercepts) 

        m = m_func.ev(self.phi, self.cos_theta)
        c = c_func.ev(self.phi, self.cos_theta)

        return m, c 

    def _getPowerLawParameters(self):

        df_power = pd.read_csv("data/m3_power_scaling_laws.csv")

        phi = df_power['phi']
        cos_theta = df_power['cos_theta']
        coefs = df_power['coefficient']
        exponents = df_power['exponent']

        a_func = SmoothBivariateSpline(phi, cos_theta, coefs) 
        n_func = SmoothBivariateSpline(phi, cos_theta, exponents) 

        a = a_func.ev(self.phi, self.cos_theta)
        n = n_func.ev(self.phi, self.cos_theta)

        return a, n 
  
    def _setLinearScalingFactor(self):
        
        m, c = self._getLinearLawParameters()

        # scaling values
        real_total_mej = self.mej_dyn + self.mej_wind
        closest_total_mej = self.nearest_grid_mej_dyn + self.nearest_grid_mej_wind

        # Find the scaling constant using law based on total bolometric flux across all phases
        scaling_factor = (m * real_total_mej + c) / (m * closest_total_mej + c)

        self.linear_scaling_factor = scaling_factor
    
    def _setPowerScalingFactor(self):

        a, n = self._getPowerLawParameters()

        # scaling values
        real_total_mej = self.mej_dyn + self.mej_wind
        closest_total_mej = self.nearest_grid_mej_dyn + self.nearest_grid_mej_wind

        # Find the scaling constant using law based on total bolometric flux across all phases
        scaling_factor = (real_total_mej ** n) / (closest_total_mej ** n)

        self.power_scaling_factor = scaling_factor
    
    def getSed(self, phases=phases, wavelengths = lmbd, scaling='piecewise'):

        scaling_types = ['power', 'linear', 'piecewise']
        if scaling not in scaling_types:
            raise ValueError(f"Invalid fitting method. Expected one of: {scaling_types}, got {scaling}" )
        
        # This finds the closest sed on the grid.
        closest_sed = self._getInterpolatedSed(phases=phases, wavelengths=wavelengths, remove_negative=True)

        # Scale appropriately
        if scaling == "power":
            self.scaling_factor = self.power_scaling_factor
        elif scaling == "linear":
            self.scaling_factor = self.linear_scaling_factor
        elif scaling == 'piecewise':

            if (self.mej_dyn + self.mej_wind) < (mej_dyn_grid_low + mej_wind_grid_low):
                # use power law to extrapolate on the low end
                self.scaling_factor = self.power_scaling_factor
            else: 
                # use linear law to extrapolate on the high end
                self.scaling_factor = self.linear_scaling_factor 
                
        sed = self.scaling_factor * closest_sed
        return sed

    def makeSedPlot(self):

        source_name = f"Interpolated Object\ncos_theta: {self.cos_theta}, mej_dyn: {self.mej_dyn}, mej_wind: {self.mej_wind}, phi: {self.phi}"

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X, Y = np.meshgrid(lmbd[:200], phases[:50])


        # Plot the surface.
        surf = ax.plot_surface(X, Y, self.sed[:50, :200], cmap=cm.plasma)

        # ax.axes.set_xlim3d(left=0, right=30000) 
        # ax.axes.set_ylim3d(bottom=0, top=10) 

        ax.set_xlabel('Wavelength (A)')
        ax.set_ylabel('Phase (days)')
        ax.set_zlabel('Spectral flux density (erg / s / cm^2 / A)')
        ax.set_title(source_name)
        return ax

    
    def getAbsMagsInPassbands(self, passbands, lc_phases = phases, apply_extinction = True, apply_redshift = True):

        lcs = {}
        
        for passband in passbands:
            source_name = f"Interpolated Object\ncos_theta: {self.cos_theta}, mej_dyn: {self.mej_dyn}, mej_wind: {self.mej_wind}, phi: {self.phi}"

            interpolated_sed = self.sed

            source = sncosmo.TimeSeriesSource(phase=phases, wave=lmbd, flux = interpolated_sed, name=source_name, zero_before=True)

            model = sncosmo.Model(source)

            if apply_extinction:

                # add host galaxy extinction E(B-V)
                model.add_effect(sncosmo.CCM89Dust(), 'host', 'rest')
                model.set(hostebv = self.host_ebv)

                # add MW extinction to observing frame
                model.add_effect(sncosmo.F99Dust(), 'mw', 'obs')
                model.set(mwebv=self.mw_ebv)

            if apply_redshift:

                # Adding redshift based on distance: https://docs.astropy.org/en/stable/api/astropy.coordinates.Distance.html#astropy.coordinates.Distance.z
                z = self.distance.z
                model.set(z=z)

            abs_mags = model.bandmag(band=passband, time = lc_phases, magsys="ab")
            lcs[passband] = abs_mags

        return lcs

    def getAppMagsInPassbands(self, passbands, lc_phases = phases, apply_extinction = True, apply_redshift = True):

        # Get abs mags first
        lcs = self.getAbsMagsInPassbands(passbands, lc_phases=lc_phases, apply_extinction = apply_extinction, apply_redshift= apply_redshift)

        # Add the distance modulus
        for passband in passbands:
            lcs[passband] += self.distance.distmod.value

        return lcs

if __name__ == '__main__':


    def plot_GW170817_lc_and_spectra():

        # Pass band stuff
        bands = ['g','r','i']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
        mej_wind = 0.05
        mej_dyn = 0.005
        phi = 30
        cos_theta = 0.9

        # coordinates for GW170817
        c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        d = 43*u.Mpc

        av = 0.0

        # LC from sed
        GW170817 = SEDDerviedLC(mej_dyn=mej_dyn, mej_wind = mej_wind, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
        lcs = GW170817.getAppMagsInPassbands(lsst_bands)
        GW170817.makeSedPlot()
        plt.show()
        
        # table from https://iopscience.iop.org/article/10.3847/2041-8213/aa8fc7#apjlaa8fc7t2
        data = pd.read_csv('data/gw170817photometry.csv', delimiter='\t' )  
        data['mag'] = [float(re.findall("\d+\.\d+", i)[0]) for i in data['Mag [AB]']]

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

        plt.title(f'Interpolated Data: mej_total = {mej_dyn + mej_wind} phi = {phi} cos theta = {cos_theta}')
        plt.show()
    
    def plot_mag_vs_mej():

        # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
        mej_wind = 0.05
        mej_dyn = 0.001
        phi = 30
        cos_theta = 0.9

        # coordinates for GW170817
        c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        d = 43*u.Mpc

        av = 0.1

        k = np.array([1,2,3,4,5,6,7,8,9])
        mej_vals = np.concatenate((k*0.001, k*0.01, k*0.1))
        mej_mag = {}


        for band in lsst_bands:
            mej_mag[band] = []


        for mej in mej_vals:

            GW170817 = SEDDerviedLC(mej_dyn=0.003, mej_wind = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
            lcs = GW170817.getAppMagsInPassbands(lsst_bands)
            
            for band in lcs:
                mej_mag[band].append(min(lcs[band]))


        for band in lcs:
            plt.plot(mej_vals, mej_mag[band], label = band)

        plt.axvspan(xmin=0.01, xmax=0.09, color='r', alpha=0.5)
        plt.xlabel('mej')
        plt.ylabel('min mag')
        #plt.xscale('log')
        plt.gca().invert_yaxis()
        plt.legend()

        plt.show()

    def plot_spectra_at_mej():
        mej_vals = [0.001, 0.002, 0.005, 0.007, 0.01, 0.025, 0.05, 0.075, 0.11, 0.3, 0.7, 0.9]

        # fig, ax = plt.subplots(3, 4,subplot_kw={"projection": "3d"})
        fig, ax = plt.subplots(3, 4)
        fig.set_size_inches(20, 20)
        # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
        mej_wind = 0.05
        mej_dyn = 0.001
        phi = 30
        cos_theta = 0.9

        # coordinates for GW170817
        c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        d = 43*u.Mpc

        av = 0.1

        for i, mej in enumerate(mej_vals):
            print(f"mej {mej}")
            temp = SEDDerviedLC(mej_dyn=0.003, mej_wind = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
            #temp.makeSedPlot([1,1.1117677,1.2])
            spectra = temp.getSed(phases=[1]).reshape((len(lmbd)))
            ax[int(i/4)][i%4].plot(lmbd, spectra)
            ax[int(i/4)][i%4].set_title(f'mej: {mej}, flux scaling factor: {temp.scaling_factor:4f}')
            # ax[int(i/4),i%4].set_xlabel('Wavelength (A)')
            # ax[int(i/4),i%4].set_ylabel('Spectral flux density (erg / s / cm^2 / A)')
            #plt.savefig(f'SED_mej_{mej}.png')
        fig.savefig('temp.png')
        plt.show()

    def plot_spectra_at_mej_3d(scaling='avg'):

        # fig, ax = plt.subplots(3, 4,subplot_kw={"projection": "3d"})
        fig, ax = plt.subplots(3, 4)

        # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
        mej_wind = 0.05
        mej_dyn = 0.001
        phi = 30
        cos_theta = 0.9

        # coordinates for GW170817
        c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        d = 43*u.Mpc
        av = 0.1
        mej_vals = [0.001, 0.002, 0.005, 0.007, 0.01, 0.025, 0.05, 0.075, 0.11, 0.3, 0.7, 0.9]

        fig, ax = plt.subplots(3, 4,subplot_kw={"projection": "3d"})

        fig.suptitle(f'Flux scaling used: {scaling}')


        for i, mej in enumerate(mej_vals):
            temp = SEDDerviedLC(mej_dyn=0.003, mej_wind = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
            spectra = temp.getSed(phases)

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


    plot_GW170817_lc_and_spectra()
    #plot_mag_vs_mej()
    #plot_spectra_at_mej()
    #plot_spectra_at_mej_3d(scaling='avg')
