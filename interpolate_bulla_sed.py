import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json

from astropy.io import ascii
from matplotlib.colors import LightSource
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

# Wrapper for a scipy Regular Grid Interpolator
class BullaSEDInterpolator():

    # Bulla Model SED
    class KnSED():

        def __init__(self, index, cos_theta, mej, phi, sed_file):

            self.index = index
            self.cos_theta = cos_theta # Observing angle
            self.mej = mej # Ejecta Mass
            self.phi = phi # half-opening angle of the lanthanide-rich component
            self.sed_file = sed_file # File containing the sed information

        def __str__(self):
            return f'Index: {self.index}\nCOS_THETA: {self.cos_theta}\nEJECTA_MASS: {self.mej}\nPHI: {self.phi}\nFILE_NAME: {self.sed_file}'

    def __init__(self, from_source = False,  bounds_error = False):
        """
        Wrapper the SED interpolator built using Bulla SED data. 

        Args:
            from_source (bool, optional): Set to True if you want to build the interpolator from scratch. Defaults to False.
                                          Set to True for the first time since the object is > 200 MB and too big for github.
            bounds_error (bool, optional): Set to True if you want to let the interpolator extrapolate beyond the grid. 
                                           Default is True to allow for extrapolation to small values of ejecta mass.
        Returns:
            None:
        """

        if from_source:
            # Build from scratch
            self.interpolator = self.buildFromSourceData(bounds_error=bounds_error)
        else:
            # Load the pickled object
            with open('Bulla_SED_Interpolator.pkl', 'rb') as f:
                self.interpolator = pickle.load(f)

    def interpolate(self, cos_theta, mej, phi, phase, wavelength):

        # return the interpolated result.
        return self.interpolator((cos_theta, mej, phi, phase, wavelength))
    
    def fitPowerLaw(self, x, y):

        coeffs, residuals, _, _, _ = np.polyfit(np.log10(x), np.log10(y), deg=1, full=True)

        log_a = coeffs[1]

        a = 10**log_a
        n = coeffs[0] 

        return a, n, residuals
    
    def oneMinusLogisticFunction(self, x, x_0, A, L, k):


        logistic = np.where((x-x_0) >= 0, 
                    L / (1 + np.exp(-k*(x-x_0))), 
                   (L * np.exp(k*(x-x_0)))/ (1 + np.exp(k*(x-x_0))))

        sol = L - logistic
        return sol
    
    def powerLaw(self, x, a, n):

        sol = a * (x**n)
        return sol
    
    def fitOneMinusLogisticFunction(self, x, y):

        L = np.max(y)
        popt, _ = curve_fit(self.oneMinusLogisticFunction, x, y, p0=[0.05, L, L, 100])
        return popt

    def computeMaximumFluxPhases(self, plot=False):
                    
        mej_vals = np.arange(start=0.01, stop=0.11, step=0.01)
        cos_theta_vals = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 11 counts
        phi_vals = np.array([15, 30, 45, 60, 75])
        phases = np.arange(start=0.1, stop=7.6, step=0.2)
        lmbd = np.arange(start=100, stop = 99901, step=200)

        df = pd.DataFrame(columns=['cos_theta', 'phi', 'max_flux_phase'])

        for i, cos_theta in enumerate(cos_theta_vals):
            for j, phi in enumerate(phi_vals):

                # Flux at different phases over all mej values
                total_flux_at_phases = []

                for k, phase in enumerate(phases):

                    avg_fluxes = []
                    max_fluxes = []

                    for mej in mej_vals:

                        mesh_grid = np.meshgrid(cos_theta, mej, phi, phase, lmbd)
                        points = np.array(mesh_grid).T.reshape(-1, 5)
                        sed = self.interpolator(points)
                        
                        avg_flux = np.mean(sed)

                        idx = np.argmax(sed)
                        max_flux = sed[idx]
                        peak_wavelength = lmbd[idx]

                        avg_fluxes.append(avg_flux)
                        max_fluxes.append(max_flux)

                    # Find sum of avg flux over all mej values
                    total_flux_at_phases.append(np.sum(avg_fluxes))

                idx = np.argmax(total_flux_at_phases)
                max_flux_phase = phases[idx]

                d = {
                    'cos_theta': cos_theta,
                    'phi': phi,
                    'max_flux_phase': max_flux_phase,
                }

                d = pd.DataFrame(d, index=[0])
                df = pd.concat([df, d], ignore_index = True)

        df.to_csv('data/scaling_phases.csv')

        if plot:
            plt.hist(df['max_flux_phase'])
            plt.xlabel('Phase for max flux (days)')
            plt.show()

    def computeFluxScalingLaws(self, plot=False, function_type = 'power'):

        function_types = ['power', 'logistic']
        if function_type not in function_types:
            raise ValueError(f"Invalid fitting method. Expected one of: {function_types}, got {function_type}" )
    

        mej_vals = np.arange(start=0.01, stop=0.11, step=0.01)
        cos_theta_vals = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 11 counts
        phi_vals = np.array([15, 30, 45, 60, 75])
        lmbd = np.arange(start=100, stop = 99901, step=200)

        if plot:
            fig, ax = plt.subplots(len(cos_theta_vals), len(phi_vals))
            fig.set_size_inches(60, 25)
            plt.rcParams.update({'font.size': 5})


        peak_phases = pd.read_csv('data/scaling_phases.csv')
        df = pd.DataFrame(columns=['phase','cos_theta', 'phi', 'a_max', 'n_max', 'a_avg', 'n_avg'])

        residuals_grid_max = np.zeros((len(cos_theta_vals), len(phi_vals)))
        residuals_grid_avg = np.zeros((len(cos_theta_vals), len(phi_vals)))

        for i, cos_theta in enumerate(cos_theta_vals):
            for j, phi in enumerate(phi_vals):

                # Phase for peak flux
                phase = peak_phases[(peak_phases['cos_theta']==cos_theta) & (peak_phases['phi']==phi)]['max_flux_phase'].to_numpy()[0]

                avg_fluxes = []
                max_fluxes = []
                peak_wavelengths = []

                for mej in mej_vals:

                    mesh_grid = np.meshgrid(cos_theta, mej, phi, phase, lmbd)
                    points = np.array(mesh_grid).T.reshape(-1, 5)
                    sed = self.interpolator(points)
                    
                    avg_flux = np.mean(sed)

                    idx = np.argmax(sed)
                    max_flux = sed[idx]
                    peak_wavelength = lmbd[idx]

                    avg_fluxes.append(avg_flux)
                    max_fluxes.append(max_flux)
                    peak_wavelengths.append(peak_wavelength)
                
                
                #Find best fit (1 - logistic curve) to the data. Anecdotally, it seems to reduce residuals
                if function_type=='logistic':

                    
                    try:
                        x_0_max, A_max, L_max, k_max = self.fitOneMinusLogisticFunction(mej_vals, max_fluxes)   
                        x_0_avg, A_avg, L_avg, k_avg = self.fitOneMinusLogisticFunction(mej_vals, avg_fluxes)

                        fit_vals_max = self.oneMinusLogisticFunction(mej_vals, x_0_max, A_max, L_max, k_max)
                        fit_vals_avg = self.oneMinusLogisticFunction(mej_vals, x_0_avg, A_avg, L_avg, k_avg)

                        avg_fluxes = np.array(avg_fluxes)
                        max_fluxes = np.array(max_fluxes)

                        residuals_avg = np.sum((fit_vals_avg - avg_fluxes)**2)
                        residuals_max = np.sum((fit_vals_max - max_fluxes)**2)

                        if plot:
                            
                            fit_mej = np.arange(start=0.001, stop=0.9, step=0.001)
                            fit = self.oneMinusLogisticFunction(fit_mej, x_0_avg, A_avg, L_avg, k_avg)

                            ax[i][j].scatter(mej_vals, avg_fluxes, c=peak_wavelengths, cmap='plasma')
                            ax[i][j].plot(fit_mej, fit)


                            ax[i][j].set_title(f'Peak phase: {phase:2f}, cos theta: {cos_theta}, phi: {phi}')
                            ax[i][j].set_xscale('log')


                            ax[i][j].tick_params(axis='both', which='major', labelsize=5)
                            ax[i][j].tick_params(axis='both', which='minor', labelsize=5)

                            # plt.colorbar()

                        #TODO: add the dataframe for logistic fits

                    except RuntimeError:
                            
                            print(f"phase {phase}, cos_theta {cos_theta}, phi {phi}: Failed to find params")
                            if plot:

                                ax[i][j].scatter(mej_vals, avg_fluxes, c=peak_wavelengths, cmap='plasma')

                                ax[i][j].set_title(f'Failed: Peak phase: {phase:2f}, cos theta: {cos_theta}, phi: {phi}')
                                ax[i][j].set_xscale('log')

                                ax[i][j].tick_params(axis='both', which='major', labelsize=5)
                                ax[i][j].tick_params(axis='both', which='minor', labelsize=5)

                # Find Best fit laws of the form y = a * (x ^ n) if phi
                elif function_type=='power':

                    a_max, n_max, residuals_max = self.fitPowerLaw(mej_vals, max_fluxes)
                    a_avg, n_avg, residuals_avg = self.fitPowerLaw(mej_vals, avg_fluxes)

                    d = {

                        'phase': phase,
                        'cos_theta': cos_theta,
                        'phi': phi,
                        'a_max': a_max,
                        'n_max': n_max,
                        'a_avg': a_avg,
                        'n_avg': n_avg,
                        'fit_residuals_avg': residuals_avg,
                        'fit_residuals_max': residuals_avg,

                    }

                    if plot:

                        fit_mej = np.arange(start=0.001, stop=0.9, step=0.001)
                        fit = self.powerLaw(fit_mej, a_avg, n_avg)

                        ax[i][j].scatter(mej_vals, avg_fluxes, c=peak_wavelengths, cmap='plasma')
                        ax[i][j].plot(fit_mej, fit)


                        ax[i][j].set_title(f'Phase: {phase:2f}, cos theta: {cos_theta}, phi: {phi}')
                        ax[i][j].set_xscale('log')

                        ax[i][j].tick_params(axis='both', which='major', labelsize=5)
                        ax[i][j].tick_params(axis='both', which='minor', labelsize=5)

                    d = pd.DataFrame(d, index=[0])
                    df = pd.concat([df, d], ignore_index = True)

                residuals_grid_max[i][j] += residuals_max
                residuals_grid_avg[i][j] += residuals_avg

        # Plot the surface.
        if plot:
            plt.show()
            fig.savefig(f'all_{function_type}_fits.pdf')
            # plt.imshow(residuals_grid_avg.T, cmap=cm.plasma)
            # plt.xlabel('cos theta')
            # plt.ylabel('phi')
            # plt.xticks(range(len(cos_theta_vals)),labels=cos_theta_vals)
            # plt.yticks(range(len(phi_vals)),labels=phi_vals)
            # plt.colorbar(label='Average residuals (over all phases)')
            # plt.show()
        
        print(df)
        df.to_csv('data/scaling_laws.csv')

    def buildFromSourceData(self, sed_dir = 'SEDs/SIMSED.BULLA-BNS-M2-2COMP/', sed_info_file = 'SED.INFO', bounds_error = False, to_plot = False):

        # Info file
        data = ascii.read(sed_dir + sed_info_file, data_start=7, names = ('TEMP','FILE', 'KN_INDEX', 'COSTHETA', 'MEJ', 'PHI'), guess=False)

        uniq_phase = None # 100 counts
        uniq_wavelength = None # 50 counts

        uniq_cos_theta = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 11 counts
        uniq_mej=  np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]) # 10 counts    
        uniq_phi = np.array([15, 30, 45, 60, 75]) # 5 counts

        # cos theta, meh, phi, phase, wavelength ordering followed
        arr = np.zeros((11, 10, 5, 100, 500))

        mejs = []
        phis = []
        costhetas = []

        for i in range(len(data)):

            print(f'{i}/{len(data)}\r')

            # Creating SED object and opening the corresponding file 
            sed = self.KnSED(data['KN_INDEX'][i], data['COSTHETA'][i], data['MEJ'][i], data['PHI'][i], sed_dir + data['FILE'][i])
            t = pd.read_csv(sed.sed_file, delimiter=' ', names = ['Phase', 'Wavelength', 'Flux'])

            # Finding the unique phases and wavelength values. Same for all SED's 
            uniq_phase = np.unique(t['Phase'])
            uniq_wavelength = np.unique(t['Wavelength'])

            # Creating the mesh function for flux 
            flux_mesh = np.array(t['Flux']).reshape((len(uniq_phase), len(uniq_wavelength)))

            # Indeces corresponding to sed parameters
            cos_idx = np.where(uniq_cos_theta == sed.cos_theta)[0]
            mej_idx = np.where(uniq_mej == sed.mej)[0]
            phi_idx = np.where(uniq_phi == sed.phi)[0]

            # # Replacing zero flux with a really small number
            # # This is to avoid negative flux values after extrapolating.
            # flux_mesh[flux_mesh == 0] = 10e-18

            # # Converting to log flux to ensure +ve flux value after extrapolation.
            # log_flux_mesh = np.log10(flux_mesh)

            # Adding the mesh the correct part 
            arr[cos_idx, mej_idx, phi_idx, :, :] = flux_mesh

            mejs.append(sed.mej)
            phis.append(sed.phi)
            costhetas.append(sed.cos_theta)

        interpolator = RegularGridInterpolator((uniq_cos_theta, uniq_mej, uniq_phi, uniq_phase, uniq_wavelength), arr, bounds_error=bounds_error, fill_value=None)

        # Sanity check: Checking correct interpolation at values 0.3, 0.09, 45, 0.1, 5500 with SED file
        assert arr[3,8,2,0,27] == interpolator((0.3, 0.09, 45, 0.1, 5500)), 'Estimator fails sanity check'

        # Pickle the file 
        with open('Bulla_SED_Interpolator.pkl', 'wb') as f:
            pickle.dump(interpolator, f)

        if to_plot:

            plt.hist(mejs)
            plt.xlabel('Ejecta Mass')
            plt.ylabel('Count')
            plt.show()

            plt.hist(phis)
            plt.xlabel('Phi')
            plt.ylabel('Count')
            plt.show()

            plt.hist(costhetas)
            plt.xlabel('Cos theta')
            plt.ylabel('Count')
            plt.show()

        return interpolator
if __name__ == '__main__':

    # temp1 = BullaSEDInterpolator(from_source=True, bounds_error=False)
    temp2 = BullaSEDInterpolator(from_source=False)

    temp2.computeMaximumFluxPhases()
    temp2.computeFluxScalingLaws(plot=True, function_type='power')
    







