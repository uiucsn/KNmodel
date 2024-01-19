import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from astropy.io import ascii
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# Grid values for Bulla m3 model: https://github.com/mbulla/kilonova_models/tree/87a25e1c4dd1d7b18a0dfa59808672e36978313d/bns_m3_3comp
uniq_cos_theta = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 11 counts
uniq_mej_dyn  = np.array([0.001, 0.005, 0.01, 0.02]) # 4 counts   
uniq_mej_wind =  np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]) # 7 counts
uniq_phi = np.array([15, 30, 45, 60, 75]) # 5 counts

phases = np.arange(start=0.1, stop=20, step=0.2)
lmbd = np.arange(start=100, stop = 99901, step=200)

# Wrapper for a scipy Regular Grid Interpolator
class BullaSEDInterpolator():

    # Bulla Model SED
    class KnSED():

        def __init__(self, index, cos_theta, mej_dyn, mej_wind, phi, sed_file):

            self.index = index
            self.cos_theta = cos_theta # Observing angle
            self.mej_dyn = mej_dyn # Ejecta Mass
            self.mej_wind = mej_wind # Ejecta Mass
            self.phi = phi # half-opening angle of the lanthanide-rich component
            self.sed_file = sed_file # File containing the sed information

        def __str__(self):
            return f'Index: {self.index}\nCOS_THETA: {self.cos_theta}\nEJECTA_MASS_DYN: {self.mej_dyn}\nEJECTA_MASS_WIND: {self.mej_wind}\nPHI: {self.phi}\nFILE_NAME: {self.sed_file}'

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
            with open('data/Bulla_SED_Interpolator.pkl', 'rb') as f:
                self.interpolator = pickle.load(f)
    
    def fitLinearFunction(self, x, y):

        coeffs, residuals, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        m = coeffs[0]
        c = coeffs[1]

        return m, c, residuals
        
    def linearFunction(self, x, m, c):

        sol = m * x + c
        return sol

    def fitPowerLaw(self, x, y):

        coeffs, residuals, _, _, _ = np.polyfit(np.log10(x), np.log10(y), deg=1, full=True)

        log_a = coeffs[1]

        a = 10**log_a
        n = coeffs[0] 

        return a, n, residuals

    def powerFunction(self, x, a, n):

        sol = a * (x**n)
        return sol

    def computeFluxScalingLaws(self, plot=False, show_plot=False):
                    
        if plot:
            fig, ax = plt.subplots(4, len(uniq_phi))
            fig.set_size_inches(20, 20)
            font = {
                'size'   : 10}

            plt.rc('font', **font)

        df_linear = pd.DataFrame(columns=['cos_theta', 'phi', 'slope', 'intercept'])
        df_power = pd.DataFrame(columns=['cos_theta', 'phi', 'coefficient', 'exponent'])

        print('Computing flux scaling laws...')
        for i, cos_theta in tqdm(enumerate(uniq_cos_theta), total=len(uniq_cos_theta)):
            for j, phi in enumerate(uniq_phi):

                total_fluxes = []
                total_mej = []

                for k, mej_wind in enumerate(uniq_mej_wind):
                    for l, mej_dyn in enumerate(uniq_mej_dyn):

                        f = 0 # flux over all phases
                    
                        for phase in phases:

                            mesh_grid = np.meshgrid(cos_theta, mej_dyn, mej_wind, phi, phase, lmbd)
                            points = np.array(mesh_grid).T.reshape(-1, 6)
                            sed = self.interpolator(points)
                            f += np.sum(sed)

                        total_fluxes.append(f)
                        total_mej.append(mej_dyn + mej_wind)

            
                total_mej = np.array(total_mej)
                total_fluxes = np.array(total_fluxes)

                # Find best linear fit
                m, c, _ = self.fitLinearFunction(total_mej, total_fluxes)

                # Add fit to the data frame
                d_linear = {
                    'cos_theta': cos_theta,
                    'phi': phi,
                    'slope': m, 
                    'intercept': c,
                }
                d_linear = pd.DataFrame(d_linear, index=[0])
                df_linear = pd.concat([df_linear, d_linear], ignore_index = True)

                # Find best linear fit
                a, n, _ = self.fitPowerLaw(total_mej, total_fluxes)

                # Add fit to the data frame
                d_power = {
                    'cos_theta': cos_theta,
                    'phi': phi,
                    'coefficient': a, 
                    'exponent': n,
                }
                d_power = pd.DataFrame(d_power, index=[0])
                df_power = pd.concat([df_power, d_power], ignore_index = True)

                if plot:
                    
                    if i == 0 or i == 1:
                        # Plotting fits for linear laws
                        fit_mej = np.arange(start=0.001, stop=0.9, step=0.001)
                        fit = self.linearFunction(fit_mej, m, c)

                        ax[2*i][j].plot(fit_mej, fit, label=fr"$y = {m:.3f} \cdot x + {c:.3f}$", color='green')
                        ax[2*i][j].scatter(total_mej, total_fluxes,  marker='.')

                        # Plotting relative errors
                        grid_fit = self.linearFunction(total_mej, m, c)
                        relative_error = (grid_fit - total_fluxes) / total_fluxes 

                        ax[2*i][j].axvspan(xmin=np.min(total_mej), xmax=np.max(total_mej), alpha = 0.2, color='orange')
                        ax[2*i + 1][j].scatter(total_mej, relative_error, marker='.', color='green', label='Linear fit')



                        # Plotting fits for power scaling
                        fit_mej = np.arange(start=0.001, stop=0.9, step=0.001)
                        fit = self.powerFunction(fit_mej, a, n)

                        ax[2*i][j].plot(fit_mej, fit, label=f"$y = {a:.3f} \cdot x^{{{n:.3f}}}$", color='red')

                        ax[2*i][j].set_title(f'cos theta: {cos_theta}, phi: {phi}', fontsize=15, fontweight="bold")
                        ax[2*i][j].set_xscale('log')

                        ax[2*i][j].set_xlabel('mej', fontsize=0)
                        ax[2*i][j].set_ylabel('Total bolometric flux \n(over 20 days)', fontsize=12)


                        ax[2*i][j].set_xlim(left=np.min(fit_mej), right=np.max(fit_mej))

                        # Plotting relative errors
                        grid_fit = self.powerFunction(total_mej, a, n)
                        relative_error = (grid_fit - total_fluxes) / total_fluxes 

                        ax[2*i + 1][j].scatter(total_mej, relative_error, marker='.', color='red', label='Power fit')

                        ax[2*i + 1][j].set_xscale('log')
                        ax[2*i + 1][j].set_xlim(left=np.min(fit_mej), right=np.max(fit_mej))


                        ax[2*i + 1][j].set_xlabel('mej', fontsize=12)
                        ax[2*i + 1][j].set_ylabel(r"$\Delta F/F$", fontsize=12)

                        ax[2*i + 1][j].legend()
                        ax[2*i][j].legend()

        print('Saving flux scaling laws for Bulla m3...')

        df_linear.to_csv('data/m3_linear_scaling_laws.csv')
        df_power.to_csv('data/m3_power_scaling_laws.csv')

        print(df_linear.to_latex(index=False, label="table:linear-laws"))
        print(df_power.to_latex(index=False, label="table:power-laws"))

        if plot:
            plt.tight_layout()
            fig.savefig(f'paper_figures/scaling_fits.pdf')
            if show_plot:        
                plt.show()

    def buildFromSourceData(self, sed_dir = 'SEDs/SIMSED.BULLA-BNS-M3-3COMP/', sed_info_file = 'SED.INFO', bounds_error = False, to_plot=False):

        # Info file
        data = ascii.read(sed_dir + sed_info_file, data_start=7, names = ('TEMP','FILE', 'KN_INDEX', 'COSTHETA', 'MEJDYN', 'MEJWIND', 'PHI'), guess=False)
        
        uniq_phase = None # 100 counts
        uniq_wavelength = None # 50 counts

        uniq_cos_theta = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 11 counts
        uniq_mej_dyn  = np.array([0.001, 0.005, 0.01, 0.02]) # 4 counts   
        uniq_mej_wind =  np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]) # 7 counts
        uniq_phi = np.array([15, 30, 45, 60, 75]) # 5 counts

        # cos theta, mej_dyn, mej_wind, phi, phase, wavelength ordering followed
        arr = np.zeros((len(uniq_cos_theta), len(uniq_mej_dyn), len(uniq_mej_wind), len(uniq_phi), 100, 500))

        if to_plot:
            mejs_dyn = []
            mejs_wind = []
            phis = []
            costhetas = []

        print('Building grid...')

        for i in tqdm(range(len(data))):

            # Creating SED object and opening the corresponding file 
            sed = self.KnSED(data['KN_INDEX'][i], data['COSTHETA'][i], data['MEJDYN'][i], data['MEJWIND'][i], data['PHI'][i], sed_dir + data['FILE'][i])

            if sed.phi in uniq_phi:
                
                # Table contain sed
                table = pd.read_csv(sed.sed_file, delimiter=' ', names = ['Phase', 'Wavelength', 'Flux'])

                # Finding the unique phases and wavelength values. Same for all SED's 
                uniq_phase = np.unique(table['Phase'])
                uniq_wavelength = np.unique(table['Wavelength'])

                # Creating the mesh function for flux 
                flux_mesh = np.array(table['Flux']).reshape((len(uniq_phase), len(uniq_wavelength)))

                # Indices corresponding to sed parameters
                cos_idx = np.where(uniq_cos_theta == sed.cos_theta)[0]
                mej_wind_idx = np.where(uniq_mej_wind == sed.mej_wind)[0]
                mej_dyn_idx = np.where(uniq_mej_dyn == sed.mej_dyn)[0]
                phi_idx = np.where(uniq_phi == sed.phi)[0]

                # Adding the mesh the correct part 
                arr[cos_idx, mej_dyn_idx, mej_wind_idx, phi_idx, :, :] = flux_mesh

                if to_plot:
                    mejs_dyn.append(sed.mej_dyn)
                    mejs_wind.append(sed.mej_wind)
                    phis.append(sed.phi)
                    costhetas.append(sed.cos_theta)

        interpolator = RegularGridInterpolator((uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, uniq_phase, uniq_wavelength), arr, bounds_error=bounds_error, fill_value=None)

        print('Verifying interpolator...')
        
        # TEST: This loop verifies that all points on the grid are interpolated precisely

        for i in tqdm(range(len(data))):

            # Creating SED object and opening the corresponding file 
            sed = self.KnSED(data['KN_INDEX'][i], data['COSTHETA'][i], data['MEJDYN'][i], data['MEJWIND'][i], data['PHI'][i], sed_dir + data['FILE'][i])

            if sed.phi in uniq_phi:
                
                # Table contain sed
                table = pd.read_csv(sed.sed_file, delimiter=' ', names = ['Phase', 'Wavelength', 'Flux'])

                # Finding the unique phases and wavelength values. Same for all SED's 
                uniq_phase = np.unique(table['Phase'])
                uniq_wavelength = np.unique(table['Wavelength'])

                # Creating the mesh function for flux 
                real_sed = np.array(table['Flux']).reshape((len(uniq_phase), len(uniq_wavelength)))
                interpolated_sed = np.zeros_like(real_sed)

                for j, phase in enumerate(uniq_phase):
                    
                    mesh_grid = np.meshgrid(sed.cos_theta, sed.mej_dyn, sed.mej_wind, sed.phi, phase, uniq_wavelength)
                    points = np.array(mesh_grid).T.reshape(-1, 6)
                    interpolated_sed[j,:] = interpolator(points)
                
                # If interpolation at any grid point is incorrect, 
                assert np.array_equal(real_sed, interpolated_sed), f'Interpolator check failed at {sed}'
        
        print('Grid check successful! Saving...')

        # Pickle the file 
        with open('data/Bulla_SED_Interpolator.pkl', 'wb') as f:
            pickle.dump(interpolator, f)
            print('Done!')

        if to_plot:

            plt.hist(mejs_dyn)
            plt.xlabel('Dynamical Ejecta Mass')
            plt.ylabel('Count')
            plt.show()

            plt.hist(mejs_wind)
            plt.xlabel('Wind Ejecta Mass')
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
    i1 = BullaSEDInterpolator(from_source=True)
    i1.computeFluxScalingLaws()





    





