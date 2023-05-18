import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from astropy.io import ascii
from matplotlib.colors import LightSource
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator

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
    temp1 = BullaSEDInterpolator(from_source=True, bounds_error=False)
    temp2 = BullaSEDInterpolator(from_source=False)
    vals = (1.0, 0.09, 45, 0.1, 5500
    )
    print(temp1.interpolate(1.0, 0.09, 45, 0.1, 5500))
    print(temp2.interpolate(1.0, 0.09, 45, 0.1, 5500))