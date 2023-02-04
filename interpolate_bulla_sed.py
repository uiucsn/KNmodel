import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import ascii
from matplotlib.colors import LightSource
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator

sed_dir = 'SEDs/SIMSED.BULLA-BNS-M2-2COMP/'
sed_info_file = 'SED.INFO'
to_plot = False

# Info file
data = ascii.read(sed_dir + sed_info_file, data_start=7, names = ('TEMP','FILE', 'KN_INDEX', 'COSTHETA', 'MEJ', 'PHI'), guess=False)

# Bulla Model SED
class KN_SED():

    def __init__(self, index, cos_theta, mej, phi, sed_file):

        self.index = index
        self.cos_theta = cos_theta # Observing angle
        self.mej = mej # Ejecta Mass
        self.phi = phi # half-opening angle of the lanthanide-rich component
        self.sed_file = sed_file # File containing the sed information

    def __str__(self):
        return f'Index: {self.index}\nCOS_THETA: {self.cos_theta}\nEJECTA_MASS: {self.mej}\nPHI: {self.phi}\nFILE_NAME: {self.sed_file}'

uniq_phase = None # 100 counts
uniq_wavelength = None # 50 counts

uniq_cos_theta = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 11 counts
uniq_mej=  np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]) # 10 counts    
uniq_phi = np.array([15, 30, 45, 60, 75]) # 5 counts

# cos theta, meh, phi, phase, wavelength ordering followed
arr = np.zeros((11, 10, 5, 100, 500))


for i in range(len(data)):

    print(f'{i}/{len(data)}\r')

    # Creating SED object and opening the corresponding file 
    sed = KN_SED(data['KN_INDEX'][i], data['COSTHETA'][i], data['MEJ'][i], data['PHI'][i], sed_dir + data['FILE'][i])
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

    # Adding the mesh the correct part 
    arr[cos_idx, mej_idx, phi_idx, :, :] = flux_mesh


np.save('Bulla_data.npy', arr)

interpolator = RegularGridInterpolator((uniq_cos_theta, uniq_mej, uniq_phi, uniq_phase, uniq_wavelength), arr)
val = interpolator((1.0, 0.09, 45, 0.1, 5500))
print(val)