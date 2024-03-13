import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance, SkyCoord
import astropy.coordinates as coord
import afterglowpy as grb
from astropy.table import Table
import scipy.stats as sts
from scipy.interpolate import interp1d

# load and format Fong et al 2015 CDF for n0
    # assumes e_e = 0.1, e_B = 0.01
FONG15_N0_FILE = 'data/fong15_n0_eb0.01.csv'
N0_CDF = Table.read(FONG15_N0_FILE, format='csv')
N0_CDF.sort('n')
N0_CDF['logn'] = np.log10(N0_CDF['n'])

CDF_interpolator_N0 = interp1d(N0_CDF['cdf'], N0_CDF['logn'], kind='linear', fill_value='extrapolate')

# load and format Rouco Escorial et al. 2023 
RE_THETA_FILE = 'data/RE23_theta_core.csv'
THETA_PDF_f = Table.read(RE_THETA_FILE, format='csv') # data in full
pdf_norm_f = THETA_PDF_f['pdf'][1:] / np.sum(THETA_PDF_f['pdf'][1:]*np.diff(THETA_PDF_f['theta'])) # normalize
THETA_interpolator_full = interp1d(np.cumsum(pdf_norm_f*np.diff(THETA_PDF_f['theta'])), THETA_PDF_f['theta'][1:], kind='linear', fill_value='extrapolate')

    # repeat for theta <= 12.4 deg (cut off other peaks)
THETA_PDF = THETA_PDF_f[THETA_PDF_f['theta'] <= 12.4]
pdf_norm = THETA_PDF['pdf'][1:] / np.sum(THETA_PDF['pdf'][1:]*np.diff(THETA_PDF['theta'])) # normalize
THETA_interpolator = interp1d(np.cumsum(pdf_norm*np.diff(THETA_PDF['theta'])), THETA_PDF['theta'][1:], kind='linear', fill_value='extrapolate')

#powerlaw idx from obs'd grbs - Fong15
FONG15_P_FILE = 'data/fong15_p.csv'
P_CDF = Table.read(FONG15_P_FILE, format='csv')

CDF_interpolator_P = interp1d(P_CDF['cdf'][1:], P_CDF['p'][1:], kind='linear', fill_value='extrapolate')


def get_opening_angle(n, distr='RE23'):

    # limits for trunc norm
    theta_min = 0.3 # based on smallest theta min from ER23
    theta_max = 90
    
    # mean = 5.35 # obtained via mean/std of the kde of the distr evaluated 1e8 times 
    # std = 2.97          # as the kde pdf = followed a gauss
    # sts.truncnorm.rvs(a=(theta_min - mean)/std, b=(theta_max - mean)/std, loc=mean, scale = std, size=n)

    u = sts.uniform.rvs(size=n)

    if distr == 'RE23 full': # TODO
        theta =  THETA_interpolator_full(u)
    if distr == 'RE23':
        theta = THETA_interpolator(u)
    if distr =='Fong15':
        mean = 16
        std = 10
        theta = sts.truncnorm.rvs(a=(theta_min - mean)/std, b=(theta_max - mean)/std, loc=16, scale = 10, size=n)

    return theta

def get_logn0(n, distr='Fong15'):

    # assumes e_e = 0.1, e_B = 0.01
    if distr == 'Fong15':
        return CDF_interpolator_N0(sts.uniform.rvs(size=n))
    
def get_loge0(n, distr='Zhu22'):
    
    if distr == 'Zhu22':
        return sts.norm.rvs(49.3, 0.4, size=n)

def get_p(n, distr='Fong15'):

    if distr == 'Fong15':
        return CDF_interpolator_P(sts.uniform.rvs(size=n))
    
    if distr == 'Zhu22':
        mean, std = 2.25, 0.1
        pmin, pmax = 2, np.inf
        return sts.truncnorm.rvs(a=(pmin - mean)/std, b=(pmax - mean)/std, loc=mean, scale = std, size=n)
         


# get sample of opening angles based on  Rouco Escorial et al. 2023 (ER23)
    # n is the number of events to be generated
# def theta_core(n, distr):

#     # limits for trunc norm
#     theta_min = 0.3 # based on smallest theta min from ER23
#     theta_max = 90
    
#     #mean_full = 
#     #std_full = 
    
#     mean = 5.35 # obtained via mean/std of the kde of the distr evaluated 1e8 times 
#     std = 2.97          # as the kde pdf = followed a gauss

#     if distr == 'ER23 full': # TODO
#         theta =  sts.truncnorm.rvs(a=(theta_min - mean)/std, b=(theta_max - mean)/std, loc=mean, scale = std, size=n)
#     if distr == 'ER23':
#         theta = sts.truncnorm.rvs(a=(theta_min - mean)/std, b=(theta_max - mean)/std, loc=mean, scale = std, size=n) 
#     if distr =='Fong15':
#         theta = sts.truncnorm.rvs(a=(theta_min - mean)/std, b=(theta_max - mean)/std, loc=16, scale = 10, size=n)

#     return theta


if __name__ == '__main__':

    # double check that the interpolators capture the original data

    n = 100000

    fig, axs = plt.subplots(2, 2, figsize=(16,16))
    axs = axs.ravel()

    # check Fong15 
    axs[0].plot(N0_CDF['logn'], N0_CDF['cdf'], label='CDF from F15')
    samples = get_logn0(n)
        # bin up the samples and get cdf
    height, edges, _ = axs[0].hist(samples, density=True, bins=100)
        # cdf = cumulative area of the bins up to a point
    axs[0].plot(edges[1:], np.cumsum(height*np.diff(edges)), 'g--', label='CDF from samples')
    axs[0].set_title(r'$n_0$')

    # Fong et al p distribution
    axs[1].plot(P_CDF['p'][1:], P_CDF['cdf'][1:], label='CDF from F15')
    ps = [2.31, 2.29, 2.24, 2.24, 2.03, 2.39, 2.35, 2.30, 2.24, 2.12, 1.92, 2.29, 2.06, 2.97, 2.40, 2.27, 2.13, 2.65, 2.40, 2.64, 2.40, 2.36, 2.73, 2.49, 2.08, 2.27, 2.87, 2.08, 2.50, 2.7, 2.49, 2.57, 3.0, 2.4, 2.1, 2.27, 2.67, 2.4]
    axs[1].hist(ps, bins=10, color='b', label='sGRBs', density=True, edgecolor='black', alpha=0.7)

    samples = get_p(n)
    height, edges, _ = axs[1].hist(samples, density=True, bins=P_CDF['p'], alpha=0.5) # make bins the same
    axs[1].plot(edges[1:], np.cumsum(height*np.diff(edges)), 'g--', label='CDF from samples')
    axs[1].set_title(r'$p$')

    # check RE23
    edges, heights = THETA_PDF_f['theta'], THETA_PDF_f['pdf'][1:]
    axs[2].hist(edges[:-1], edges, weights=heights, edgecolor='black', alpha=0.7, density=True)
    samples = get_opening_angle(n, distr='RE23 full')
    axs[2].hist(samples, bins=edges, density=True, alpha=0.5)
    axs[2].set_title(r'$\theta$ Full')

    edges, heights = THETA_PDF['theta'], THETA_PDF['pdf'][1:]
    axs[3].hist(edges[:-1], edges, weights=heights, edgecolor='black', alpha=0.7, density=True)
    samples = get_opening_angle(n, distr='RE23')
    axs[3].hist(samples, bins=edges, density=True, alpha=0.5) # use the same bin edges
    axs[3].set_title(r'$\theta$ 1st Peak')

    fig.savefig(f'img/params_check.png')
    plt.show()






