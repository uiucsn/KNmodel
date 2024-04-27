import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import matplotlib.ticker as ticker
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Distance
import afterglowpy as grb
import sncosmo
from tqdm import tqdm
import pickle

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands

from afterglow_addition import AfterglowAddition

COLOR = 'white'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

    
if __name__ == '__main__':

    params_grb = { # from Ryan 2020
        'E0': 10**52.96,
        'thetaCore': 0.066,
        'n0':10**-2.7,
        'p':2.17,
        'epsilon_e':10**-1.4, 
        'epsilon_B':10**-4.,
    }

    # Dietrich 2019
    mej_dyn = 10**-2.27
    mej_wind = 10**-1.28
    phi = 49.5

    theta = 0
    ct = 1 #np.cos(theta*u.deg.to(u.rad))
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 40*u.Mpc
    filename = 'caps'
    load = False
    if not load: 
        with open(f'data/sims/{filename}.pkl', 'wb') as f:
            KN = SEDDerviedLC(mej_dyn, mej_wind, phi, ct, dist=d, coord=c, av=0.0)
            afterglow = AfterglowAddition(KN, addKN=False) 
            data = np.array([KN, afterglow])
            pickle.dump(data, f)
    else:
        # load in the values
        with open(f'data/sims/{filename}.pkl', 'rb') as f:
            data = pickle.load(f)
    #t_day = phases
  
    KN = data[0]
    afterglow = data[1]

    for i, sed in enumerate([KN.sed, afterglow.sed, KN.sed + afterglow.sed]):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        cmap = cm['plasma']
        X, Y = np.meshgrid(phases, lmbd)
        Z = np.log10(sed)
        ax.plot_surface(X, Y, Z.T, lw=0.5, alpha=0.7,cmap=cmap)
        ax.set_xlabel('time [days]')
        

        ax.set_ylabel(r'wavelength [$\AA$]')
        ax.invert_yaxis()

        ax.set_zlabel(r'log $F_{\lambda}$')
        ax.view_init(elev=30., azim=-45)

        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # ax.zaxis.set_ticklabels([])

        plt.savefig(f'img/caps/{i}_sed_transp.png', transparent=True)


    #ax.yaxis.set_major_locator(ticker.LinearLocator())
    #ax.set_yticklabels(labels)

    plt.show()

    # first 10 days
    # idx_10 = np.where(np.isclose(phases, 10.1))[0][0]

    # # common band from uv to ir in sncosmo
    # sncosmo_bands = ['uvot::uvw2', 'uvot::uvw1', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty', 'f125w', 'f160w', 'f200w']
    # labels = ["uv2", "uv1", "u", "g", "r", "i", "z", "y", "J", "H", "K"]
    # labels_idx = np.arange(len(labels))

    # # abs mag of KN+afterglow
    # mag_band_aftKN = afterglow.getAbsMagsInPassbands(sncosmo_bands, lc_phases=phases[:idx_10])
    # mag_band_aftKN = np.array(np.array([list(item) for item in mag_band_aftKN.values()]))
    
    # # abs mag of KN
    # mag_band_KN = KN.getAbsMagsInPassbands(sncosmo_bands, lc_phases=phases[:idx_10])
    # mag_band_KN = np.array(np.array([list(item) for item in mag_band_KN.values()]))

    # # since each band is a row in the mag array, the y values are the bands (param help const thru row = y val)
    # X, Y = np.meshgrid(phases[:idx_10], labels_idx) # need 2D arrays from the 1D
    # Z = mag_band_aftKN - mag_band_KN # magnitude enhancement
