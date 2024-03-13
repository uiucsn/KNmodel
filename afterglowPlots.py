import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import matplotlib.ticker as ticker
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Distance
import afterglowpy as grb
import sncosmo
from tqdm import tqdm

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands

from afterglow_addition import AfterglowAddition

    
if __name__ == '__main__':

    # GW170817 object on axis
    mej_wind = 0.05
    mej_dyn = 0.005
    phi = 30
    theta = 7
    cos_theta = np.cos(theta*u.deg.to(u.rad))
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 40*u.Mpc

    KN = SEDDerviedLC(mej_dyn, mej_wind, phi, cos_theta, dist=d, coord=c, av=0.0)      
    afterglow = AfterglowAddition(KN)   

    # first 10 days
    idx_10 = np.where(np.isclose(phases, 10.1))[0][0]

    # common band from uv to ir in sncosmo
    sncosmo_bands = ['uvot::uvw2', 'uvot::uvw1', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty', 'f125w', 'f160w', 'f200w']
    labels = ["uv2", "uv1", "u", "g", "r", "i", "z", "y", "J", "H", "K"]
    labels_idx = np.arange(len(labels))

    # abs mag of KN+afterglow
    mag_band_aftKN = afterglow.getAbsMagsInPassbands(sncosmo_bands, lc_phases=phases[:idx_10])
    mag_band_aftKN = np.array(np.array([list(item) for item in mag_band_aftKN.values()]))
    
    # abs mag of KN
    mag_band_KN = KN.getAbsMagsInPassbands(sncosmo_bands, lc_phases=phases[:idx_10])
    mag_band_KN = np.array(np.array([list(item) for item in mag_band_KN.values()]))

    # since each band is a row in the mag array, the y values are the bands (param help const thru row = y val)
    X, Y = np.meshgrid(phases[:idx_10], labels_idx) # need 2D arrays from the 1D
    Z = mag_band_aftKN - mag_band_KN # magnitude enhancement

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    cmap = cm['bwr_r']
    ax.plot_surface(X, Y, Z, lw=0.5, alpha=0.5,cmap=cmap)

    ax.set_title(f'{np.round(theta,1)} deg off-axis')

    
    ax.set_xlabel('time (days)')

    ax.set_ylabel('band')
    ax.yaxis.set_major_locator(ticker.LinearLocator())
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_zlabel(r'$\delta M$')
    ax.invert_zaxis()
    ax.view_init(elev=30., azim=-45)
    plt.show()