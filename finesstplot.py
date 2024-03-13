import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter, MultipleLocator
from matplotlib import colormaps as cm
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
from afterglow_addition import AfterglowAddition#, t_s, nu

# make the same plot made for NSF
t_day = np.arange(start=0.1, stop=22, step=0.2) # note finesst_iuv2 used 0.01 seperatation

# lc_model = SVDLightCurveModel(model='Bu2019lm', sample_times = t_day, interpolation_type='tensorflow', svd_path = 'svdmodels', mag_ncoeff=None,  lbol_ncoeff=None)
# grb_model = GRBLightCurveModel(t_day, resolution=12, jetType=0)

# params_grb = { # from Troja 2020 
#   'inclination_EM': 0,
#   'log10_E0': 52.9,
#   'thetaCore': 0.07,
#   'thetaWing': 0.47,
#   'log10_n0':-2.7,
#   'p':2.17,
#   'log10_epsilon_e':-1.4, 
#   'log10_epsilon_B':-4.,
#   'luminosity_distance': 40,}

# params_kn = { 
#   'log10_mej_dyn': -2.27,
#   'log10_mej_wind': -1.28,
#   'KNphi': 49.5,
#   'inclination_EM': 0, 
#   'luminosity_distance': 40}

# band = ztfi
save = False
if save == True:
    params_grb = { # from Troja 2020
        'E0': 10**52.9,
        'thetaCore': 0.07,
        'n0':10**-2.7,
        'p':2.17,
        'epsilon_e':10**-1.4, 
        'epsilon_B':10**-4.,
    }

    # Dietrich 2019
    mej_dyn = 10**-2.27
    mej_wind = 10**-1.28
    phi = 49.5

    theta = 4
    ct = np.cos(theta*u.deg.to(u.rad))
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 40*u.Mpc

    #t_day = phases
    KN = SEDDerviedLC(mej_dyn, mej_wind, phi, ct, dist=d, coord=c, av=0.0)
    KN.sed = KN.getSed(t_day, lmbd)  # interpolate to 2 days out past bulla grid  
    print(KN.sed.shape, flush=True)
    print(t_day.shape, lmbd.shape, flush=True)   
    afterglow = AfterglowAddition(KN, **params_grb, time = t_day, addKN=False) # use typical values
    tot = afterglow = AfterglowAddition(KN, **params_grb, time = t_day, addKN=True)


    with open(f'data/sims/finesst_22d.pkl', 'wb') as f:
        pickle.dump((KN, afterglow, tot), f)
else:
    with open(f'data/sims/finesst_22d.pkl', 'rb') as f:
        KN, afterglow, tot = pickle.load(f)

bands = ['ztfi', 'uvot::uvw2']
labels = [r'$i$-band', 'UV']
color = ['red', 'purple']

# note: default figsize is (6.4, 4.8)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

ax = axs[0]
mag = KN.getAbsMagsInPassbands(bands, lc_phases=t_day, apply_extinction=False)
mag_grb = afterglow.getAbsMagsInPassbands(bands, lc_phases=t_day, apply_extinction=False)
mag_tot = tot.getAbsMagsInPassbands(bands, lc_phases=t_day, apply_extinction=False)

for i, b in enumerate(bands):
    ax.plot(t_day, mag[b], label=labels[i]+' KN', color=color[i], linestyle='--')
    #ax.plot(t_day, mag_grb[b], color=color[i], linestyle='-.', label=f'Afterglow {b}')
    ax.plot(t_day, mag_tot[b], color=color[i], label=labels[i]+' Total')

ax.invert_yaxis()
#ax.set_ylim(-10, -18)
ax.set_xlim(0.1, 20)
ax.set_xlabel('t [days]', fontsize=16)
ax.tick_params(labelsize=14)
ax.set_ylabel('$M$ [mag]', fontsize=16)
ax.set_xscale('log')
ax.set_title(r'GW170817-like KNe+Afterglow viewed at $\theta=4^{\circ}$', fontsize=17)
ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax.yaxis.set_major_locator(MultipleLocator(base=4))

ax.legend(loc="lower left", fontsize=16)

# spectra plot
ax = axs[1]

# get the days of interest +1.5, +14, +21
idx_1d = np.where(np.isclose(t_day, 1.5))[0][0]
idx_14d = np.where(np.isclose(t_day, 14.1))[0][0]
idx_21d = np.where(np.isclose(t_day, 21.1))[0][0]

labels = ['+1.5 days', '+14 days', '+21 days']
colors = ['blue', 'red', 'green']
scale = 1/max(tot.sed[idx_1d][25:100])
print(np.argmax(tot.sed[idx_1d]), flush=True)
for i, idx in enumerate([idx_1d, idx_14d, idx_21d]):

    ax.plot(lmbd/1e4, scale*KN.sed[idx], color=colors[i], alpha=0.45)
    ax.plot(lmbd/1e4, scale*tot.sed[idx], color=colors[i], label=labels[i])
# plot specta at +1.5, +14, + 20

ax.set_xlim(0.3, 10) # 0.3 to 10 micron
ax.set_ylim(1e-4, 1.1)
ax.set_yscale('log')
ax.set_xlabel(r'Wavelength [$\mu m$]', fontsize=16)
ax.tick_params(labelsize=14)
ax.set_ylabel('Relative $F_{\lambda}$', fontsize=16)
ax.set_title(r'GW170817-like Spectra at 3 unique phases', fontsize=17)
#ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
#ax.yaxis.set_major_locator(MultipleLocator(base=4))

ax.legend(loc="upper right", fontsize=16)


plt.subplots_adjust(wspace=0.3)
fig.savefig('img/finesst_22d.png')