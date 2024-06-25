import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colormaps as cm
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance, SkyCoord
import astropy.coordinates as coord
import afterglowpy as grb
import sncosmo
from tqdm import tqdm
import schwimmbad
import pickle
import scipy.stats as sts
from scipy.interpolate import interp1d
import sys
import argparse
import os


from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands
from afterglow_addition import AfterglowAddition
from dns_mass_distribution import Galaudage21, Farrow19
from monte_carlo_sims import get_ejecta_mass
from afterglow_params import get_logn0, get_opening_angle, get_loge0, get_p

# 10 days out
idx_10 = np.where(np.isclose(phases, 10.1))[0][0]
phases_10 = phases[:idx_10]

# common band from uv to ir in sncosmo
sncosmo_bands = ['uvot::uvw2', 'uvot::uvw1', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty', 'f125w', 'f160w', 'f200w']
labels = ["uv2", "uv1", r"$u$-band", r"$g$-band", "r", "i", "z", "y", "J", "H", "K"]
labels_idx = np.arange(len(labels))


def get_params(n, save=False, filename=''):

    if save:

        # loop over kn params - stolen from Ved unless otherwise noted
            # mej_dyn, mej_wind, phi, cos_theta, dist=d, coord=c, av steal from ved
        mass1, mass2 = Galaudage21(n)
        masses = np.array([mass1, mass2]).T
        mej_dyns, mej_winds = get_ejecta_mass(mass1, mass2)
        
        # simulate coordinates. Additional term ensures minimum distance of 0.05 Mpc
        box_size = 600 # increase?
        x = np.random.uniform(-box_size/2., box_size/2., n)*u.Mpc
        y = np.random.uniform(-box_size/2., box_size/2., n)*u.Mpc
        z = np.random.uniform(-box_size/2., box_size/2., n)*u.Mpc
        #dists = (x**2. + y**2. + z**2.)**0.5 + (0.05*u.Mpc)
        
    
        coords = np.empty(n, dtype=object)
        dists = np.empty(n, dtype=object)
        for i in range(n):
            r, dec, ra = coord.cartesian_to_spherical(x[i], y[i], z[i])
            coords[i] = coord.SkyCoord(ra=ra, dec=dec)
            dists[i] = r
        
        thetaCores = np.deg2rad(get_opening_angle(n, distr='RE23'))
        cos_thetas = np.random.uniform(np.cos(np.deg2rad(30)), 1, size=n) # for EK_aft
        #cos_thetas = np.cos(thetaCores/2) # for EK_aft0tc
        #cos_thetas = np.cos(np.deg2rad(cos_thetas))
        #cos_thetas = np.full((n,), 1) # on-axis for all 
        phis = np.random.uniform(15, 75, size=n)
        avs = np.random.exponential(0.334, size=n)*0.334
        rvs = np.full((n,), 3.1)

        param_names = ["mej_dyn", "mej_wind", "phi", "cos_theta", "dist", "coord", "av", "rv"]
        params_array = np.array([mej_dyns, mej_winds, phis, cos_thetas, dists, coords, avs, rvs]).T
        kn_params = [{param_names[i]: value for i, value in enumerate(row)} for row in params_array]
        print(f'done KN params {filename}', flush=True)
        # loop over aft params
        # self.grb_params = { # use the same params I used for nsf fig
        #         'jetType':     grb.jet.Gaussian,   # Gaussian jet!! - not flat
        #         'specType':    0,                  # Basic Synchrotron Spectrum
        #         'thetaObs':    theta_v,            # Viewing angle in radians
        #         'E0':          E0,                 # Isotropic-equivalent energy in erg
        #         'thetaCore':   0.07,               # Half-opening angle in radians
        #         'thetaWing':   0.47,               # Wing angle in radians
        #         'n0':          n0,                 # circumburst density in cm^{-2}
        #         'p':           2.17,               # electron energy distribution index
        #         'epsilon_e':   10**-1.4,           # epsilon_e
        #         'epsilon_B':   10**-4,             # epsilon_B
        #         'xi_N':        1.0,                # Fraction of electrons accelerated
        #         'd_L':         10*u.pc.to(u.cm),   # Luminosity distance in cm
        #     }
        
        # from Zhu et al 2022 I unless otherwise specified
        #logE0s = np.random.normal(49.3, 0.4**2, n) # ergs
        #logn0s = np.random.normal(-2, 0.4**2, n) # cm^-2
        #ps = np.random.normal(2.25, 0.1**2, n)   # spectral index
        # logees = np.random.normal(-1, 0.3**2, n)
        # logebs = np.random.normal(-2, 0.4**2, n)

        logE0s = get_loge0(n, distr='Fong15')
        ps = get_p(n, distr='Fong15')
        logn0s = get_logn0(n, distr='Fong15') #sts.norm.rvs(-2, 0.4**2, size=n)
        # fix ee and eb and use n0 distribution
            # e_e = 0.1, e_B = 0.01
        logees = np.full((n,), -1.0) #sts.norm.rvs(-1, 0.3**2, size=n)
        logebs = np.full((n,), -2.0) #sts.norm.rvs(-2, 0.4**2, size=n)

        # make afterglow param dicts
        aft_array = np.array([10**logE0s, thetaCores, 10**logn0s, ps, 10**logees, 10**logebs]).T
        aft_names = ["E0", "thetaCore", "n0", "p", "epsilon_e", "epsilon_B"]
        aft_params = [{aft_names[i]: value for i, value in enumerate(row)} for row in aft_array]
        print(f'done aft params {filename}', flush=True)
        params = list(np.array([kn_params, aft_params]).T)
        #print(params, flush=True)

        # save them
        with open(f'data/sims/{n}_params_{filename}.pkl', 'wb') as f:
            print(f'done params {filename}', flush=True)
            pickle.dump(params, f)
        #np.savetxt(f"{n}_events.csv", params, delimiter=",")
        with open(f'data/sims/{n}_masses_{filename}.pkl', 'wb') as f:
            print(f'done masses {filename}', flush=True)
            pickle.dump(masses, f)
    else:
        # load in the values
        with open(f'data/sims/{n}_params_{filename}.pkl', 'rb') as f:
            params = pickle.load(f)

    # list of dicts
    return params

# note: changed to afterglow curves not Z
def gen_event(params):

    kn_params, aft_params = params

    KN = SEDDerviedLC(**kn_params)
    afterglow = AfterglowAddition(KN, **aft_params, addKN=False)

    mag_band_aft = afterglow.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_aft = np.array([list(item) for item in mag_band_aft.values()])
    afterglow.sed += afterglow.KNsed # add the KN on top

    # get the diff and save those
    # abs mag of KN+afterglow
    mag_band_aftKN = afterglow.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_aftKN = np.array([list(item) for item in mag_band_aftKN.values()])
    
    # abs mag of KN
    mag_band_KN = KN.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_KN = np.array([list(item) for item in mag_band_KN.values()])

    # since each band is a row in the mag array, the y values are the bands (param help const thru row = y val)
    #Z = mag_band_aftKN - mag_band_KN # magnitude enhancement, (11, 50)
    return np.array([mag_band_aft, mag_band_aftKN, mag_band_KN]) # ()


# issue with NaNs in UV: check and interpolate over them
def smooth_out_Nans(lc):    
    if np.isnan(lc).any():
        not_nan_indices = np.logical_not(np.isnan(lc))
        interpolator = interp1d(phases[not_nan_indices], lc[not_nan_indices], kind='linear', fill_value="extrapolate")
        interpolated_data = interpolator(phases)
        return interpolated_data
    else:
        return lc


def gen_events(n, save=False, filename=''):

    if save: 

        params = get_params(n, save, filename)

        with schwimmbad.JoblibPool(5) as pool:
             values = np.array(pool.map(gen_event, params))

        with open(f'data/sims/{n}_events_{filename}.pkl', 'wb') as f:
             pickle.dump(values, f)

    else:
        # load in the values
        with open(f'data/sims/{n}_events_{filename}.pkl', 'rb') as f:
            values = pickle.load(f)

    return values

def plot(n, save, filename=''):

    # load them in
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)
        # 10 events each with a Z, mag aftKN and KN

    # values is a 3d array each entry is an 2d grid of enhancements
    distr = np.percentile(values[:,0], [16, 50, 84], axis=0)

    fig, axs = plt.subplots(int(len(sncosmo_bands)/2), 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T
    #axs[-1].set_axis_off() # dont need the last one
    for idx in labels_idx:
        ax = axs[idx]
        # plot distr 1 - median
        # fill btwn distr 0 and 2
        ax.fill_between(phases, distr[0][idx, :], distr[2][idx, :], alpha=0.3)
        ax.plot(phases, distr[1][idx, :])

        ax.set_xlabel('time (days)')    
        ax.set_ylabel(r'$\Delta M$')
        ax.invert_yaxis()
        ax.set_title(labels[idx])

    # ax = axs[-1]
    
    # event = values[6]
    # for idx in labels_idx[:3]:
    #     ax.plot(phases, event[1][idx, :], linestyle='-')
    #     ax.plot(phases, event[2][idx, :], linestyle='--')   

    #     ax.set_xlabel('time (days)')    
    #     ax.set_ylabel(r'$\Delta M$')
    #     ax.invert_yaxis()
    #     ax.set_xscale('log')
    #     ax.set_title(labels[idx])

    fig.tight_layout()
    fig.savefig(f'img/{n}_events_{filename}.png')
    plt.show()

def plot_avglc(n, save, filename='', log=False):

    # load them in
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)
        # 10 events each with a Z, mag aftKN and KN
    
    # overlay GW170817 like event
        # at: Dietrich fit Fig 
    # ang = 0.03 # core = 0.07
    # at2017gfo = SEDDerviedLC(mej_dyn=10**-2.27, mej_wind=10**-1.28, phi=49.5, cos_theta=np.cos(ang), 
    #                          coord=SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec"), dist = 40*u.Mpc, av=0.0)
    #     # afterglow: Table 3 (default in afterglow addition)
    # gw170817 = AfterglowAddition(at2017gfo, addKN=False)
    # lcs = gw170817.getAbsMagsInPassbands(sncosmo_bands)

    # values is a 3d array each entry is an 2d grid of enhancements
    #distr = np.percentile(values[:,0], [16, 50, 84], axis=0) # get Z
    distr_aft = np.percentile(values[:,0], [16, 50, 84], axis=0)
    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) # get magAftKN
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)

    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(18, 8))
    plt.subplots_adjust(wspace=0.15, hspace=0.6)
    axs = axs.flatten().T
    #axs[-1].set_axis_off() # dont need the last one

    idx5 = np.where(np.isclose(phases, 5.1))[0][0]
    axs[0].set_ylabel(r'$M$')
    for i, idx in enumerate(labels_idx):
        ax = axs[i]
        # plot distr 1 - median
        # fill btwn distr 0 and 2
            # aft + KN
        ax.fill_between(phases, smooth_out_Nans(distr[0][idx, :]), smooth_out_Nans(distr[2][idx, :]), alpha=0.3, color='b')
        ax.plot(phases, smooth_out_Nans(distr[1][idx, :]), color='b', label='Afterglow + KN')
        #ax.plot(phases, lcs[sncosmo_bands[idx]], color='k', linestyle='--', label='GW170817 @ 2deg')
            # just KN
        ax.fill_between(phases, smooth_out_Nans(distr_KN[0][idx, :]), smooth_out_Nans(distr_KN[2][idx, :]), alpha=0.3, color='orange')
        ax.plot(phases, smooth_out_Nans(distr_KN[1][idx, :]), color='orange', label='KN only')

        # ax.fill_between(phases, smooth_out_Nans(distr_aft[0][idx, :]), smooth_out_Nans(distr_aft[2][idx, :]), alpha=0.1, color='g')
        # ax.plot(phases, smooth_out_Nans(distr_aft[1][idx, :]), color='g', label='aft only', linewidth=0.5)

        print(labels[idx], phases[idx5], flush=True)
        print(distr[1][idx, idx5] - distr_KN[1][idx, idx5], flush=True)

        # if idx == 2:
        #     print(distr_KN[1][idx, :], flush=True)

        ax.set_xlabel('time (days)')    
        ax.invert_yaxis()

        if log:
            ax.set_xscale('log')
        #ax.set_ylabel(r'$M$')
        ax.set_title(labels[idx])
        ax.legend()

    fig.tight_layout()
    if log:
        filename += 'log'

    fig.savefig(f'img/caps/{n}_events_{filename}_lc_ug.png')
    #fig.savefig(f'img/{n}_events_{filename}_lc_lsst_noaft.png')
    #fig.savefig(f'img/caps/lsst.png')
    plt.show()

def plot_color(n, save, filename):

    # load them in
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)
        # 10 events each with a Z, mag aftKN and KN

    # values is a 3d array each entry is an 2d grid of enhancements
    #distr = np.percentile(values[:,0], [16, 50, 84], axis=0) # get Z

    idx_uv2 = 0
    idx_r = 4
    # for all events get the total/KN Mag in uv and r and take the diff
    color_diff = np.squeeze(values[:, 1:, idx_uv2, :] - values[:, 1:, idx_r, :])

    # take all combined events and get the percentiles
        # repeat for KN only
    distr = np.percentile(color_diff[:, 0], [16, 50, 84], axis=0) # get magAftKN
    distr_KN = np.percentile(color_diff[:, 1], [16, 50, 84], axis=0) # get magKN

    fig, ax = plt.subplots(1,1)

    ax.fill_between(phases, distr[0,:], distr[2,:], alpha=0.3, color='b')
    ax.plot(phases, distr[1,:], color='b')
        # just KN
    ax.fill_between(phases, distr_KN[0,:], distr_KN[2,:], alpha=0.3, color='orange')
    ax.plot(phases, distr_KN[1,:], color='orange')

    ax.set_ylabel(r'$M_{uv2} - M_{r}$ [mag]')
    ax.set_xscale('log')
    ax.set_xlabel(r'phase [day]')
    fig.savefig(f'img/{n}_events_{filename}_uv-r.png')

    plt.show()

def plot_distance(n, save, filename, limiting_mags):

    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) # get magAftKN
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)

    def max_distance(M, limiting_mag):
        mu = limiting_mag - smooth_out_Nans(M)
        return 10**(1 + (mu/5)) / 1e6
    
    # lsst bands
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.ravel()

    for i, idx in enumerate(labels_idx):
        lim_mag = limiting_mags[idx]
        ax = axs[i]

        # plot distr 1 - median
        # fill btwn distr 0 and 2
            # aft + KN
        ax.fill_between(phases, max_distance(distr[0][idx, :], lim_mag), max_distance(distr[2][idx, :], lim_mag), alpha=0.3, color='b')
        ax.plot(phases, max_distance(distr[1][idx, :], lim_mag), color='b')
            # just KN
        ax.fill_between(phases, max_distance(distr_KN[2][idx, :], lim_mag), max_distance(distr_KN[0][idx, :], lim_mag), alpha=0.3, color='orange')
        ax.plot(phases, max_distance(distr_KN[1][idx, :], lim_mag), color='orange')

        ax.set_title(labels[idx] + ' Limiting Magnitude: '+str(lim_mag))

        ax.set_ylabel(r'distance [Mpc]')
        ax.set_yscale('log')
        ax.set_xlabel(r'phase [day]')
    
    fig.tight_layout()
    fig.savefig(f'img/{n}_events_{filename}_distlsst.png')
    plt.show() 

def merge(n, n_files, fname):

    # join the parameter arrays
    params_arr = []
    param_files = [f'data/sims/{n}_params_{fname}{i}.pkl' for i in range(1,11)]
    for f in param_files:
        with open(f, 'rb') as f:
                params = pickle.load(f)
                params_arr.append(params)
            
    params = np.vstack(params_arr)
    print(params.shape, flush=True)
    with open(f'data/sims/{n*n_files}_params_{fname}.pkl', 'wb') as f:
            pickle.dump(params, f)

    # join the value arrays
    values_arr = []
    val_files = [f'data/sims/{n}_events_{fname}{i}.pkl' for i in range(1,11)]
    for f in val_files:
        with open(f, 'rb') as f:
            values = pickle.load(f)
            values_arr.append(values)

    values = np.vstack(values_arr)
    print(values.shape, flush=True)
    with open(f'data/sims/{n*n_files}_events_{fname}.pkl', 'wb') as f:
            pickle.dump(values, f)

    # join the mass arrays
    mass_arr = []
    mass_files = [f'data/sims/{n}_masses_{fname}{i}.pkl' for i in range(1,11)]
    for f in mass_files:
        with open(f, 'rb') as f:
            params = pickle.load(f)
            mass_arr.append(params)
            
    masses = np.vstack(mass_arr)
    print(masses.shape, flush=True)
    with open(f'data/sims/{n*n_files}_masses_{fname}.pkl', 'wb') as f:
            pickle.dump(masses, f)

    # clean up
    for f in param_files+val_files+mass_files:
        os.remove(f)

def compare_GW170817():
    ang = 0.03 # core = 0.07
    at2017gfo = SEDDerviedLC(mej_dyn=10**-2.27, mej_wind=10**-1.28, phi=49.5, cos_theta=np.cos(ang), 
                             coord=SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec"), dist = 40*u.Mpc, av=0.0)
        # afterglow: Table 3 (default in afterglow addition)
    gw170817 = AfterglowAddition(at2017gfo, addKN=False)
    lcs = gw170817.getAbsMagsInPassbands(sncosmo_bands)

    theta_c = np.deg2rad(6) # 6 is ~ peak of RE
    mean_p = {"E0": 10**(49.3) * (1/(1-np.cos(theta_c))), 
              "thetaCore": theta_c, 
              "n0": 1e-2, 
              "p": 2.3, 
              "epsilon_e": 0.1, 
              "epsilon_B": 0.01}
    mean = AfterglowAddition(at2017gfo, **mean_p, addKN=False)
    mean_lcs = mean.getAbsMagsInPassbands(sncosmo_bands)
    mean.sed += mean.KNsed # then get combined curve
    comb_lcs = mean.getAbsMagsInPassbands(sncosmo_bands)

    fig, axs = plt.subplots(int(len(sncosmo_bands)/2)+1, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T

    for idx, band in enumerate(sncosmo_bands):

        ax=axs[idx]

        ax.plot(phases, lcs[band], color='k', linestyle='--', label='GW170817')
        ax.plot(phases, mean_lcs[band], color='g', linestyle='-', label='mean aft')
        ax.plot(phases, comb_lcs[band], color='b', linestyle='-', label='mean aft')
        ax.set_title(band)

        ax.set_ylabel(r'M')
        ax.invert_yaxis()
        ax.set_xlabel(r'phase [day]')

    fig.tight_layout()
    fig.savefig(f'img/mean_lc.png')
    plt.show()

def afterglows(n, save, filename='', log=False):

    # load them in
    params = get_params(n, save, filename)
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)

    # get just afterglows
    afterglows = values[:,0]

    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T
    #axs[-1].set_axis_off() # dont need the last one
    for i, idx in enumerate(labels_idx):
        ax = axs[i]

        for j in range(n):
            ax.plot(phases, smooth_out_Nans(afterglows[j][idx, :]), color='gray', 
                    alpha=0.1, linewidth=0.5)

            if afterglows[j][idx, 0] > 80:
                print(np.arccos(params[j][0]['cos_theta']), params[j][1], params[j][0], flush = True)


        # if idx == 2:
        #     print(distr_KN[1][idx, :], flush=True)

        ax.set_xlabel('time (days)')    
        ax.set_ylabel(r'$M$')
        ax.invert_yaxis()

        if log:
            ax.set_xscale('log')

        ax.set_title(labels[idx])

    fig.tight_layout()
    if log:
        filename += 'log'
    fig.savefig(f'img/{n}_events_{filename}_afts_lsst.png')
    plt.show()


if __name__ == '__main__':

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_events', default=500, type=int, required=False, help='number of events')
    parser.add_argument('--iter', type=int, required=False, help='Filename of simulation results')
    parser.add_argument('--plot', help='If true, plot else iter', action='store_true')

    args = parser.parse_args(args=argv)

    #dir = args.dir

    #np.random.seed(1674 % i) # each i will be different

    # https://www.lsst.org/scientists/keynumbers
        # u, g, r, i, z, y
    #limiting_mags = [23.8, 24.5, 24.03, 23.41, 22.74, 22.96]


    UV_bands = ['UVEX::FUV', 'UVEX::NUV']
    UV_labels = ['UVEX FUV', 'UVEX NUV']
    #labels_idx = np.arange(len(labels))

    sncosmo_bands = UV_bands + sncosmo_bands
    labels = UV_labels + labels
    labels_idx = np.arange(len(labels))

    # STAR-X: http://star-x.xraydeep.org/observatory/
    # UVEX: https://www.uvex.caltech.edu/page/about
    # UVOT: https://swift.gsfc.nasa.gov/about_swift/uvot_desc.html
    # LSST: https://www.lsst.org/scientists/keynumbers
    UV_limiting_mags = [24.5, 24.5]
    sncosmo_lim_mags = [22.3, 22.3, 23.8, 24.5, 24.03, 23.41, 22.74, 22.96, 26, 26, 26]
    UV_limiting_mags += sncosmo_lim_mags

    n = args.n_events
    n_files = 10
    fname = 'EK_aft' #EK_aft_0tc'
    if not args.plot:
        i = args.iter
        print(i, flush=True)
        np.random.seed(2667 % i)
        fname += str(i)
        gen_events(n, save=True, filename=fname)

        # TODO: re-run param gen for Ek_aft
            # if params are the same, 
            # then do again but just save the m1,m2
        #get_params(n, save=True, filename=fname)
        # done - now check that these are the correct ones, then save the masses

    if args.plot:
        #merge(n, n_files=n_files, fname=fname)
        #print('now plotting', flush=True)

        # select bands for plotting
        labels_idx = np.array([0, 1, 4, 5, 6, 7, 8, 9]) # UV + LSST
        labels_idx = np.array([4,5])
        font = {'family' : 'normal',
                 'size'   : 20}
        matplotlib.rc('font', **font)

        # #compare_GW170817()
        # #plot(n, save=False, filename=fname)
        # #afterglows(n*n_files, save=False, filename=fname)
        plot_avglc(n*n_files, save=False, filename=fname)
        #plot_color(n, save=False, filename=fname)
        #plot_distance(n*n_files, save=False, filename=fname, limiting_mags=UV_limiting_mags)
    # params = get_params(500, False, filename=fname)
    # values = gen_events(500, False, filename=fname)

    # # for idx in labels_idx:
    # #     band = idx
    # #     KN_v = values[:,2, band, :] # check all the  u light curves

    # #     print(values.shape, KN_v.shape, flush=True)
    # #     # for i in [12, 70, 87, 90, 99]:
    # #     #     print(params[i], flush=True)
    # #     #     print(values[i, 2, 2,:], flush=True)
    # #     print(np.argwhere(np.isnan(KN_v)),'\n',flush=True)

    # band = 1
    # KN_v = values[:,2, band, :]
    # #for event, phase in np.argwhere(np.isnan(KN_v))[0]:
    # print(np.argwhere(np.isnan(KN_v)))
    # event, phase = np.argwhere(np.isnan(KN_v))[7]
    
    
    # print(params[event])
    
    # kn_p, aft_p = params[event]

    # KN = SEDDerviedLC(**kn_p)
    # print(KN.getAbsMagsInPassbands([sncosmo_bands[band],]), flush=True)
    # #print(np.argwhere(np.isnan(KN.sed[phase])),'\n',flush=True)
    # idx_130nm = np.where(lmbd == 1300)[0][0]
    # idx_300nm = np.where(lmbd == 3100)[0][0]
    # source = sncosmo.TimeSeriesSource(phase=phases[phase-2:phase+3], wave=lmbd, flux = KN.sed[phase-2:phase+3], name='test', zero_before=True)
    # model = sncosmo.Model(source)

    # print(KN.sed[phase-2:phase+3, idx_130nm:idx_300nm], flush=True)
    # abs_mags = model.bandmag(band=sncosmo_bands[band], time = phases[phase-2:phase+3], magsys="ab")
    # print(abs_mags, flush=True)

    # model.add_effect(sncosmo.CCM89Dust(), 'host', 'rest')
    # model.set(hostebv = KN.host_ebv)
    # abs_mags = model.bandmag(band=sncosmo_bands[band], time = phases[phase-2:phase+3], magsys="ab")
    # print(abs_mags, flush=True)
    # # add MW extinction to observing frame
    # model.add_effect(sncosmo.F99Dust(), 'mw', 'obs')
    # model.set(mwebv=KN.mw_ebv)
    # abs_mags = model.bandmag(band=sncosmo_bands[band], time = phases[phase-2:phase+3], magsys="ab")
    # print(abs_mags, flush=True)


    # fig, axs = plt.subplots(2,1)
    # axs = axs.ravel()
    # #ax.plot(lmbd*u.AA.to(u.nm), KN.sed[phase])
    # axs[0].plot(phases, KN.getAbsMagsInPassbands([sncosmo_bands[band],])[sncosmo_bands[band]])
    # axs[0].plot(phases, KN.getAbsMagsInPassbands([sncosmo_bands[band],], apply_extinction=False)[sncosmo_bands[band]])
    # axs[0].axvline(phases[phase],ymin=min(KN.getAbsMagsInPassbands([sncosmo_bands[band],])[sncosmo_bands[band]]), ymax=max(KN.getAbsMagsInPassbands([sncosmo_bands[band],])[sncosmo_bands[band]]),
    #                linewidth=5, alpha=0.5, label=phases[phase])
    # axs[0].set_xlabel('day')
    # axs[0].invert_yaxis()
    # axs[0].legend()

    # for p in range(phase-2, phase+3):
    #     axs[1].plot(lmbd[:idx_300nm], KN.sed[p, :idx_300nm], label=phases[p])
    # axs[1].set_xlabel('AA')
    # axs[1].legend()

    # # there keeps being a weird big spike in the flux
    # print(lmbd[np.argmax(KN.sed[phase, :idx_300nm])])
    # fig.savefig(f'img/sed_check.png')
        #print(KN.getAbsMagsInPassbands([sncosmo_bands[band],]))

    # distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)
    # print(np.argwhere(np.isnan(distr_KN)), flush=True)








    

    









