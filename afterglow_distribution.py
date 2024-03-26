import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands
from afterglow_addition import AfterglowAddition
from dns_mass_distribution import Galaudage21, Farrow19
from monte_carlo_sims import get_ejecta_mass
from afterglow_params import get_logn0, get_opening_angle, get_loge0, get_p

np.random.seed(1674)

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
        mej_dyns, mej_winds = get_ejecta_mass(mass1, mass2)
        
        # simulate coordinates. Additional term ensures minimum distance of 0.05 Mpc
        box_size = 600 # increase?
        x = np.random.uniform(-box_size/2., box_size/2., n)*u.Mpc
        y = np.random.uniform(-box_size/2., box_size/2., n)*u.Mpc
        z = np.random.uniform(-box_size/2., box_size/2., n)*u.Mpc
        dists = (x**2. + y**2. + z**2.)**0.5 + (0.05*u.Mpc)
        
    
        coords = np.empty(n, dtype=object)
        dists = np.empty(n, dtype=object)
        for i in range(n):
            r, dec, ra = coord.cartesian_to_spherical(x[i], y[i], z[i])
            coords[i] = coord.SkyCoord(ra=ra, dec=dec)
            dists[i] = r
        
        # loc = 16
        # scale = 10
        # a, b = (np.rad2deg(0.01) - loc) / scale , (90 - loc) / scale
        # thetaCores = sts.truncnorm.rvs(a, b, loc=loc, scale=scale, size=n) # deg, Fong et al 2015
        thetaCores = np.deg2rad(get_opening_angle(n, distr='RE23'))
        cos_thetas = np.random.uniform(np.cos(2*thetaCores), 1, size=n)
        # cos_thetas = np.random.uniform(0, thetaCores, size=n)
        # thetaCores = np.deg2rad(thetaCores)
        # cos_thetas = np.cos(np.deg2rad(cos_thetas))
        #cos_thetas = np.full((n,), 1) # on-axis for all 
        phis = np.random.uniform(15, 75, size=n)
        avs = np.random.exponential(0.334, size=n)*0.334
        rvs = np.full((n,), 3.1)

        param_names = ["mej_dyn", "mej_wind", "phi", "cos_theta", "dist", "coord", "av", "rv"]
        params_array = np.array([mej_dyns, mej_winds, phis, cos_thetas, dists, coords, avs, rvs]).T
        kn_params = [{param_names[i]: value for i, value in enumerate(row)} for row in params_array]

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

        # cannot have cores below like 0.01 rad and 2 pi rad
        # loc = 16
        # scale = 10
        # a, b = (np.rad2deg(0.01) - loc) / scale , (90 - loc) / scale
        # thetaCores = np.deg2rad(sts.truncnorm.rvs(a, b, loc=loc, scale=scale, size=n)) # deg, Fong et al 2015
        #logn0s = np.random.normal(-2, 0.4**2, n) # cm^-2
        #ps = np.random.normal(2.25, 0.1**2, n)   # spectral index
        # logees = np.random.normal(-1, 0.3**2, n)
        # logebs = np.random.normal(-2, 0.4**2, n)

        logE0s = get_loge0(n, distr='Zhu22')
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

        params = list(np.array([kn_params, aft_params]).T)
        #print(params, flush=True)

        # save them
        with open(f'data/sims/{n}_params_{filename}.pkl', 'wb') as f:
            pickle.dump(params, f)
        #np.savetxt(f"{n}_events.csv", params, delimiter=",")
    else:
        # load in the values
        with open(f'data/sims/{n}_params_{filename}.pkl', 'rb') as f:
            params = pickle.load(f)

    # list of dicts
    return params

def gen_event(params):

    kn_params, aft_params = params

    KN = SEDDerviedLC(**kn_params)
    afterglow = AfterglowAddition(KN, **aft_params, addKN=True)

    # get the diff and save those
    # abs mag of KN+afterglow
    mag_band_aftKN = afterglow.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_aftKN = np.array([list(item) for item in mag_band_aftKN.values()])
    
    # abs mag of KN
    mag_band_KN = KN.getAbsMagsInPassbands(sncosmo_bands)
    mag_band_KN = np.array([list(item) for item in mag_band_KN.values()])

    # TODO smooth any holes

    # since each band is a row in the mag array, the y values are the bands (param help const thru row = y val)
    Z = mag_band_aftKN - mag_band_KN # magnitude enhancement, (11, 50)
    return np.array([Z, mag_band_aftKN, mag_band_KN]) # ()


# issue with NaNs in UV: check and interpolate over them
def smooth_out_Nans(lc):    
    if np.isnan(lc).any():
        not_nan_indices = np.logical_not(np.isnan(lc))
        interpolator = interp1d(phases[not_nan_indices], lc[not_nan_indices], kind='linear', fill_value="extrapolate")
        interpolated_data = interpolator(phases)
        return interpolated_data
    else:
        return lc


def gen_events(n, save=True, filename=''):

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

def plot_avglc(n, save, filename=''):

    # load them in
    values = gen_events(n, save, filename) # shape 11, 50 (each row is an LC)
        # 10 events each with a Z, mag aftKN and KN

    # values is a 3d array each entry is an 2d grid of enhancements
    #distr = np.percentile(values[:,0], [16, 50, 84], axis=0) # get Z
    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) # get magAftKN
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)
    #mag_tot = -2.5*np.log10(10**(-0.4*mag['ztfi']) + 10**(-0.4*mag_grb['i']))
        # 
    #distr_justAft = -2.5*(np.log10(10**(-0.4*distr) - 10**(-0.4*distr_KN)))

    fig, axs = plt.subplots(int(len(sncosmo_bands)/2), 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T
    #axs[-1].set_axis_off() # dont need the last one
    for idx in labels_idx:
        ax = axs[idx]
        # plot distr 1 - median
        # fill btwn distr 0 and 2
            # aft + KN
        ax.fill_between(phases, smooth_out_Nans(distr[0][idx, :]), smooth_out_Nans(distr[2][idx, :]), alpha=0.3, color='b')
        ax.plot(phases, smooth_out_Nans(distr[1][idx, :]), color='b', label='Afterglow + KN')
            # just KN
        ax.fill_between(phases, smooth_out_Nans(distr_KN[0][idx, :]), smooth_out_Nans(distr_KN[2][idx, :]), alpha=0.3, color='orange')
        ax.plot(phases, smooth_out_Nans(distr_KN[1][idx, :]), color='orange', label='KN only')

        if idx == 2:
            print(distr_KN[1][idx, :], flush=True)

        ax.set_xlabel('time (days)')    
        ax.set_ylabel(r'$M$')
        ax.invert_yaxis()
        ax.set_title(labels[idx])
        ax.legend()

    fig.tight_layout()
    fig.savefig(f'img/{n}_events_{filename}_lc_interp.png')
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

    # bands: 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty'
    idx_b = 0

    def max_distance(M, limiting_mag):
        mu = limiting_mag - smooth_out_Nans(M)
        return 10**(1 + (mu/5)) / 1e6
    
    # lsst bands
    fig, axs = plt.subplots(int(len(limiting_mags)/2), 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.ravel()

    for idx, lim_mag in enumerate(limiting_mags):
        ax = axs[idx]

        # plot distr 1 - median
        # fill btwn distr 0 and 2
            # aft + KN
        band = idx + idx_b
        ax.fill_between(phases, max_distance(distr[0][band, :], lim_mag), max_distance(distr[2][band, :], lim_mag), alpha=0.3, color='b')
        ax.plot(phases, max_distance(distr[1][band, :], lim_mag), color='b')
            # just KN
        ax.fill_between(phases, max_distance(distr_KN[2][band, :], lim_mag), max_distance(distr_KN[0][band, :], lim_mag), alpha=0.3, color='orange')
        ax.plot(phases, max_distance(distr_KN[1][band, :], lim_mag), color='orange')

        ax.set_title(labels[band] + ' Limiting Magnitude: '+str(lim_mag))

        ax.set_ylabel(r'distance [Mpc]')
        ax.set_yscale('log')
        ax.set_xlabel(r'phase [day]')
    
    fig.tight_layout()
    fig.savefig(f'img/{n}_events_{filename}_dist.png')
    plt.show() 




if __name__ == '__main__':

    # https://www.lsst.org/scientists/keynumbers
        # u, g, r, i, z, y
    #limiting_mags = [23.8, 24.5, 24.03, 23.41, 22.74, 22.96]

    fname = 're23_offa'
    UV_bands = ['UVEX::FUV', 'UVEX::NUV']
    UV_labels = ['UVEX FUV', 'UVEX NUV']
    #labels_idx = np.arange(len(labels))

    sncosmo_bands = UV_bands + sncosmo_bands[:4]
    labels = UV_labels + labels[:4]
    labels_idx = np.arange(len(labels))

    # STAR-X: http://star-x.xraydeep.org/observatory/
    # UVEX: https://www.uvex.caltech.edu/page/about
    # UVOT: https://swift.gsfc.nasa.gov/about_swift/uvot_desc.html
    # LSST: https://www.lsst.org/scientists/keynumbers
    UV_limiting_mags = [24.5, 24.5, 22.3, 22.3, 23.8, 24.5, 24.03, 23.41, 22.74, 22.96, 26, 26, 26]

    n = 500  
    plot(n, save=True, filename=fname)
    plot_avglc(n, save=False, filename=fname) # use the data gen'd in the previous plotting
    #plot_color(n, save=False, filename=fname)
    plot_distance(n, save=False, filename=fname, limiting_mags=UV_limiting_mags[:6])
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








    

    









