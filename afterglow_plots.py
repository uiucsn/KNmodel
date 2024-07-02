import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from labellines import labelLine

import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Distance

import numpy as np
import pandas as pd
import afterglowpy as grb
import sncosmo
from tqdm import tqdm
import pickle
import sys
import argparse

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands, mej_dyn_grid_high, mej_dyn_grid_low, mej_wind_grid_high, mej_wind_grid_low


from afterglow_addition import AfterglowAddition
from afterglow_distribution import sncosmo_bands, labels, labels_idx, gen_events, get_params, smooth_out_Nans

# for a given saved data, plot the apparent mag the event were at distance dist
    # also for the bands of interest give the limitiing mags
def plot_appmag(n, save, filename, dist, limiting_mags):

    distmod = Distance(dist*u.Mpc).distmod.value

    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) + distmod# get magAftKN
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0) + distmod

    # lsst bands
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(17, 8))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.ravel()

    for i, idx in enumerate(labels_idx):
        lim_mag = limiting_mags[idx]
        ax = axs[i]

        ax.fill_between(phases, smooth_out_Nans(distr[0][idx, :]), smooth_out_Nans(distr[2][idx, :]), alpha=0.3, color='b')
        ax.plot(phases, smooth_out_Nans(distr[1][idx, :]), color='b', label='Afterglow + KN')

        ax.fill_between(phases, smooth_out_Nans(distr_KN[0][idx, :]), smooth_out_Nans(distr_KN[2][idx, :]), alpha=0.3, color='orange')
        ax.plot(phases, smooth_out_Nans(distr_KN[1][idx, :]), color='orange', label='KN only')

        line = ax.axhline(y=lim_mag, color='gray', label='limiting mag')

        labelLine(line, x = 17, label=lim_mag)

        ax.set_title(labels[idx])

        ax.set_ylabel(r'Apparent Magnitude')
        #ax.set_yscale('log')
        ax.set_xlabel(r'phase [day]')
        ax.invert_yaxis()
        ax.set_xlim(0,20)
    axs[0].legend()
    fig.tight_layout()
    
    fig.savefig(f'img/caps/{n}_events_{filename}_appmag_{dist}_ug_desc.png')
    plt.show() 

def plot_openingAngle(n, save, filename):
    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    params = get_params(n, save, filename)
    openingAngles = np.rad2deg([d['thetaCore'] for d in params.T[1]])

    cmap = cm['jet']
    norm = colors.Normalize(vmin=min(openingAngles), vmax=max(openingAngles))

    total_lcs = values[:,1]

    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) 
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0)

    # lsst bands
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 20))
    plt.subplots_adjust(hspace=0.6)
    axs = axs.ravel()

    for i, idx in enumerate(labels_idx):
        ax = axs[i]
        for f in range(n): # plot 1/10th so the lines
            if f % 10 == 0:
                ang = np.rad2deg(params[f][1]['thetaCore'])
                ax.plot(phases, smooth_out_Nans(total_lcs[f][idx, :]), 
                        c=cmap(ang/max(openingAngles)), linewidth=0.5, alpha=0.1)

        #ax.fill_between(phases, smooth_out_Nans(distr[0][idx, :]), smooth_out_Nans(distr[2][idx, :]), alpha=0.3, color='b')
        ax.plot(phases, smooth_out_Nans(distr[1][idx, :]), color='b', label='aft+KN')

        #ax.fill_between(phases, smooth_out_Nans(distr_KN[0][idx, :]), smooth_out_Nans(distr_KN[2][idx, :]), alpha=0.3, color='orange')
        ax.plot(phases, smooth_out_Nans(distr_KN[1][idx, :]), color='orange', label='KN only')

        ax.set_title(labels[idx])

        ax.set_ylabel(r'App Mag')
        #ax.set_yscale('log')
        ax.set_xlabel(r'phase [day]')
        ax.invert_yaxis()
    
    fig.colorbar(ScalarMappable(norm=norm, cmap=cm['jet']), ax=axs, orientation='horizontal', pad=0.05)
    #fig.tight_layout()
    fig.suptitle("Colored by opening angle", y = 0.93, fontsize='xx-large')
    fig.savefig(f'img/{n}_events_{filename}_opening.png')
    plt.show() 

# stolen from paper_figs
def makeTrialsEjectaHistogram():

    # 170817 params
    GW170817_mej_wind = 10**-1.28 
    GW170817_mej_wind_errors = [[10**-1.28 - 10**-1.63],[10**-0.86 - 10**-1.28]]
    GW170817_mej_dyn = 10**-2.27
    GW170817_mej_dyn_errors = [[10**-2.27 - 10**-2.81],[10**-1.26 - 10**-2.27]]

    # TODO: add params from other KN
        # 130506
        # 190445
        # 211211
    grb211211_mej_wind = 0.025
    grb211211_mej_dyn = 0.015
        # 230307 
    grb230307_mej_wind = 0.05
    grb230307_mej_dyn = 0.005

    df_Galaudage = pd.read_csv('O4-DECam-r-23mag-Galaudage/trials_df.csv')
    df_Farrow = pd.read_csv('O4-DECam-r-23mag-Farrow/trials_df.csv')

    mej_wind = df_Farrow['mej_wind']
    mej_dyn = df_Farrow['mej_dyn']

    sns.kdeplot(x=mej_wind, y=mej_dyn, levels=[0.2, 0.5, 0.8])

    mej_wind = df_Galaudage['mej_wind']
    mej_dyn = df_Galaudage['mej_dyn']

    sns.kdeplot(x=mej_wind, y=mej_dyn, levels=[0.2, 0.5, 0.8])

    plt.xlabel(r'$\mathrm{m_{ej}^{wind}} (M_{\odot})$', fontsize='x-large')
    plt.ylabel(r'$\mathrm{m_{ej}^{dyn}} (M_{\odot})$', fontsize='x-large')

    colors = ['C0', 'C1']
    lines = [Line2D([0], [0], color=colors[0]), Line2D([0], [0], color=colors[1])]
    labels = ['Farrow et al.', 'Galaudage et al.']
    plt.legend(lines, labels, loc='upper left', prop={'size': 13})

    plt.vlines(mej_wind_grid_low, ymin=mej_dyn_grid_low, ymax=mej_dyn_grid_high, color='black', linestyle='dotted')
    plt.vlines(mej_wind_grid_high,  ymin=mej_dyn_grid_low, ymax=mej_dyn_grid_high, color='black', linestyle='dotted')
    plt.hlines(mej_dyn_grid_low, xmin = mej_wind_grid_low, xmax= mej_wind_grid_high, color='black', linestyle='dotted')
    plt.hlines(mej_dyn_grid_high, xmin = mej_wind_grid_low, xmax= mej_wind_grid_high, color='black', linestyle='dotted')
    
    plt.errorbar(GW170817_mej_wind, GW170817_mej_dyn, xerr= GW170817_mej_wind_errors, yerr=GW170817_mej_dyn_errors, marker='*', c='black',ecolor='black', markersize= 15)
    plt.scatter(grb230307_mej_wind, grb230307_mej_dyn, marker='*', c='r', s= 50)
    plt.scatter(grb211211_mej_wind, grb211211_mej_dyn, marker='*', c='r', s= 50, alpha=0.5)
    plt.text(0.06 , 0.006 ,'SSS17a', c='black')

    plt.loglog()
    
    plt.savefig('img/mej_scatter_hist.pdf')


    plt.show()

def plot_3D_SED():


    COLOR = 'white'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR

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
    # if not args.plot:
    #     i = args.iter
    #     print(i, flush=True)
    #     np.random.seed(1647 % i)
    #     fname += str(i)
    #     gen_events(n, save=True, filename=fname)

    if args.plot:
        #merge(n, n_files=n_files, fname=fname)
        print('now plotting', flush=True)

        # select bands for plotting
        labels_idx = np.array([0, 1, 4, 5, 6, 7, 8, 9]) # UV + LSST
        labels_idx = np.array([4,5])
        font = {'family' : 'normal',
                 'size'   : 20}
        mpl.rc('font', **font)

        #compare_GW170817()
        plot_appmag(n*n_files, save=False, filename=fname, dist=160, limiting_mags=UV_limiting_mags) # use the data gen'd in the previous plotting
        #plot_openingAngle(n*n_files, save=False, filename=fname)
        #makeTrialsEjectaHistogram()

