import matplotlib as mpl
# mpl.use('PDF')
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
# plt.rcParams['text.usetex'] = True
# plt.style.use('./redback.mplstyle')

import seaborn as sns
from labellines import labelLine
import corner

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
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# for a given saved data, plot the apparent mag the event were at distance dist
    # also for the bands of interest give the limitiing mags
def plot_appmag(n, save, filename, plotname, dist, limiting_mags):

    fs = 20

    distmod = Distance(dist*u.Mpc).distmod.value

    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    distr = np.percentile(values[:,1], [16, 50, 84], axis=0) + distmod# get magAftKN
    distr_KN = np.percentile(values[:,2], [16, 50, 84], axis=0) + distmod

    # lsst bands
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(2*7.4, 2*3.25)) # figsize=(12, 5*n_plots)
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.ravel()

    for i, idx in enumerate(labels_idx):
        lim_mag = limiting_mags[idx]
        ax = axs[i]

        ax.fill_between(phases, smooth_out_Nans(distr[0][idx, :]), smooth_out_Nans(distr[2][idx, :]), alpha=0.3, color='b')
        ax.plot(phases, smooth_out_Nans(distr[1][idx, :]), color='b', label='Afterglow + KN')

        ax.fill_between(phases, smooth_out_Nans(distr_KN[0][idx, :]), smooth_out_Nans(distr_KN[2][idx, :]), alpha=0.3, color='orange')
        ax.plot(phases, smooth_out_Nans(distr_KN[1][idx, :]), color='orange', label='KN only')

        line = ax.axhline(y=lim_mag, color='gray', label=r'5-$\sigma$ Depth')

        if plotname.startswith('jwstroman'): # also .T axs
            if i == 3:
                line = ax.axhline(y=lim_mag, color='gray', label=r'10-$\sigma$ Depth')    
            else:
                line = ax.axhline(y=lim_mag, color='gray', label=r'5-$\sigma$ Depth')
            if i < 3:
                ax.set_ylabel(r'Apparent Magnitude', fontsize=fs)
                
            ax[2].set_xlabel(r'phase [day]', fontsize=fs)
            ax[5].set_xlabel(r'phase [day]', fontsize=fs)

            axs[3].legend()
            axs[3].scatter(1, 28.4, color='white')
    
        
        labelLine(line, x=7.5, label=lim_mag)

        if plotname == 'lsstToO':
            
            # find when the lc is = lim_mag
            interp_band = interp1d(phases, smooth_out_Nans(distr[1][idx, :]))
            def equ_to_solve(t):
                return interp_band(t) - lim_mag            

            e120 = [24.7, 25.8]
            e180 = [24.9, 26.0]
            
            sol1 = fsolve(equ_to_solve, 5)[0]

            lim_mag = e120[i]
            ax.axhline(y=lim_mag, color='gray', linestyle = '--',label=r'5-$\sigma$ 120s exp')
            sol2 = fsolve(equ_to_solve, 5)[0]

            lim_mag = e180[i]
            ax.axhline(y=lim_mag, color='gray', linestyle = 'dotted', label=r'5-$\sigma$ 180s exp')
            sol3 = fsolve(equ_to_solve, 5)[0]

            print(f"regular: {sol1}, 120s: {sol2}, 180s: {sol3}", flush=True)
            print(f"120 diff: {sol2-sol1}, 180 diff: {sol3-sol1}", flush=True)
            

        ax.set_title(labels[idx], fontsize=fs+3)
        ax.set_xlabel('phase [day]', fontsize=fs)
        ax.invert_yaxis()
        ax.set_xlim(0,12)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(fs)
        
    axs[0].legend(fontsize=fs)

    if plotname == 'poster_lsst':
        axs[0].set_ylim(38, 19)
        axs[1].set_ylim(32, 19)
    if plotname == 'poster_uvex':
        axs[0].set_ylim(42, 20)
        axs[1].set_ylim(42, 20)

    axs[0].set_ylabel(r'Apparent Magnitude ($d = 160 \ \rm Mpc$)', fontsize=fs)
    fig.tight_layout()
    plt.rcParams['font.size'] = str(fs)
    fig.savefig(f'img/caps/{n}_events_{filename}{plotname}_appmag_{dist}_all.png')
    plt.show() 

    
def plot_appmag_outlier(n, save, filename, plotname, dist, limiting_mags):

    distmod = Distance(dist*u.Mpc).distmod.value

    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    distr = np.percentile(values[:,1]+ distmod, [0, 16], axis=0) # get magAftKN
    distr_KN = np.percentile(values[:,2]+ distmod, [0, 16], axis=0) 

    #bright_mask = values[:,1] < distr[1]
    bright_idx = np.where(np.all(values[:,1]+distmod < distr[1], axis=1))[0]

    # get the individal lcs
    # lcs = values[bright_idx,1] + distmod
    # lcs_KN = values[bright_idx,2] + distmod
    
    # lsst bands
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 4*n_plots))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.ravel()

    for i, idx in enumerate(labels_idx):
        lim_mag = limiting_mags[idx]
        ax = axs[i]

        ax.fill_between(phases, smooth_out_Nans(distr[0][idx, :]), smooth_out_Nans(distr[1][idx, :]), alpha=0.3, color='b')
        #ax.plot(phases, smooth_out_Nans(distr[1][idx, :]), color='b', label='Afterglow + KN')

        ax.fill_between(phases, smooth_out_Nans(distr_KN[0][idx, :]), smooth_out_Nans(distr_KN[1][idx, :]), alpha=0.3, color='orange')
        #ax.plot(phases, smooth_out_Nans(distr_KN[1][idx, :]), color='orange', label='KN only')

        line = ax.axhline(y=lim_mag, color='gray', label='limiting mag')

        labelLine(line, x = 6, label=lim_mag)

        for b_idx in bright_idx[::10]:
            ax.plot(phases, smooth_out_Nans(values[b_idx, 2, idx, :]+distmod), color='gray', alpha=0.5)
            ax.plot(phases, smooth_out_Nans(values[b_idx, 1, idx, :]+distmod), color='black', alpha=0.5)

        ax.set_title(labels[idx])

        ax.set_ylabel(r'Apparent Magnitude')
        #ax.set_yscale('log')
        ax.set_xlabel(r'phase [day]')
        ax.invert_yaxis()
        #ax.set_xlim(0,8)
        #ax.set_ylim(30, 20)
    axs[0].legend()
    fig.tight_layout()
    
    fig.savefig(f'img/caps/{n}_events_{filename}{plotname}_appmag_{dist}_all_bright.png')
    plt.show()

    # corner 
    param_filename = filename
    with open(f'data/sims/{n}_params_{param_filename}.pkl', 'rb') as f:
        params = pickle.load(f)

    fig, axs = plt.subplots(5, 2, figsize=(16,16))
    axs = axs.ravel()
    plt.subplots_adjust(hspace=0.4)
    #idx_det_20 = np.where(np.array(data['discovery_window']) >= 19)[0]


    legends = ['all events', 'bright']
    for j, parms in enumerate([params.T, params[bright_idx].T]):
        kn_p, aft_p = parms # [ [kn, aft], [kn, aft]] -> [[kn, kn], [aft, aft]]

        kn_params = {"mej_dyn": [], 
                    "mej_wind": [], 
                    "phi": [], 
                    "cos_theta": [], 
                    "dist": [], 
                    #  "coord": [], 
                    "av": [], 
                    # "rv": []
                    }

        for d in kn_p:
            for key, value in d.items():
                if key in kn_params.keys():
                    kn_params[key].append(value)

        #kn_params['mej_dyn'] = np.log10(kn_params['mej_dyn']) 
        #kn_params['mej_wind'] = np.log10(kn_params['mej_wind']) 
        
        aft_params = {"E0": [], 
                    "thetaCore": [], 
                    "n0": [], 
                    "p": [], 
                    #  "epsilon_e": [],
                    #  "epsilon_B": []
                    }

        for d in aft_p:
            for key, value in d.items():
                if key in aft_params.keys():
                    aft_params[key].append(value)

        
        for i, (param, value) in enumerate(kn_params.items()):
            axs[i].hist(value, bins=100, density=True, alpha=0.6, label=legends[j])
            axs[i].set_title(param)
        axs[0].legend()
        for i, (param, value) in enumerate(aft_params.items()):
            
            if param == 'E0' or param == 'n0':
                value = np.log10(value)
            
            axs[i+6].hist(value, bins=100, density=True, alpha=0.6)
            axs[i+6].set_title(param)

        
        
    fig.savefig(f'img/caps/{n}_events_{filename}{plotname}_detParam.png')
    plt.show()

    # corners
    corner_dicts = []
    for parms in [params.T, params[bright_idx].T]:

        params_dict = {"mej_dyn": [], 
                    "mej_wind": [], 
                    "phi": [], 
                    "cos_theta": [], 
                    #"dist": [], 
                    #  "coord": [], 
                    "av": [], 
                    # "rv": [],
                    "E0": [], 
                    "thetaCore": [], 
                    "n0": [], 
                    "p": [], 
                    #  "epsilon_e": [],
                    #  "epsilon_B": []
                    }

        kn_p, aft_p = parms
        for d in kn_p:
            for key, value in d.items():
                if key in params_dict.keys():
                    params_dict[key].append(value)
        for d in aft_p:
            for key, value in d.items():
                if key in params_dict.keys():
                    print(params_dict[key])
                    params_dict[key].append(value)

        params_dict['E0'] = np.log10(params_dict['E0'])
        params_dict['n0'] = np.log10(params_dict['n0'])
        params_dict['thetav/c'] = np.arccos(params_dict['cos_theta']) / params_dict['thetaCore']

        data_df = pd.DataFrame(params_dict)
        print(data_df, flush=True)
        corner_dicts.append(data_df)  

    cols = ['E0', 'n0', 'thetaCore', 'thetav/c']
    fig2 = corner.corner(corner_dicts[0][cols], plot_countours=True, show_titles=True, smooth=2, )
    corner.corner(corner_dicts[1][cols], plot_countours=True, show_titles=True, smooth=2, color='C1', fig=fig2)
    fig2.savefig(f'img/caps/{n}_events_{filename}{plotname}_subset.png')
    

def plot_stratify(n, save, filename, plotname=''):

    # two possible version 
        # events from entire sample or events that live in the middle area

    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    params = get_params(n, save, filename)

    events = np.arange(5, n, 50)# every 50th event
    values = values[events]
    params = params[events]

    fig, axs = plt.subplots(4, 2, figsize=(12, 16))
    axs = axs.ravel()
    labels = ['E0', 'thetaCore', 'n0', 'cos_theta', 'p']

    param_dict = pd.DataFrame([{**d2, **d1} for d1, d2 in params])[labels].to_dict(orient='list')
    p_dict2 = {'logE0': np.log10(param_dict['E0']),
               'thetaCore': np.array(param_dict['thetaCore']),
               'logn0': np.log10(param_dict['n0']),
               'thetaView': np.arccos(param_dict['cos_theta']),
               }
    p_dict2['tV/tC'] = p_dict2['thetaView'] / p_dict2['thetaCore']
    p_dict2['E0/n0'] = np.log10(np.array(param_dict['E0']) / np.array(param_dict['n0']))
    p_dict2['tbin'] = 2.95*(((10**p_dict2['E0/n0'])/1e53)**(1/3))*(p_dict2['thetaCore']/0.1)**(8/3)
    
    p = np.array(param_dict['p'])
    g = 0.25*(p_dict2['thetaView'] / p_dict2['thetaCore'])
    ee_bar = 0.1*((p-2)/(p-1))
    nu_gband = ((5000*u.AA).to(u.Hz, equivalencies=u.spectral()).value)/1e14 # in 1e14 Hz
    t = 5.1 # days

    p_dict2['E0/n0 - G'] = -np.log10( 0.461*(p-0.04)*np.exp(2.53*p)
        * (ee_bar**(p-1))
        * (0.01**((1+p/4)))
        * ((np.array(param_dict['E0'])/1e52)**((3+p)/4)) 
        * (np.array(param_dict['n0'])**(1/2)) 
        * (t**(3-(6*p)-(3*g))/(8+g)) * (nu_gband**((1-p)/2)) ) # fixed at 5 days
    # p_dict2['E0/n0 - H'] = -np.log10( 0.855*(p-0.98)*np.exp(1.95*p)
    #     * (ee_bar**(p-1))
    #     * (0.01**((p-2/4)))
    #     * ((np.array(param_dict['E0'])/1e52)**((2+p)/4))
    #     * (t**(1-(6*p)-(2*g))/(8+g)) * (nu_gband**(-p/2))) # fixed at 5 days

    # p_dict2['E0/n0 - G'] = -np.log10(np.array(param_dict['E0'])**((3+p)/4) 
    #     * np.array(param_dict['n0'])**(1/2)) * ((3-(6*p)-(3*g))/(8+g))

    for i, name in enumerate(p_dict2.keys()):
        ax = axs[i]

        # set up colorbar
        cmap = cm['jet']
        param = p_dict2[name]

        if name == 'tbin':
            ax.hist(p_dict2[name], bins=100)
            ax.set_xlim(0, 25)
            ax.set_title('distr of break times')
            continue
        
        if name == 'tV/tC':
            norm = colors.Normalize(vmin=min(param), vmax=10)
        else:
            norm = colors.Normalize(vmin=min(param), vmax=max(param))
        print(min(param), max(param), flush=True)
        
        for j, _ in enumerate(values):
            distmod = 0
            band = 5 # g band

            # get g-band afterglow only light curve
            val = p_dict2[name][j]
            ax.plot(phases, values[j, 0, band, :]+distmod, c=cmap(norm(val)))
            

        ax.set_title(name)
        ax.set_ylabel(r'Absolute Magnitude')
        ax.set_xlabel(r'phase [day]')
        ax.invert_yaxis()
        #ax.set_xlim(0,8)
        #ax.set_ylim(30, 20)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cm['jet']), ax=ax, orientation='vertical', pad=0.05)

    fig.tight_layout()
    fig.savefig(f'img/caps/{n}_events_{filename}{plotname}_stratifyaft_Ggband.png')
    plt.show()
    
    # loop thru events, select a sample (every 50/100)

        # per event, add it to a plot colored by param value
    
    return None

def plot_mag_scatter(n, save, filename, plotname=''):
    values = gen_events(n, save, filename) # shape n events, 3 LCs, 11, 50 (each row is an LC)
    params = get_params(n, save, filename)

    # events = np.arange(5, n, 50)# every 50th event
    values = values
    params = params

    # fig, axs = plt.subplots(4, 2, figsize=(12, 16))
    # axs = axs.ravel()
    ps_needed = ['E0', 'thetaCore', 'n0', 'cos_theta', 'p']

    param_dict = pd.DataFrame([{**d2, **d1} for d1, d2 in params])[ps_needed].to_dict(orient='list')
    p_dict2 = {'logE0': np.log10(param_dict['E0']),
               'thetaCore': np.array(param_dict['thetaCore']),
               'logn0': np.log10(param_dict['n0']),
               'thetaView': np.arccos(param_dict['cos_theta']),
               }
    p_dict2['tV/tC'] = p_dict2['thetaView'] / p_dict2['thetaCore']
    p_dict2['E0/n0'] = np.log10(np.array(param_dict['E0']) / np.array(param_dict['n0']))
    p_dict2['tbin'] = 2.95*(((10**p_dict2['E0/n0'])/1e53)**(1/3))*(p_dict2['thetaCore']/0.1)**(8/3)
    
    p = np.array(param_dict['p'])
    g = 0.25*(p_dict2['thetaView'] / p_dict2['thetaCore'])
    ee_bar = 0.1*((p-2)/(p-1))
    nu_gband = ((5000*u.AA).to(u.Hz, equivalencies=u.spectral()).value)/1e14 # in 1e14 Hz
    t = 5.1 # days
    p_dict2['G'] = -np.log10(0.461*(p-0.04)*np.exp(2.53*p)
        * (ee_bar**(p-1))
        * (0.01**((1+p/4)))
        * ((np.array(param_dict['E0'])/1e52)**((3+p)/4)) 
        * (np.array(param_dict['n0'])**(1/2)) 
        * (t**(3-(6*p)-(3*g))/(8+g)) * (nu_gband**((1-p)/2))) # fixed at 5 days
    p_dict2['H'] = -np.log10( 0.855*(p-0.98)*np.exp(1.95*p)
        * (ee_bar**(p-1))
        * (0.01**((p-2/4)))
        * ((np.array(param_dict['E0'])/1e52)**((2+p)/4))
        * (t**(1-(6*p)-(2*g))/(8+g)) * (nu_gband**(-p/2)))
    p_dict2['D'] = -np.log10(27.9 * ((p-1)/((3*p)-1))
        * (ee_bar**(-2/3))
        * (0.01**((1/3)))
        * (np.array(param_dict['n0'])**(1/2))
        * ((np.array(param_dict['E0'])/1e52)**(5/6))
        * (t**(1+(3*g))/(8+g)) * (nu_gband**(1/3)))
    p_dict2['E'] = -np.log10(73.0
        * (0.01)
        * (np.array(param_dict['n0'])**(5/6))
        * ((np.array(param_dict['E0'])/1e52)**(7/6))
        * (t**((-5/3)+(11*g/3))/(8+g)) * (nu_gband**(1/3)))
    p_dict2['F'] = -np.log10(6.87
        * (0.01**((-1/4)))
        * ((np.array(param_dict['E0'])/1e52)**(3/4))
        * (t**(-5+(2*g))/(8+g)) * (nu_gband**(-1/2)))
    

    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    axs = axs.ravel()
    # values = gen_events(n, save, filename)
    for i, metric in enumerate(['G', 'H', 'D', 'E', 'F']):
        ax = axs[i]
        
        distmod = 0
        band = 5 # g band
        idx_5day = np.where(np.isclose(phases, t))[0][0]

        # get g-band afterglow only magnitude at 5d
        m = p_dict2[metric]
        ax.scatter(m, values[:, 0, band, idx_5day]+distmod, color=f'C{i}', alpha=0.4)
        ax.set_title(metric + '(scaled)')
        ax.set_ylabel('Abs Mag')
        ax.set_xlabel('metric')
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(f'img/caps/{n}_events_{filename}{plotname}_stratifyaft_5dayPLSmin.png')
    plt.suptitle('Abs g-band Magnitude at 5 days')
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


def plot_aft_varied(i):

    params = ['E0', 'n0', 'thetaCore', 'theta_v']

    # set up afterglow
    aft_params = {'KN': None,  
                'E0': 10**52.96, 
                'thetaCore': 0.066,
                'n0': 10**-2.7, 
                'p': 2.17,
                'epsilon_e': 10**-1.4,
                'epsilon_B': 10**-4, 
                'theta_v':  0.0, 
                'coord':  SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec"),
                'dist': 40*u.Mpc}
    
    param_name = params[i]

    cmap = cm['jet']

    if param_name =='E0':
        param = np.linspace(50, 56, 7)
        vals = 10**param
    if param_name == 'n0':
        param = np.linspace(-5, 5, 11)
        vals = 10**param
    if param_name == 'thetaCore':
        param = np.arange(0.3, 13.3)
        vals = (param*u.deg).to(u.rad).value
    if param_name == 'theta_v':
        param = np.arange(0,11)
        vals = param*aft_params['thetaCore']

    cmap = cm['jet']
    norm = colors.Normalize(vmin=min(param), vmax=max(param))
    cs = [cmap(norm(p)) for p in param]
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    for j, val in tqdm(enumerate(vals)):
        aft_params[param_name] = val 
        print(param_name, j, flush=True)
        afterglow = AfterglowAddition(**aft_params) # use typical values
        
        #t = np.arccos(ct)*u.rad.to(u.deg)
        ax.plot(phases, afterglow.getAbsMagsInPassbands(lsst_bands, apply_extinction=False)['lsstg'], color=cs[j])
        
    ax.set_title(param_name)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cm['jet']), ax=ax, orientation='vertical', pad=0.05)
    
    ax.set_xlabel("time (days)")
    ax.set_ylabel("M")
    ax.invert_yaxis()
    plt.savefig(f"img/caps/aft_{param_name}_varied.png")
    plt.show()

# stolen from paper_figs (Shah et al.)
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
    limiting_mags = [24.5, 24.5]
    # #labels_idx = np.arange(len(labels))

    sncosmo_bands = UV_bands + sncosmo_bands
    labels = UV_labels + labels
    # labels_idx = np.arange(len(labels))

    # # STAR-X: http://star-x.xraydeep.org/observatory/
    # # UVEX: https://www.uvex.caltech.edu/page/about
    # # UVOT: https://swift.gsfc.nasa.gov/about_swift/uvot_desc.html
    # # LSST: Bianco+ 2022
        #https://www.lsst.org/scientists/keynumbers : 23.8, 24.5, 24.03, 23.41, 22.74, 22.96
    sncosmo_lim_mags = [22.3, 22.3, 23.9, 25.0, 24.7, 24.0, 23.3, 22.1] # , 26, 26, 26
    limiting_mags += sncosmo_lim_mags

    sncosmo_bands += ['f070w', 'f277w', 'f444w', 'f062', 'f146', 'f213']
    labels += ['JWST 70w', 'JWST 200w', 'JWST 444w', 'Roman 62', 'Roman 146wide', 'Roman 213']
    limiting_mags += [28.5, 28.7, 28.3, 24.77, 25.37, 23.14]
    labels_idx = np.arange(len(labels))

    n = args.n_events
    n_files = 10
    fname = 'All' #'EK_nir' #'EK_red' #
    plotname='poster'
    # if not args.plot:
    #     i = args.iter
    #     print(i, flush=True)
    #     np.random.seed(1647 % i)
    #     fname += str(i)
    #     gen_events(n, save=True, filename=fname)

    # plot_aft_varied(args.iter)

    if args.plot:
        #merge(n, n_files=n_files, fname=fname)
        print('now plotting', flush=True)

        # select bands for plotting
        #labels_idx = np.array([0, 1, 4, 5, 6, 7, 8, 9]) # UV + LSST
        # labels_idx = np.array([4,5])
        # labels_idx = np.arange(len(labels))
        # font = { 'size'   : 15}
        # mpl.rc('font', **font)

        # plt_params = {'n': n*n_files, 'save':False, 'filename': fname, 'plotname':plotname}
        # plot_stratify(**plt_params)
        # plot_mag_scatter(**plt_params)

        #compare_GW170817()
        labels_idx = np.array([4,5])
        plot_appmag(n*n_files, save=False, filename=fname, plotname=plotname+'_lsst', 
                            dist=160, limiting_mags=limiting_mags)
        # # plot_appmag_outlier(n*n_files, save=False, filename=fname, plotname=plotname, 
        # #                     dist=160, limiting_mags=limiting_mags) # use the data gen'd in the previous plotting
        # #plot_openingAngle(n*n_files, save=False, filename=fname)
        # #makeTrialsEjectaHistogram()

        # labels_idx = np.array([0,1])
        # plot_appmag(n*n_files, save=False, filename=fname, plotname=plotname+'_uvex', 
        #                     dist=160, limiting_mags=limiting_mags)
        
        # labels_idx = np.array([len(labels)-1-i for i in range(6)])# last 6 is roman/jwst
        # plot_appmag(n*n_files, save=False, filename=fname, plotname='_clean_jwstromanBianco', 
        #                     dist=160, limiting_mags=limiting_mags)
        
