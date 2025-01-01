import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Distance
import afterglowpy as grb
import sncosmo
from tqdm import tqdm
import pickle
import corner
import os

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import phases
from sed_to_lc import SEDDerviedLC, lsst_bands
from afterglow_distribution import gen_events, get_params, sncosmo_bands, labels, labels_idx
from waveforms import get_snr

import healpy as hp
import scipy.stats as sts
from astropy.coordinates import SkyCoord

# import rubin_sim.maf as maf
# from rubin_sim.data import get_baseline

def get_depth(coord, band):

    baseline_file = get_baseline()
    name = os.path.basename(baseline_file).replace('.db','')
    out_dir = 'temp'
    results_db = maf.db.ResultsDb(out_dir=out_dir)

    bundle_list = []
    # The point on the sky we would like to get visits for
    ra = [coord.ra.deg]
    dec = [coord.dec.deg]

    metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth', 'visitExposureTime'])
    sql = ''
    slicer = maf.slicers.UserPointsSlicer(ra=ra, dec=dec)
    bundle_list.append(maf.MetricBundle(metric, slicer, sql, run_name=name))
    bd = maf.metricBundles.make_bundles_dict_from_list(bundle_list)
    bg = maf.metricBundles.MetricBundleGroup(bd, baseline_file, out_dir=out_dir, results_db=results_db)
    bg.run_all()

    data_slice = bundle_list[0].metric_values[0]

    # crop off short exposure times
    data_slice = data_slice[np.where(data_slice["visitExposureTime"] > 25.)]

    # Give each filter it's own color
    f2c = {'u': 'purple', 'g': 'blue', 'r': 'green',
        'i': 'cyan', 'z': 'orange', 'y': 'red'}
    
    in_filt = np.where(data_slice['filter']==band)[0]
    data = data_slice['fiveSigmaDepth'][in_filt]
    kde = sts.gaussian_kde(data)

    return kde.resample(1)

def calc_detections_lsst(n, filename, plotname='', 
                         bands=[4, 5, 6, 7, 8, 9], detection_threshold=[23.8, 24.5, 24.03, 23.41, 22.74, 22.96]):
    
    # params = get_params(n_events, filename=filename)
    values = gen_events(n, filename=filename) # 10 for now
    params = get_params(n, filename=filename) #EK_aft
    print('working???', flush=True)
    # get masses
    # with open(f'data/sims/{n}_masses_EK_aft.pkl', 'rb') as f:
    #     masses = pickle.load(f)

    idx_lsst = bands # not UVEX/uvot
        # https://www.lsst.org/scientists/keynumbers
        # u, g, r, i, z, y
    # detection_threshold = detection_threshold

    # params from KN only
    discovery_mag_KN = np.empty(n)
    discovery_band_KN = np.empty(n, dtype='<U11')
    discovery_discwindow_KN = np.empty(n)
    discovery_phase_KN = np.empty(n)
    discovery_window_KN = np.empty(n)
    discovery_windowband_KN = np.empty(n, dtype='<U11')

    # now with the afterglow
    discovery_mag = np.empty(n)
    discovery_band = np.empty(n, dtype='<U11')
    discovery_discwindow = np.empty(n)
    discovery_phase = np.empty(n) 
    discovery_window = np.empty(n)   
    discovery_windowband = np.empty(n, dtype='<U11')

    afterglow_enhance = np.zeros(n)
    discovery_distances = np.full(n, np.nan)

    # TODO: calc the SNR here
    # snr = np.empty(n)

    # may need to adjust per object
    # dist = 160*u.Mpc
    # distmod = Distance(dist).distmod.value

    for i, event in enumerate(values):

        kn_p, _ = params[i]
        dist = kn_p['dist']#160*u.Mpc#
        distmod = Distance(dist).distmod.value

        # convert to app mag
        total = event[1] + distmod
        KN = event[2] + distmod

        # m1, m2 = masses[i]
        # snr[i] = get_snr([m1, m2, dist])

        # check to see which band surpassed the det limit 1st
        discovery_mags_KN = np.zeros(len(idx_lsst)) # should never be larger than this ever
        discovery_mags = np.zeros(len(idx_lsst))
        discovery_phases_KN = np.full(len(idx_lsst), 22.0)
        discovery_phases = np.full(len(idx_lsst), 22.0)
        discovery_windows_KN = np.zeros(len(idx_lsst))
        discovery_windows = np.zeros(len(idx_lsst))
        for j, band in enumerate(idx_lsst):

            # if want to get a depth from distribtuion of rubin sim obs at the location
            # detection_threshold = get_depth(kn_p['coord'], sncosmo_bands[band][-1])

            peak = np.min(KN[band])
            idx_det = KN[band] < detection_threshold[j]
            if peak < detection_threshold[j]:
                discovery_mags_KN[j] = (KN[band][idx_det])[0]
                discovery_phases_KN[j] = (phases[idx_det])[0]
                discovery_windows_KN[j] = (phases[idx_det])[-1] - (phases[idx_det])[0] + 0.2
            #print(sncosmo_bands[band], discovery_mags_KN[j], discovery_phases_KN[j], discovery_windows_KN[j], flush=True)

            # afterglow can cause a rise again
                # get the idx of where the idxs of points above the line are not sequential
                # take first region over the line
            peak = np.min(total[band])
            idx_det = np.where(total[band] < detection_threshold[j])[0]
            if len(idx_det) > 0 and len(np.where(np.diff(idx_det) > 1)[0]) > 0:
                idx2 = np.where(np.diff(idx_det) > 1)[0][0]+1
            #    print(len(idx_det))
                idx_det = idx_det[:idx2]
            if peak < detection_threshold[j]:
                discovery_mags[j] = (total[band][idx_det])[0]
                discovery_phases[j] = (phases[idx_det])[0]
                discovery_windows[j] = (phases[idx_det])[-1] - (phases[idx_det])[0] + 0.2

            #print(sncosmo_bands[band], sncosmo_bands[idx_lsst[j]], discovery_mags[j], discovery_phases[j], discovery_windows[j], flush=True)

        # get mag and band in first band that goes over limit
            # blue bands get silly so need to avoid nans
        idx_disc = np.nanargmin(discovery_phases_KN[np.nonzero(discovery_phases_KN)]) # time when 1st over det limit
        discovery_mag_KN[i] = discovery_mags_KN[idx_disc] # mag that was over det limit
        discovery_band_KN[i] = sncosmo_bands[idx_lsst[idx_disc]] # corresponding band
        discovery_phase_KN[i] = discovery_phases_KN[idx_disc] # time that occurred
        discovery_discwindow_KN[i] = discovery_windows_KN[idx_disc] # time brighter than limit
        idx_maxwindow = np.nanargmax(discovery_windows_KN)
        discovery_window_KN[i] = discovery_windows_KN[idx_maxwindow]
        discovery_windowband_KN[i] = sncosmo_bands[idx_lsst[idx_maxwindow]]

        # repeat with afterglow included
        idx_disc = np.nanargmin(discovery_phases[np.nonzero(discovery_phases)]) 
        discovery_mag[i] = discovery_mags[idx_disc]
        discovery_band[i] = sncosmo_bands[idx_lsst[idx_disc]]
        discovery_phase[i] = discovery_phases[idx_disc]
        discovery_discwindow[i] = discovery_windows[idx_disc]
        idx_maxwindow = np.nanargmax(discovery_windows)
        discovery_window[i] = discovery_windows[idx_maxwindow]
        discovery_windowband[i] = sncosmo_bands[idx_lsst[idx_maxwindow]]

        # if the discovery window is extended by a day, save
        if discovery_window[i] - discovery_window_KN[i] > 3:
            afterglow_enhance[i] = 1
            # save discovery_distances - above init: discovery_distances = full array of nans
            discovery_distances[i] = dist.value
        # if discovery_mag[i] - discovery_mag_KN[i] < -0.5:
        #     afterglow_enhance[i] = 1


    # save it   
    det_df = pd.DataFrame()
    det_df['discovery_mag_KN'] = discovery_mag_KN
    det_df['discovery_band_KN'] = discovery_band_KN
    det_df['discovery_phase_KN'] = discovery_phase_KN
    det_df['discovery_discwindow_KN'] = discovery_discwindow_KN
    det_df['discovery_window_KN'] = discovery_window_KN
    det_df['discovery_windowband_KN'] = discovery_windowband_KN


    det_df['discovery_mag'] = discovery_mag
    det_df['discovery_band'] = discovery_band
    det_df['discovery_phase'] = discovery_phase
    det_df['discovery_discwindow'] = discovery_discwindow
    det_df['discovery_window'] = discovery_window
    det_df['discovery_windowband'] = discovery_windowband
    
    det_df['discovery_distances'] = discovery_distances

    det_df['afterglow_enhance'] = afterglow_enhance

    with open(f'data/sims/{n}_{filename}_detectionStats{plotname}.pkl', 'wb') as f:
        print(f'done detection calc {filename} {plotname}', flush=True)
        pickle.dump(det_df, f)


def hist_detections(n, filename, plotname=''):
    # load in the values
    with open(f'data/sims/{n}_{filename}_detectionStats{plotname}.pkl', 'rb') as f:
        data = pickle.load(f)

    fig, axs = plt.subplots(4, 4, figsize=(20,20))
    axs = axs.flatten()
    for i, (name, vals) in enumerate(data.items()):
        
        if i != len(data.columns)-1:
            # print(vals)

            #count/display discovered KN
            if name == 'discovery_window_KN' or name == 'discovery_window':
                discovered = len([v for v in vals if v >= 3])
                textstr = r'viewable for $\geq 3 = %d$' % (discovered, )
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axs[i].text(0.9, 0.95, textstr, transform=axs[i].transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment ='right', bbox=props)

            axs[i].hist(vals, bins=100)
            axs[i].set_title(name)

    aft_enh = len(np.where(data['afterglow_enhance']!= 0)[0])
    fig.suptitle(f'Afterglow improve detection of {aft_enh} / 5000 events', y = 0.93, fontsize='xx-large')
    fig.savefig(f'img/caps/{n}_events_{filename}_detectionStats{plotname}.png')
    plt.show()

# like above, but split up by bands
def hist_detections_bands(n, filename, plotname=''):
    # load in the values
    with open(f'data/sims/{n}_{filename}_detectionStats{plotname}.pkl', 'rb') as f:
        data = pickle.load(f)

    c = {}
    for i, b in enumerate(sncosmo_bands):
        c[b] = f'C{i}'

    fig, axs = plt.subplots(2, 3, figsize=(13,12))
    axs = axs.flatten().T
    for i, b in enumerate(sncosmo_bands):
        band = b
        color = c[band]

        ax = axs[0]
        mask = data['discovery_band_KN'] == band
        ax.hist(data['discovery_mag_KN'][mask], color=color, bins=50, label=band)
        ax.set_title('discovery_mag_KN')

        ax = axs[1]
        mask = data['discovery_band_KN'] == band
        ax.hist(data['discovery_phase_KN'][mask], color=color, bins=5)
        ax.set_title('discovery_phase_KN')

        ax = axs[2]
        mask = data['discovery_windowband_KN'] == band
        ax.hist(data['discovery_window_KN'][mask], color=color, bins=50)
        ax.set_title('discovery_window_KN')

        ax = axs[3]
        mask = data['discovery_band'] == band
        ax.hist(data['discovery_mag'][mask], color=color, bins=50)
        ax.set_title('discovery_mag')

        ax = axs[4]
        mask = data['discovery_band_KN'] == band
        ax.hist(data['discovery_phase'][mask], color=color, bins=5)
        ax.set_title('discovery_phase')

        ax = axs[5]
        mask = data['discovery_windowband'] == band
        ax.hist(data['discovery_window'][mask], color=color, bins=50)
        ax.set_title('discovery_window')

    axs[0].legend()
    aft_enh = len(np.where(data['afterglow_enhance']!= 0)[0])
    fig.suptitle(f'Afterglow improve detection of {aft_enh} / 5000 events', y = 0.93, fontsize='xx-large')
    fig.savefig(f'img/caps/{n}_events_{filename}_detectionBands{plotname}.png')
    plt.show()

def plotting(n, filename, plotname='',
                         band_idx=[4, 5, 6, 7, 8, 9], detection_threshold=[23.8, 24.5, 24.03, 23.41, 22.74, 22.96]):
    
    # load in the lcs and detections stats
    params = get_params(n, filename='EK_aft')
    values = gen_events(n, filename=filename) 

    with open(f'data/sims/{n}_{filename}_detectionStats{plotname}.pkl', 'rb') as f:
        data = pickle.load(f)

    # select events of interest
    idx_det = np.where(np.array(data['discovery_window']) >= 19)[0]
    bands = np.array(data['discovery_windowband'][idx_det])
    c = {}
    for i, b in enumerate(sncosmo_bands[band_idx[0]: band_idx[-1]+1]):
        c[b] = f'C{i}'
    print(c, flush=True)
    print(bands, flush=True)
    
    # TODO: change per event?
    dist = 160*u.Mpc
    distmod = Distance(dist).distmod.value
    kn = values[idx_det,2] + distmod
    tot = values[idx_det,1] + distmod

    labels_idx = band_idx
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T
    detection_threshold = detection_threshold
    for i, idx in enumerate(labels_idx):
        ax = axs[i]
        ax.axhline(detection_threshold[i], color='black', linestyle='--')
        
        for j in range(len(kn)):
            if bands[j] == 'lsstu':
                ax.plot(phases, kn[j][idx, :], color='gray', 
                        alpha=0.1, linewidth=0.5)
                ax.plot(phases, tot[j][idx, :], color=c[bands[j]], alpha=0.3, linewidth=0.5)
                # if kn[j][idx, 30] < kn[j][idx, -1]:
                #     print(params[idx_det][j], flush=True)

        ax.set_xlabel('time (days)')    
        ax.set_ylabel(r'$M$')
        ax.invert_yaxis()
        ax.set_title(labels[idx])
    axs[0].legend()

    fig.tight_layout()
    fig.savefig(f'img/caps/{n}_events_{filename}_det{plotname}.png')
    plt.show()


    # TODO: get hist of the parameters for these events
    param_filename = 'EK_aft'
    with open(f'data/sims/{n}_params_{param_filename}.pkl', 'rb') as f:
        params = pickle.load(f)

    fig, axs = plt.subplots(5, 2, figsize=(16,16))
    axs = axs.ravel()
    plt.subplots_adjust(hspace=0.4)
    idx_det_20 = np.where(np.array(data['discovery_window']) >= 19)[0]


    legends = ['all events', 'observable for 1 week', 'observable for 3 week']
    for j, parms in enumerate([params.T, params[idx_det].T, params[idx_det_20].T]):
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

        
        
    fig.savefig(f'img/caps/{n}_events_{filename}_detParam{plotname}.png')
    plt.show()

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
    
    # make param data frame
    kn_p, aft_p = params.T
    for d in kn_p:
        for key, value in d.items():
            if key in params_dict.keys():
                params_dict[key].append(value)
    for d in aft_p:
        for key, value in d.items():
            if key in params_dict.keys():
                params_dict[key].append(value)

    params_dict['E0'] = np.log10(params_dict['E0'])
    params_dict['n0'] = np.log10(params_dict['n0'])
    params_dict['thetav/c'] = np.arccos(params_dict['cos_theta']) / params_dict['thetaCore']

    data_df = pd.DataFrame(params_dict)
    print(data_df, flush=True)
    
    cols = ['E0', 'n0', 'thetaCore', 'thetav/c']
    data_df = data_df[cols]
    fig2 = corner.corner(data_df, plot_countours=True, show_titles=True, smooth=2, )
    corner.corner(data_df[np.array(data['discovery_window']) >= 7], plot_countours=True, show_titles=True, smooth=2, color='C1', fig=fig2)
    corner.corner(data_df[np.array(data['discovery_window']) >= 19], plot_countours=True, show_titles=True, smooth=2, color='C2', fig=fig2)
    fig2.savefig(f'img/caps/corner_{filename}{plotname}_subset.png')
    
if __name__ == '__main__':

    print('Starting', flush=True)

    np.random.seed(1647)
    
    UV_bands = ['UVEX::FUV', 'UVEX::NUV']
    UV_labels = ['UVEX FUV', 'UVEX NUV']
    sncosmo_bands = UV_bands + sncosmo_bands
    labels = UV_labels + labels
    
    UV_limiting_mags = [24.5, 24.5]
    sncosmo_lim_mags = [22.3, 22.3, 23.9, 25.0, 24.7, 24.0, 23.3, 22.1]
    sncosmo_lim_mags = UV_limiting_mags + sncosmo_lim_mags
    
    sncosmo_bands += ['f070w', 'f277w', 'f444w', 'f062', 'f146', 'f213']
    labels += ['JWST 70w', 'JWST 200w', 'JWST 444w', 'Roman 62', 'Roman 146wide', 'Roman 213']
    sncosmo_lim_mags += [28.5, 28.7, 28.3, 24.77, 25.37, 23.14]

    labels_idx = np.arange(len(labels))
    
    # default is lsst bands 
    params = {'n': 5000, 'filename': "All", 'plotname': "lsstdist"}

    calc_detections_lsst(**params) #, bands=bands, detection_threshold=detection_threshold)
    hist_detections(**params)
    # hist_detections_bands(**params)
    # plotting(**params, band_idx=bands, detection_threshold=detection_threshold)
