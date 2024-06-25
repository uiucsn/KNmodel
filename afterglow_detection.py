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

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import phases
from sed_to_lc import SEDDerviedLC, lsst_bands
from afterglow_distribution import gen_events, get_params, sncosmo_bands, labels, labels_idx

UV_bands = ['UVEX::FUV', 'UVEX::NUV']
UV_labels = ['UVEX FUV', 'UVEX NUV']
#labels_idx = np.arange(len(labels))

sncosmo_bands = UV_bands + sncosmo_bands
labels = UV_labels + labels
labels_idx = np.arange(len(labels))


def calc_detections_lsst(n, filename):
    
    # params = get_params(n_events, filename=filename)
    values = gen_events(n, filename=filename) # 10 for now

    idx_lsst = [4,5,6,7,8,9] # not UVEX/uvot
        # https://www.lsst.org/scientists/keynumbers
        # u, g, r, i, z, y
    detection_threshold = [23.8, 24.5, 24.03, 23.41, 22.74, 22.96]

    # params from KN only
    discovery_mag_KN = np.empty(n)
    discovery_band_KN = np.empty(n, dtype='<U11')
    discovery_phase_KN = np.empty(n)
    discovery_window_KN = np.empty(n)
    discovery_windowband_KN = np.empty(n, dtype='<U11')

    # now with the afterglow
    discovery_mag = np.empty(n)
    discovery_band = np.empty(n, dtype='<U11')
    discovery_phase = np.empty(n) 
    discovery_window = np.empty(n)   
    discovery_windowband = np.empty(n, dtype='<U11')

    afterglow_enhance = np.zeros(n)

    # may need to adjust per object
    dist = 160*u.Mpc
    distmod = Distance(dist).distmod.value

    for i, event in enumerate(values):

        # convert to app mag
        total = event[1] + distmod
        KN = event[2] + distmod

        # check to see which band surpassed the det limit 1st
        discovery_mags_KN = np.zeros(len(lsst_bands))
        discovery_mags = np.zeros(len(lsst_bands))
        discovery_phases_KN = np.zeros(len(lsst_bands))
        discovery_phases = np.zeros(len(lsst_bands))
        discovery_windows_KN = np.zeros(len(lsst_bands))
        discovery_windows = np.zeros(len(lsst_bands))
        for j, band in enumerate(idx_lsst):

            peak = np.min(KN[band])
            idx_det = KN[band] < detection_threshold[j]
            if peak < detection_threshold[j]:
                discovery_mags_KN[j] = (KN[band][idx_det])[0]
                discovery_phases_KN[j] = (phases[idx_det])[0]
                discovery_windows_KN[j] = (phases[idx_det])[-1] - (phases[idx_det])[0] + 0.2
            print(sncosmo_bands[band], discovery_mags_KN[j], discovery_phases_KN[j], discovery_windows_KN[j], flush=True)

            # afterglow can cause a rise again
                # get the idx of where the idxs of points above the line are not sequential
                # take first region over the line
            peak = np.min(total[band])
            idx_det = np.where(total[band] < detection_threshold[j])[0]
            if len(idx_det) > 0 and len(np.where(np.diff(idx_det) > 1)[0]) > 0:
                idx2 = np.where(np.diff(idx_det) > 1)[0][0]+1
                print(len(idx_det))
                idx_det = idx_det[:idx2]
            if peak < detection_threshold[j]:
                discovery_mags[j] = (total[band][idx_det])[0]
                discovery_phases[j] = (phases[idx_det])[0]
                discovery_windows[j] = (phases[idx_det])[-1] - (phases[idx_det])[0] + 0.2

            print(sncosmo_bands[band], sncosmo_bands[idx_lsst[j]], discovery_mags[j], discovery_phases[j], discovery_windows[j], flush=True)

        # get mag and band in first band that goes over limit
        discovery_mag_KN[i] = discovery_mags_KN[np.argmax(-1*discovery_phases_KN)]
        discovery_band_KN[i] = sncosmo_bands[idx_lsst[np.argmax(-1*discovery_phases_KN)]]
        discovery_phase_KN[i] = np.min(discovery_phases_KN)
        # get the band with the longest time over limit
        discovery_window_KN[i] = np.max(discovery_windows_KN)
        discovery_windowband_KN[i] = sncosmo_bands[idx_lsst[np.argmax(discovery_windows_KN)]]

        # repeat with afterglow included
        discovery_mag[i] = discovery_mags[np.argmax(-1*discovery_phases)]
        discovery_band[i] = sncosmo_bands[idx_lsst[np.argmax(-1*discovery_phases)]]
        discovery_phase[i] = np.min(discovery_phases)
        discovery_window[i] = np.max(discovery_windows)
        discovery_windowband[i] = sncosmo_bands[idx_lsst[np.argmax(discovery_windows)]]

        # if the discovery window is extended by a day, save
        if discovery_window[i] - discovery_window_KN[i] > 3:
            afterglow_enhance[i] = 1
        # if discovery_mag[i] - discovery_mag_KN[i] < -0.5:
        #     afterglow_enhance[i] = 1


    # save it   
    det_df = pd.DataFrame()
    det_df['discovery_mag_KN'] = discovery_mag_KN
    det_df['discovery_band_KN'] = discovery_band_KN
    det_df['discovery_phase_KN'] = discovery_phase_KN
    det_df['discovery_window_KN'] = discovery_window_KN
    det_df['discovery_windowband_KN'] = discovery_windowband_KN


    det_df['discovery_mag'] = discovery_mag
    det_df['discovery_band'] = discovery_band
    det_df['discovery_phase'] = discovery_phase
    det_df['discovery_window'] = discovery_window
    det_df['discovery_windowband'] = discovery_windowband
    det_df['afterglow_enhance'] = afterglow_enhance

    print(det_df, flush=True)

    with open(f'data/sims/{n}_{filename}_detectionStats.pkl', 'wb') as f:
        print(f'done detection calc {filename}', flush=True)
        pickle.dump(det_df, f)


def hist_detections(n, filename):
    # load in the values
    with open(f'data/sims/{n}_{filename}_detectionStats.pkl', 'rb') as f:
        data = pickle.load(f)

    fig, axs = plt.subplots(2, 5, figsize=(16,12))
    axs = axs.flatten()
    for i, (name, vals) in enumerate(data.items()):
        
        if i != len(data.columns)-1:
            print(vals)
            axs[i].hist(vals, bins=100)
            axs[i].set_title(name)

    aft_enh = len(np.where(data['afterglow_enhance']!= 0)[0])
    fig.suptitle(f'Afterglow improve detection of {aft_enh} / 5000 events', y = 0.93, fontsize='xx-large')
    fig.savefig(f'img/caps/{n}_events_{filename}_detectionStats.png')
    plt.show()


def plotting(n, filename):
    # load in the lcs and detections stats
    params = get_params(n, filename=filename)
    values = gen_events(n, filename=filename) 

    with open(f'data/sims/{n}_{filename}_detectionStats.pkl', 'rb') as f:
        data = pickle.load(f)

    # select events of interest
    idx_det = np.where(np.array(data['discovery_window']) >= 7)[0]
    bands = np.array(data['discovery_windowband'][idx_det])
    c = {}
    for i, b in enumerate('ugrizy'):
        c[f'lsst{b}'] = f'C{i}'
    print(c, flush=True)
    print(bands, flush=True)
    
    dist = 160*u.Mpc
    distmod = Distance(dist).distmod.value
    kn = values[idx_det,2] + distmod
    tot = values[idx_det,1] + distmod

    labels_idx = [4,5,6,7,8,9]
    n_plots = int(len(labels_idx)/2) + (len(labels_idx)%2)
    fig, axs = plt.subplots(n_plots, 2, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    axs = axs.flatten().T
    detection_threshold = [23.8, 24.5, 24.03, 23.41, 22.74, 22.96]
    for i, idx in enumerate(labels_idx):
        ax = axs[i]
        ax.axhline(detection_threshold[i], color='black', linestyle='--')
        
        for j in range(len(kn)):
            if bands[j] == 'lsstg':
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
    fig.savefig(f'img/caps/{n}_events_{filename}_det.png')
    plt.show()


    # TODO: get hist of the parameters for these events
    with open(f'data/sims/{n}_params_{filename}.pkl', 'rb') as f:
            params = pickle.load(f)

    fig, axs = plt.subplots(5, 2, figsize=(16,16))
    axs = axs.ravel()
    plt.subplots_adjust(hspace=0.4)
    idx_det_20 = np.where(np.array(data['discovery_window']) >= 19)[0]

    for parms in [params.T, params[idx_det].T, params[idx_det_20].T]:
        kn_p, aft_p = parms # [ [kn, aft], [kn, aft]] -> [[kn, kn], [aft, aft]]

        kn_params = {"mej_dyn": [], 
                    "mej_wind": [], 
                    "phi": [], 
                    "cos_theta": [], 
                    #  "dist": [], 
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
            axs[i].hist(value, bins=100, density=True, alpha=0.6)
            axs[i].set_title(param)

        for i, (param, value) in enumerate(aft_params.items()):
            
            if param == 'E0' or param == 'n0':
                value = np.log10(value)
            
            axs[i+5].hist(value, bins=100, density=True, alpha=0.6)
            axs[i+5].set_title(param)
        
    fig.savefig(f'img/caps/{n}_events_{filename}_detParam.png')
    plt.show()
    

if __name__ == '__main__':

    params = {'n': 5000, 'filename': "EK_aft"}
    calc_detections_lsst(**params)
    hist_detections(**params)
    plotting(**params)