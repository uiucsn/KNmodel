#!/usr/bin/env python
import io
import pickle
import sys
import os
import random
import time
from functools import partial
from urllib import parse, request

import numpy as np
import scipy.stats as spstat
from collections import namedtuple
from astropy.time import Time
from astropy.coordinates import Distance
import astropy.table as at
import astropy.units as u, astropy.constants as c
import argparse
import matplotlib.pyplot as plt
from astropy.visualization import hist
import schwimmbad
from scipy.linalg import cholesky
import scipy.integrate as scinteg
from sklearn.preprocessing import MinMaxScaler

from sed_to_lc import SEDDerviedLC
from dns_mass_distribution import extra_galactic_masses, galactic_masses

import inspiral_range
import ligo.em_bright
import ligo.em_bright.computeDiskMass
from ligo.em_bright.computeDiskMass import computeCompactness, computeDiskMass
import lalsimulation as lalsim
from gwemlightcurves.EjectaFits import DiUj2017, KrFo2019
from kilopop.kilonovae import bns_kilonovae_population_distribution as s22p

def get_options(argv=None):
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_dir', required=True, help='Directory to store simulation results')

    # TODO: Add argument for overlap between survey and ligo run
    # duty factor motivation: https://dcc.ligo.org/public/0167/G2000497/002/G2000497_OpenLVEM_02Apr2020_kk_v2.pdf
    args = parser.parse_args(args=argv)
    return args


if __name__=='__main__':
    argv = sys.argv[1:]

    args = get_options(argv=argv)
    trials_dir = args.trials_dir

    with open(f'{trials_dir}/plotting_data.pickle', 'rb') as f:
        values = pickle.load(f)

    for key,val in values.items():
        exec(key + '=val')

    n_detect2 = np.array(n_detect2)
    n_detect3 = np.array(n_detect3)
    n_detect4 = np.array(n_detect4)

    #print(f"2 det: {n_detect2};\n3 det: {n_detect3};\n4 det: {n_detect4}")
    #print(f"2 det mean: {np.mean(n_detect2)};\n3 det mean: {np.mean(n_detect3)};\n4 det mean: {np.mean(n_detect4)}")
    fig_kw = {'figsize':(9.5/0.7, 3.5)}
    fig, axes = plt.subplots(nrows=1, ncols=3, **fig_kw)

    #ebins = np.logspace(0, 1.53, 10)
    #ebins = np.insert(ebins, 0, 0)
    ebins = np.arange(32)
    norm = np.sum(n_detect3)/np.sum(n_detect2)
    vals, _, _ = axes[0].hist(n_detect2, histtype='stepfilled', \
            bins=ebins, color='C0', alpha=0.3, density=True, zorder=0)

    axes[0].hist(n_detect2, histtype='step', \
                    bins=ebins, color='C0', lw=3, density=True, zorder=3)
    bin_centers = (ebins[0:-1] + ebins[1:])/2.
    mean_nevents = np.mean(n_detect2)
    five_percent, ninetyfive_percent = np.percentile(n_detect2, 5), np.percentile(n_detect2, 95) 
    axes[0].axvline(mean_nevents, color='C0', linestyle='--', lw=2,
                    label=r'$\langle N\rangle = %.2f ;~ N_{95} = %.2f$' % (mean_nevents, ninetyfive_percent))
    axes[0].axvline(ninetyfive_percent, color='C0',
                    linestyle='dotted', lw=1)

    #vals, bins = np.histogram(n_detect3, bins=ebins, density=True)
    mean_nevents = np.mean(n_detect3)
    #vals*=norm
    #test = dict(zip(ebins, vals))
    #print(ebins, vals)
    #print("Test")
    #print(test)
    axes[0].hist(n_detect3, density=True, histtype='stepfilled', color='C1', alpha=0.5, bins=ebins, zorder=1)
    axes[0].hist(n_detect3, density=True, histtype='step', color='C1', lw=3, bins=ebins, zorder=2)
    #axes[0].hist(list(test.keys()), weights=list(test.values()), histtype='stepfilled', color='C1', alpha=0.5, bins=ebins, zorder=1)
    #axes[0].hist(list(test.keys()), weights=list(test.values()), histtype='step', color='C1', lw=3, bins=ebins, zorder=2)
    five_percent, ninetyfive_percent = np.percentile(n_detect3, 5), np.percentile(n_detect3, 95)
    axes[0].axvline(mean_nevents, color='C1', linestyle='--', lw=2,
                label=r'$\langle N\rangle = %.2f ;~ N_{95} = %.2f$' % (mean_nevents, ninetyfive_percent))
    axes[0].axvline(ninetyfive_percent, color='C1',
                linestyle='dotted', lw=1)
    #vals, bins = np.histogram(n_detect4, bins=ebins, density=True)
    # mean_nevents = np.mean(n_detect4)
    # #vals*=nor
    # #test = dict(zip(ebins, vals))
    axes[0].hist(n_detect4, density=True, histtype='stepfilled', color='C2', alpha=0.5, bins=ebins, zorder=1)
    axes[0].hist(n_detect4, density=True, histtype='step', color='C2', lw=3, bins=ebins, zorder=2)
    five_percent, ninetyfive_percent = np.percentile(n_detect4, 5), np.percentile(n_detect4, 95)
    axes[0].axvline(round(mean_nevents), color='C2', linestyle='--', lw=2,
                    label=r'$\langle N\rangle = %.2f ;~ N_{95} = %.2f$' % (mean_nevents, ninetyfive_percent))
    axes[0].axvline(ninetyfive_percent, color='C2',
                    linestyle='dotted', lw=1)
    axes[0].legend(frameon=False, fontsize='medium', loc='upper right')
    #axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim((0., 31))
    #axes[0].set_ylim((1e-2, 1))
    #######################################################
    ### print out probabilities of greater than 1 event ###
    #######################################################
    print("P(N > 1 event detected)")
    print("For two detector", np.sum(n_detect2 > 1)/len(n_detect2))
    print("For three detector", np.sum(n_detect3 > 1)/len(n_detect2))
    print("For four detector", np.sum(n_detect4 > 1)/len(n_detect2))

    dist_range = np.arange(0, 400., 0.1)
    patches = list()
    legend_text = list()
    try:
        kde = spstat.gaussian_kde(dist_detect2, bw_method='scott')
        pdist = kde(dist_range)
        axes[1].plot(dist_range, pdist, color='C0', linestyle='-', lw=3, zorder=4)
        patch1 = axes[1].fill_between(dist_range, np.zeros(len(dist_range)), pdist, color='C0', alpha=0.3, zorder=0)
        patches.append(patch1)
        legend_text.append('2 Detector Events')
        mean_dist = np.mean(dist_detect2)
        axes[1].axvline(mean_dist, color='C0', linestyle='--', lw=1.5, zorder=6, label=r'$\langle D \rangle = {:.0f}$ Mpc'.format(mean_dist))
        ind0_40 = dist_range <= 40.
        ind40_80 = (dist_range <= 100.) & (dist_range > 40.)
        ind80_160 = (dist_range <= 160.) & (dist_range > 100.)
        p0_40 = scinteg.trapz(pdist[ind0_40], dist_range[ind0_40])
        p40_80 = scinteg.trapz(pdist[ind40_80], dist_range[ind40_80])
        p80_160 = scinteg.trapz(pdist[ind80_160], dist_range[ind80_160])
        print(p0_40*5, p40_80*5, p80_160*5)
    except ValueError:
        print("Could not create KDE since no 2-det detection")

    try:
        kde = spstat.gaussian_kde(dist_detect3, bw_method='scott')
        pdist = kde(dist_range)
        axes[1].plot(dist_range, pdist, color='C1', linestyle='-', lw=3, zorder=2)
        patch2 = axes[1].fill_between(dist_range, np.zeros(len(dist_range)), pdist, color='C1', alpha=0.5, zorder=1)
        patches.append(patch2)
        legend_text.append('3 Detector Events')
        mean_dist = np.mean(dist_detect3)
        axes[1].axvline(mean_dist, color='C1', linestyle='--', lw=1.5, zorder=6, label=r'$\langle D \rangle = {:.0f}$ Mpc'.format(mean_dist))
        axes[1].legend(frameon=False, fontsize='medium')
    except ValueError:
        print("Could not create KDE since no 3-det detection")

    try:
        kde = spstat.gaussian_kde(dist_detect4, bw_method='scott')
        pdist = kde(dist_range)
        mean_dist = np.mean(dist_detect4)
        axes[1].plot(dist_range, pdist, color='C2', linestyle='-', lw=3, zorder=2)
        axes[1].axvline(mean_dist, color='C2', linestyle='--', lw=1.5, zorder=6, label=r'$\langle D \rangle = {:.0f}$ Mpc'.format(mean_dist))
        patch3 = axes[1].fill_between(dist_range, np.zeros(len(dist_range)), pdist, color='C2', alpha=0.5, zorder=1)
        patches.append(patch3)
        legend_text.append('4 Detector Events')
        axes[1].legend(frameon=False, fontsize='medium')
    except ValueError:
        print("Could not create KDE since no 4-det detection")

    h_range = np.arange(15, 23, 0.1)
    kde = spstat.gaussian_kde(mag_detect2, bw_method='scott')
    kde_peak =  spstat.gaussian_kde(mag_peak2, bw_method='scott')
    ph = kde(h_range)
    peak_ph = kde_peak(h_range)

    axes[2].plot(h_range, ph, color='C0', linestyle='dotted', lw=3, zorder=4, label='Detection mag')
    axes[2].plot(h_range, peak_ph, color='C0', linestyle='-', lw=3, zorder=4, label='Peak mag')
    axes[2].fill_between(h_range, np.zeros(len(h_range)), peak_ph, color='C0', alpha=0.3, zorder=0)
    mean_h = np.mean(mag_peak2)
    axes[2].axvline(mean_h, color='C0', linestyle='--', lw=1.5, zorder=6, label=r'$\langle r \rangle = {:.1f}$ mag'.format(mean_h))

    kde = spstat.gaussian_kde(mag_detect3, bw_method='scott')
    kde_peak =  spstat.gaussian_kde(mag_peak3, bw_method='scott')
    ph = kde(h_range)
    peak_ph = kde_peak(h_range)

    axes[2].plot(h_range, ph, color='C1', linestyle='dotted', lw=3, zorder=2, label='Discovery mag')
    axes[2].plot(h_range, peak_ph, color='C1', linestyle='-', lw=3, zorder=2, label='Peak mag')
    axes[2].fill_between(h_range, np.zeros(len(h_range)), peak_ph, color='C1', alpha=0.5, zorder=1)
    mean_h = np.mean(mag_peak3)
    axes[2].axvline(mean_h, color='C1', linestyle='--', lw=1.5, zorder=6, label=r'$\langle r \rangle = {:.1f}$ mag'.format(mean_h))
    axes[2].legend(frameon=False, fontsize='medium')

    try:
        kde = spstat.gaussian_kde(mag_detect4, bw_method='scott')
        ph = kde(h_range)
        axes[2].plot(h_range, ph, color='C2', linestyle='-', lw=3, zorder=2)
        axes[2].fill_between(h_range, np.zeros(len(h_range)), ph, color='C1', alpha=0.5, zorder=1)
        mean_h = np.mean(mag_detect4)
        axes[2].axvline(mean_h, color='C2', linestyle='--', lw=1.5, zorder=6, label=r'$\langle H \rangle = {:.1f}$ mag'.format(mean_h))
        axes[2].legend(frameon=False, fontsize='medium')
    except ValueError:
        print("Could not create KDE for h-mag since no 4 detector events found")

    axes[1].set_xlabel('Distance ($D$, Mpc)', fontsize='x-large')
    axes[1].set_ylabel('$P(D)$', fontsize='x-large')
    #if args.mass_distrib != 'msp':
    #    axes[0].set_title(f"Masses {args.mass_distrib}; {args.masskey1} -- {args.masskey2}")
    #else:
    #    axes[0].set_title("MSP bimodal mass @ 1.393 / 1.807 $M_{\odot}$")
    axes[0].set_xlabel('Number of Events ($N$)', fontsize='x-large')
    axes[0].set_ylabel('$P(N)$', fontsize='x-large')

    axes[2].set_xlabel('Apparent ($r$, AB mag)', fontsize='x-large')
    axes[2].set_ylabel('$P(r)$', fontsize='x-large')
    axes[0].set_xlim(0, ebins.max())

    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(0, ymax)
    axes[1].set_xlim(0, 200)
    ymin, ymax = axes[2].get_ylim()
    axes[2].set_ylim(0, ymax)
    axes[2].set_xlim(16.5, 22.8)

    fig.legend(patches, legend_text,
                'upper center', frameon=False, ncol=3, fontsize='medium')
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=1.05)
    fig.savefig(f'{trials_dir}/mc_plot.pdf')
    plt.show()

    gw_lost = 1 - np.array(gw_recovered)
    em_lost = 1 - np.array(em_recovered)

    gw_lost_mean = np.mean(gw_lost)
    em_lost_mean = np.mean(em_lost)

    plt.hist(gw_lost, density=True, label=r"$LF_{GW}$", histtype=u'step', linewidth=3, color='C0')
    plt.hist(em_lost, density=True, label=r"$LF_{EM}$", histtype=u'step', linewidth=3, color='C1')

    plt.axvline(gw_lost_mean, linestyle='dotted', label=r"$\langle LF_{GW} \rangle$", color='C0')
    plt.axvline(em_lost_mean, linestyle='dotted', label=r"$\langle LF_{EM} \rangle$", color='C1')

    plt.xlabel('Fraction of events lost')
    plt.ylabel('Relative Number of trials')
    plt.legend()


    plt.savefig(f'{trials_dir}/loss_fractions.pdf')
    plt.show()