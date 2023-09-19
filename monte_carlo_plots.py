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
    parser.add_argument('--obs_run', required=True, choices=['O4','O5'], help='Observing run')

    args = parser.parse_args(args=argv)
    return args


if __name__=='__main__':
    argv = sys.argv[1:]

    args = get_options(argv=argv)
    trials_dir = args.trials_dir
    obs_run = args.obs_run

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
    ebins = np.arange(np.max(n_detect2))
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
    mean_nevents = np.mean(n_detect4)
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
    axes[0].set_xlim((0., np.max(n_detect2)))
    #axes[0].set_ylim((1e-2, 1))
    #######################################################
    ### print out probabilities of greater than 1 event ###
    #######################################################
    print("P(N > 1 event detected)")
    print("For two detector", np.sum(n_detect2 > 1)/len(n_detect2))
    print("For three detector", np.sum(n_detect3 > 1)/len(n_detect2))
    print("For four detector", np.sum(n_detect4 > 1)/len(n_detect2))

    dist_range = np.arange(0, 450., 0.1)
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

    h_range = np.arange(15, 24, 0.1)
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
    if obs_run == 'O4':
        axes[1].set_xlim(0, 255)
    elif obs_run == 'O5':
        axes[1].set_xlim(0, 455)
    ymin, ymax = axes[2].get_ylim()
    axes[2].set_ylim(0, ymax)
    if obs_run == 'O4':
        axes[2].set_xlim(16.5, 23)
    elif obs_run == 'O5':
        axes[2].set_xlim(16.5, 24)

    fig.legend(patches, legend_text,
                'upper center', frameon=False, ncol=3, fontsize='medium')
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=1.05)
    fig.savefig(f'{trials_dir}/mc_plot.pdf')
    plt.show()

    # # Figure for losses

    # gw_mean = np.mean(gw_recovered) * 100
    # em_mean = np.mean(em_recovered) * 100
    # single_gw_detection_mean = np.mean(single_gw_detection) * 100

    # plt.hist(np.array(gw_recovered) * 100, density=True, label=r"$F_{GW}$", histtype=u'step', linewidth=3, color='C0')
    # plt.hist(np.array(em_recovered) * 100, density=True, label=r"$F_{EM}$", histtype=u'step', linewidth=3, color='C1')
    # plt.hist(np.array(single_gw_detection) * 100, density=True, label=r"$F_{1GW+EM}$", histtype=u'step', linewidth=3, color='C2')

    # plt.axvline(gw_mean, linestyle='dotted', label=r"$\langle F_{{GW}} \rangle = {:.1f}$".format(gw_mean), color='C0')
    # plt.axvline(em_mean, linestyle='dotted', label=r"$\langle F_{{EM}} \rangle = {:.1f}$".format(em_mean), color='C1')
    # plt.axvline(single_gw_detection_mean, linestyle='dotted', label=r"$\langle F_{{1GW+EM}} \rangle = {:.1f}$".format(single_gw_detection_mean), color='C2')

    # plt.xlabel('Percent of events')
    # plt.ylabel('Relative Number of trials')
    # plt.legend()


    # plt.savefig(f'{trials_dir}/loss_fractions.pdf')
    # plt.show()

    # # Figure for discovery window 

    # discovery_window2_mean = np.mean(discovery_window2)
    # discovery_window3_mean = np.mean(discovery_window3)
    # discovery_window4_mean = np.mean(discovery_window4)

    # discovery_window2_std = np.std(discovery_window2)
    # discovery_window3_std = np.std(discovery_window3)
    # discovery_window4_std = np.std(discovery_window4)

    # discovery_window2_median = np.median(discovery_window2)
    # discovery_window3_median = np.median(discovery_window3)
    # discovery_window4_median = np.median(discovery_window4)

    # discovery_window2_95 = np.percentile(discovery_window2, 95)
    # discovery_window3_95 = np.percentile(discovery_window3, 95)
    # #discovery_window4_95 = np.percentile(discovery_window4, 95)

    # discovery_window2_5 = np.percentile(discovery_window2, 5)
    # discovery_window3_5 = np.percentile(discovery_window3, 5)
    # #discovery_window4_5 = np.percentile(discovery_window4, 5)


    # plt.hist(discovery_window2, density=True, histtype=u'step', linewidth=3, color='C0', label='2 detector')
    # plt.hist(discovery_window3, density=True, histtype=u'step', linewidth=3, color='C1', label='3 detector')
    # plt.hist(discovery_window4, density=True, histtype=u'step', linewidth=3, color='C2', label='4 detector')

    # plt.axvline(discovery_window2_mean, linestyle='dotted', color='C0')
    # plt.axvline(discovery_window3_mean, linestyle='dotted', color='C1')

    # plt.xlabel('Discovery window (days)')
    # plt.ylabel('Relative number of events')


    # plt.savefig(f'{trials_dir}/discovery_windows.pdf')
    # plt.show()

    # print(f"Discovery windows:\n2 det mean: {discovery_window2_mean} std: {discovery_window2_std}\n3 det: {discovery_window3_mean} std: {discovery_window3_std}\n4 det: {discovery_window4_mean} std: {discovery_window4_std}")
    # print(f"Discovery windows:\n2 det 5th: {discovery_window2_5} median: {discovery_window2_median} 95th: {discovery_window2_95}\n3 det 5th: {discovery_window3_5} median: {discovery_window3_median} 95th: {discovery_window3_95}") # \n4 det 5th: {discovery_window4_5} median: {discovery_window4_median} 95th: {discovery_window4_95}\n")

    n_iterations = np.arange(1, len(n_detect2) + 1)

    cumulative_averages2 = np.cumsum(n_detect2) / n_iterations
    cumulative_averages3 = np.cumsum(n_detect3) / n_iterations
    cumulative_averages4 = np.cumsum(n_detect4) / n_iterations

    iteration_start = 10

    plt.plot(n_iterations[iteration_start:], cumulative_averages2[iteration_start:])
    plt.xlabel('Number of iterations')
    plt.ylabel('Cumulative average of 2 detector events')
    plt.axhline(y = np.mean(n_detect2), color='red')
    plt.savefig(f'{trials_dir}/cumulative_avg_2.png')
    plt.show()

    plt.plot(n_iterations[iteration_start:], cumulative_averages3[iteration_start:])
    plt.xlabel('Number of iterations')
    plt.ylabel('Cumulative average of 3 detector events')
    plt.axhline(y = np.mean(n_detect3), color='red')
    plt.savefig(f'{trials_dir}/cumulative_avg_3.png')
    plt.show()

    plt.plot(n_iterations[iteration_start:], cumulative_averages4[iteration_start:])
    plt.xlabel('Number of iterations')
    plt.ylabel('Cumulative average of 4 detector events')
    plt.axhline(y = np.mean(n_detect4), color='red')
    plt.savefig(f'{trials_dir}/cumulative_avg_4.png')
    plt.show()