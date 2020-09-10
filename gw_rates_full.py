#!/usr/bin/env python
import sys
import os
import numpy as np
import scipy.stats as spstat
from collections import namedtuple
from astropy.time import Time
from astropy.coordinates import Distance
import astropy.table as at
import astropy.units as u
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.visualization import hist
import schwimmbad
from scipy.linalg import cholesky
import scipy.integrate as scinteg
from sklearn.preprocessing import MinMaxScaler


def get_correlated_series(n_events, upper_chol):
    """
    Get some correlated uniformly distributed random series between 0 and 1
    """
    rnd = np.random.uniform(0., 1., size=(n_events, 3))
    series = rnd @ upper_chol
    return series

def get_sim_dutycycles(n_events, upper_chol, h_duty, l_duty, v_duty):
    """
    Get some correlated duty cycle series
    """
    series = get_correlated_series(n_events, upper_chol)
    scaler = MinMaxScaler()
    scaler.fit(series)
    series = scaler.transform(series)
    series = series.T
    duty_cycles = np.zeros(series.shape)

    h_series = series[0,:]
    l_series = series[1,:]
    v_series = series[2,:]

    h_on = duty_cycles[0,:]
    l_on = duty_cycles[1,:]
    v_on = duty_cycles[2,:]

    h_on[h_series >= h_duty] = 1
    l_on[l_series >= l_duty] = 1
    v_on[v_series >= v_duty] = 1

    h_on = h_on.astype(bool)
    l_on = l_on.astype(bool)
    v_on = v_on.astype(bool)

    return h_on, l_on, v_on


class MinZeroAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values <= 0 :
            parser.error("Minimum value for {0} is 0".format(option_string))
        setattr(namespace, self.dest, values)


def get_options(argv=None):
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mass_distrib', choices=['mw','flat'], default='mw', help='Picky BNS mass distribution')
    parser.add_argument('--masskey1', type=float, action=MinZeroAction, default=1.4, help='Specify  Mass Keyword 1 (mw = mean, flat=lower bound)')
    parser.add_argument('--masskey2', type=float, action=MinZeroAction, default=0.09, help='Specify  Mass Keyword 2 (mw = sigma, flat=upper bound)')
    parser.add_argument('--ligo_horizon', default=105., action=MinZeroAction, type=float,\
            help='Specify the minimum horizon distance for BNS events from LIGO')
    parser.add_argument('--virgo_horizon', default=60., action=MinZeroAction, type=float,\
            help='Specify the minimum horizon distance for Virgo events from LIGO')
    # Ryan's original value was -5.95 - the update comes from Alexandra Corsi's conservative estimate
    # 4.7d-6*4./3.*!pi*(170.)^3.*0.75*0.7 --> ~50
    # 3.2d-7*4./3.*!pi*(120.)^3.*0.75*0.7 --> 1
    # BTW: conservative and reasonable choice is 1.54d-6*4./3.*!pi*(120.)^3.*0.75*0.7 --> 5-6 events (median)
    parser.add_argument('--ntry', default=10000, type=int, action=MinZeroAction, help='Set the number of MC samples')
    parser.add_argument('--box_size', default=500., action=MinZeroAction, type=float,\
            help='Specify the side of the box in which to simulate events')
    parser.add_argument('--mean_lograte', default=-5.95, help='specify the lograthim of the mean BNS rate')
    parser.add_argument('--sig_lograte',  default=0.55, help='specify the std of the mean BNS rate')
    parser.add_argument('--chirp_scale',  default=2.66, action=MinZeroAction, type=float, help='Set the chirp scale')
    parser.add_argument('--hdutycycle', default=0.7, action=MinZeroAction, type=float, help='Set the Hanford duty cycle')
    parser.add_argument('--ldutycycle', default=0.7, action=MinZeroAction, type=float, help='Set the Livingston duty cycle')
    parser.add_argument('--vdutycycle', default=0.6, action=MinZeroAction, type=float, help='Set the Virgo duty cycle')
    args = parser.parse_args(args=argv)
    return args


def main(argv=None):

    args = get_options(argv=argv)
    np.random.seed(seed=42)

    # setup time-ranges
    ligo_run_start = Time('2019-04-01T00:00:00.0')
    ligo_run_end   = Time('2020-04-01T00:00:00.0')
    hst_cyc_start  = Time('2019-10-01T00:00:00.0')
    hst_cyc_end    = Time('2020-09-30T00:00:00.0')
    eng_time       = 2.*u.week
    Range = namedtuple('Range', ['start', 'end'])
    ligo_run  = Range(start=ligo_run_start, end=ligo_run_end)
    hst_cycle = Range(start=hst_cyc_start,  end=hst_cyc_end)
    latest_start = max(ligo_run.start, hst_cycle.start)
    earliest_end = min(ligo_run.end, hst_cycle.end)
    td = (earliest_end - latest_start) + eng_time
    fractional_duration = (td/(1.*u.year)).decompose().value

    # setup horizons
    bns_ligo_horizon  =  args.ligo_horizon*u.megaparsec
    bns_virgo_horizon =  args.virgo_horizon*u.megaparsec
    box_size = args.box_size
    volume = box_size**3


    # create the mass distribution of the merging neutron star
    mass_distrib = args.mass_distrib
    if mass_distrib == 'mw':
        # the truncated normal distribution looks to be from:
        # https://arxiv.org/pdf/1309.6635.pdf
        mean_mass = args.masskey1
        sig_mass  = args.masskey2
    else:
        min_mass = args.masskey1
        max_mass = args.masskey2

    # the two ligo detectors ahve strongly correlated duty cycles
    # they are both not very correlated with Virgo
    lvc_cor_matrix = np.array([[1., 0.8, 0.2],
                        [0.8, 1., 0.2],
                        [0.2, 0.2, 1.]])
    upper_chol = cholesky(lvc_cor_matrix)

    # setup duty cycles
    h_duty = args.hdutycycle
    l_duty = args.ldutycycle
    v_duty = args.vdutycycle

    # setup event rates
    mean_lograte = args.mean_lograte
    sig_lograte  = args.sig_lograte
    chirp_scale  = args.chirp_scale
    n_try = args.ntry

    temp = at.Table.read('kilonova_phottable_40Mpc.txt', format='ascii')
    phase = temp['ofphase']
    temphmag  = temp['f160w']
    temprmag  = temp['f625w']
    def dotry(n):

        rate = 10.**(np.random.normal(mean_lograte, sig_lograte))
        n_events = np.around(rate*volume*fractional_duration).astype('int_')

        if mass_distrib == 'mw':
            mass1 = spstat.truncnorm.rvs(0, np.inf, mean_mass, sig_mass, n_events)
            mass2 = spstat.truncnorm.rvs(0, np.inf, mean_mass, sig_mass, n_events)
        else:
            mass1 = np.random.uniform(min_mass, max_mass, n_events)
            mass2 = np.random.uniform(min_mass, max_mass, n_events)
        tot_mass = mass1 + mass2

        delay = np.random.uniform(0, 365.25, n_events)
        delay[delay > 80] = 0

        av = np.random.exponential(1, n_events)*0.4
        ah = av/6.1

        sss17a = -16.9 #H-band
        sss17a_r = -15.8 #Rband
        minmag = -14.7
        maxmag = sss17a - 2.

        hmag = temphmag - min(temphmag)
        hmag[phase <2.5] = 0
        magindex = [(phase - x).argmin() for x in delay]
        magindex = np.array(magindex)

        default_value= [0,]
        if n_events == 0:
            return default_value, default_value, default_value, default_value, default_value, default_value, 0, 0



        absm = np.random.uniform(0, 1, n_events)*abs(maxmag-minmag) + sss17a + hmag[magindex] + ah
        absm = np.array(absm)

        # simulate coordinates
        x = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec
        y = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec
        z = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec
        dist = (x**2. + y**2. + z**2. + (0.05*u.megaparsec)**2.)**0.5

        h_on, l_on, v_on = get_sim_dutycycles(n_events, upper_chol, h_duty, l_duty, v_duty)

        dist_ligo_bool  = dist < bns_ligo_horizon*tot_mass/chirp_scale
        dist_virgo_bool = dist < bns_virgo_horizon*tot_mass/chirp_scale
        distmod = Distance(dist)
        obsmag = absm + distmod.distmod.value
        em_bool = obsmag < 22.

        two_det_bool = (h_on & l_on) | (v_on & (h_on | l_on))
        three_det_bool = (h_on & l_on & v_on)

        n2_good = np.where(dist_ligo_bool & two_det_bool & em_bool)[0]
        n2 = len(n2_good)
        n3_good = np.where(dist_virgo_bool & three_det_bool & em_bool)[0]
        n3 = len(n3_good)

        return dist[n2_good].value.tolist(), tot_mass[n2_good].tolist(),\
                dist[n3_good].value.tolist(), tot_mass[n3_good].tolist(),\
                obsmag[n2_good].tolist(), obsmag[n3_good].tolist(),\
                n2, n3

    with schwimmbad.SerialPool() as pool:
        values = list(pool.map(dotry, range(n_try)))

    n_detect2 = []
    n_detect3 = []
    dist_detect2 = []
    mass_detect2 = []
    dist_detect3 = []
    mass_detect3 = []
    hmag_detect2 = []
    hmag_detect3 = []
    for d2, m2, d3, m3,h2, h3,n2, n3 in values:
        if n2 >= 0:
            n_detect2.append(n2)
            if n3>0:
                dist_detect2 += d2
                mass_detect2 += m2
                hmag_detect2 += h2
        if n3>=0:
            n_detect3.append(n3)
            if n3 > 0:
                dist_detect3 += d3
                mass_detect3 += m3
                hmag_detect3 += h3

    n_detect2 = np.array(n_detect2)
    n_detect3 = np.array(n_detect3)

    fig_kw = {'figsize':(9.5/0.7, 3.5)}
    fig, axes = plt.subplots(nrows=1, ncols=3, **fig_kw)


    ebins = np.arange(-0.5,20.5, 1)
    norm = np.sum(n_detect3)/np.sum(n_detect2)
    vals, _, _ = axes[0].hist(n_detect2, histtype='stepfilled', \
            bins=ebins, color='C0', alpha=0.3, density=True, zorder=0)
    axes[0].hist(n_detect2, histtype='step', \
            bins=ebins, color='C0', lw=3, density=True, zorder=3)
    bin_centers = (ebins[0:-1] + ebins[1:])/2.
    mean_nevents = np.sum(vals*bin_centers)/np.sum(vals)
    axes[0].axvline(np.ceil(mean_nevents), color='C0', linestyle='--', lw=1.5, label=r'$\langle N \rangle = {:n}$'.format(np.ceil(mean_nevents)))

    vals, bins = np.histogram(n_detect3, bins=ebins, density=True)
    mean_nevents = np.sum(vals*bin_centers)/np.sum(vals)
    vals*=norm
    test = dict(zip(ebins, vals))
    axes[0].hist(list(test.keys()), weights=list(test.values()), histtype='stepfilled', color='C1', alpha=0.5, bins=ebins, zorder=1)
    axes[0].hist(list(test.keys()), weights=list(test.values()), histtype='step', color='C1', lw=3, bins=ebins, zorder=2)
    axes[0].axvline(np.around(mean_nevents), color='C1', linestyle='--', lw=1.5, label=r'$\langle N \rangle = {:n}$'.format(np.around(mean_nevents)))
    axes[0].legend(frameon=False, fontsize='small')

    dist_range = np.arange(0, 150., 0.1)
    kde = spstat.gaussian_kde(dist_detect2, bw_method='scott')
    pdist = kde(dist_range)
    axes[1].plot(dist_range, pdist, color='C0', linestyle='-', lw=3, zorder=4)
    patch1 = axes[1].fill_between(dist_range, np.zeros(len(dist_range)), pdist, color='C0', alpha=0.3, zorder=0)

    mean_dist = scinteg.trapz(pdist*dist_range, dist_range)
    axes[1].axvline(mean_dist, color='C0', linestyle='--', lw=1.5, zorder=6, label=r'$\langle D \rangle = {:.0f}$ Mpc'.format(mean_dist))

    ind0_40 = dist_range <= 40.
    ind40_80 = (dist_range <= 100.) & (dist_range > 40.)
    ind80_160 = (dist_range <= 160.) & (dist_range > 100.)
    p0_40 = scinteg.trapz(pdist[ind0_40], dist_range[ind0_40])
    p40_80 = scinteg.trapz(pdist[ind40_80], dist_range[ind40_80])
    p80_160 = scinteg.trapz(pdist[ind80_160], dist_range[ind80_160])
    print(p0_40*5, p40_80*5, p80_160*5)

    kde = spstat.gaussian_kde(dist_detect3, bw_method='scott')
    pdist = kde(dist_range)
    axes[1].plot(dist_range, pdist*norm, color='C1', linestyle='-', lw=3, zorder=2)
    patch2 = axes[1].fill_between(dist_range, np.zeros(len(dist_range)), pdist*norm, color='C1', alpha=0.5, zorder=1)

    h_range = np.arange(13, 22.5, 0.1)
    kde = spstat.gaussian_kde(hmag_detect2, bw_method='scott')
    ph = kde(h_range)
    axes[2].plot(h_range, ph, color='C0', linestyle='-', lw=3, zorder=4)
    axes[2].fill_between(h_range, np.zeros(len(h_range)), ph, color='C0', alpha=0.3, zorder=0)
    mean_h = scinteg.trapz(ph*h_range, h_range)
    axes[2].axvline(mean_h, color='C0', linestyle='--', lw=1.5, zorder=6, label=r'$\langle H \rangle = {:.0f}$ mag'.format(mean_h))

    kde = spstat.gaussian_kde(hmag_detect3, bw_method='scott')
    ph = kde(h_range)
    axes[2].plot(h_range, ph*norm, color='C1', linestyle='-', lw=3, zorder=2)
    axes[2].fill_between(h_range, np.zeros(len(h_range)), ph*norm, color='C1', alpha=0.5, zorder=1)
    mean_h = scinteg.trapz(ph*h_range, h_range)
    axes[2].axvline(mean_h, color='C1', linestyle='--', lw=1.5, zorder=6, label=r'$\langle H \rangle = {:.0f}$ mag'.format(mean_h))
    axes[2].legend(frameon=False, fontsize='small')


    mean_dist = scinteg.trapz(pdist*dist_range, dist_range)
    axes[1].axvline(mean_dist, ymax=0.75, color='C1', linestyle='--', lw=1.5, zorder=6, label=r'$\langle D \rangle = {:.0f}$ Mpc'.format(mean_dist))
    axes[1].legend(frameon=False, fontsize='small')




    axes[1].set_xlabel('Distance ($D$, Mpc)', fontsize='large')
    axes[1].set_ylabel('$P(D)$', fontsize='large')

    axes[0].set_xlabel('Number of Events ($N$)', fontsize='large')
    axes[0].set_ylabel('$P(N)$', fontsize='large')

    axes[2].set_xlabel('Apparent F475W ($g$, AB mag)', fontsize='large')
    axes[2].set_ylabel('$P(H)$', fontsize='large')
    axes[0].set_xlim(0, ebins.max())

    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(0, ymax)
    ymin, ymax = axes[2].get_ylim()
    axes[2].set_ylim(0, ymax)

    fig.legend((patch1, patch2), ('2 Detector Events', '3 Detector Events'), 'upper center', frameon=False, ncol=2, fontsize='medium')
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=1.05)
    fig.savefig('gw_detect.pdf')
    plt.show(fig)


if __name__=='__main__':
    argv = sys.argv[1:]
    sys.exit(main(argv=argv))
