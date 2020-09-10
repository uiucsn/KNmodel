#!/usr/bin/env python
import sys
import os
import numpy as np
import scipy.stats as spstat
from collections import namedtuple
from astropy.time import Time
import astropy.units as u
import argparse
import matplotlib.pyplot as plt
from astropy.visualization import hist


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
    parser.add_argument('--ligo_horizon', default=120., action=MinZeroAction, type=float,\
            help='Specify the horizon distance for BNS events from LIGO')
    parser.add_argument('--virgo_horizon', default=65., action=MinZeroAction, type=float,\
            help='Specify the horizon distance for Virgo events from LIGO')
    parser.add_argument('--box_size', default=500., action=MinZeroAction, type=float,\
            help='Specify the side of the box in which to simulate events')
    parser.add_argument('--mean_lograte', default=-5.95, help='specify the lograthim of the mean BNS rate')
    parser.add_argument('--sig_lograte',  default=0.55, help='specify the std of the mean BNS rate')
    parser.add_argument('--chirp_scale',  default=2.66, action=MinZeroAction, type=float, help='Set the chirp scale')
    parser.add_argument('--hdutycycle', default=0.7, action=MinZeroAction, type=float, help='Set the Hanford duty cycle')
    parser.add_argument('--ldutycycle', default=0.7, action=MinZeroAction, type=float, help='Set the Livingston duty cycle')
    parser.add_argument('--vdutycycle', default=0.7, action=MinZeroAction, type=float, help='Set the Virgo duty cycle')
    parser.add_argument('--ntry', default=10000, type=int, action=MinZeroAction, help='Set the number of MC samples')
    args = parser.parse_args(args=argv)

    if args.box_size < args.ligo_horizon or args.box_size < args.virgo_horizon:
        args.box_size = 4.*max(args.ligo_horizon, args.virgo_horizon)
    return args


def main(argv=None):

    args = get_options(argv=argv)
    np.random.seed(seed=42)

    # setup time-ranges
    ligo_run_start = Time('2019-02-01T00:00:00.0')
    ligo_run_end   = Time('2020-02-01T00:00:00.0')
    hst_cyc_start  = Time('2018-10-01T00:00:00.0')
    hst_cyc_end    = Time('2019-09-30T00:00:00.0')
    eng_time       = 2.*u.week
    Range = namedtuple('Range', ['start', 'end'])
    ligo_run  = Range(start=ligo_run_start, end=ligo_run_end)
    hst_cycle = Range(start=hst_cyc_start,  end=hst_cyc_end)
    latest_start = max(ligo_run.start, hst_cycle.start)
    earliest_end = min(ligo_run.end, hst_cycle.end)
    td = (earliest_end - latest_start) + eng_time
    fractional_duration = (td/(1.*u.year)).decompose().value

    # setup horizons
    bns_ligo_horizon  = args.ligo_horizon*u.megaparsec
    bns_virgo_horizon =  args.virgo_horizon*u.megaparsec
    # generate a bunch of events in a box of fixed size
    n_events = int(1E4)
    box_size = args.box_size
    x = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec
    y = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec
    z = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec
    dist = (x**2. + y**2. + z**2.)**0.5

    # create the mass distribution of the merging neutron star
    mass_distrib = args.mass_distrib
    if mass_distrib == 'mw':
        # the truncated normal distribution looks to be from:
        # https://arxiv.org/pdf/1309.6635.pdf
        mean_mass = args.masskey1
        sig_mass  = args.masskey2
        mass1 = spstat.truncnorm.rvs(0, np.inf, mean_mass, sig_mass, n_events)
        mass2 = spstat.truncnorm.rvs(0, np.inf, mean_mass, sig_mass, n_events)
    else:
        min_mass = args.masskey1
        max_mass = args.masskey2
        mass1 = np.random.uniform(min_mass, max_mass, n_events)
        mass2 = np.random.uniform(min_mass, max_mass, n_events)
    tot_mass = mass1 + mass2

    # setup duty cycles
    h_duty = args.hdutycycle
    l_duty = args.ldutycycle
    v_duty = args.vdutycycle
    h_on = np.random.choice([False, True], size=n_events, p=[1.-h_duty, h_duty])
    l_on = np.random.choice([False, True], size=n_events, p=[1.-l_duty, l_duty])
    v_on = np.random.choice([False, True], size=n_events, p=[1.-v_duty, v_duty])

    # setup event rates
    mean_lograte = args.mean_lograte
    sig_lograte  = args.sig_lograte
    rate = 10.**(np.random.normal(mean_lograte, sig_lograte, size=n_events))
    rate_full_volume = np.around(rate*(box_size**3.)*fractional_duration).astype('int_')

    chirp_scale  = args.chirp_scale
    n_try = int(n_events/10.)
    n_detect2 = []
    n_detect3 = []
    dist_detect = np.zeros((n_try, n_events)) -1
    for i in range(n_try):
        index = (np.random.uniform(size=rate_full_volume[i])*n_events).astype('int_')
        dist_ligo_bool  = dist[index] < bns_ligo_horizon*tot_mass[index]/chirp_scale
        dist_virgo_bool = dist[index] < bns_virgo_horizon*tot_mass[index]/chirp_scale

        # pretty sure Ryan meant l_on here and not v_on twice but should check
        two_det_bool = (h_on[index] & l_on[index]) | (v_on[index] & (h_on[index] | l_on[index]))
        three_det_bool = (h_on[index] & l_on[index] | v_on[index])

        n2_good = np.where(dist_ligo_bool & two_det_bool)[0]
        n2 = len(n2_good)
        n3_good = np.where(dist_virgo_bool & three_det_bool)[0]
        n3 = len(n3_good)

        n_detect2.append(n2)
        n_detect3.append(n3)

        if n2 > 0:
            dist_detect[i, n2_good] = dist[index][n2_good]

    n_detect2 = np.array(n_detect2)
    n_detect3 = np.array(n_detect3)

    fig_kw = {'figsize':(15, 5)}
    fig, axes = plt.subplots(nrows=1, ncols=3, **fig_kw)
    out_dist = dist_detect.ravel()
    hist(out_dist[out_dist > 0], bins='scott', ax=axes[0], density=True)
    hist(n_detect2, bins='scott', ax=axes[1], density=True)
    hist(n_detect3, bins='scott', ax=axes[2], density=True)
    axes[0].set_xlabel('Distance (Mpc)')
    axes[1].set_xlabel('N 2 Detector')
    axes[2].set_xlabel('N 3 Detector')

    print('n2 > 3', len(n_detect2[n_detect2>3])/n_try)
    print('n2 > 5', len(n_detect2[n_detect2>5])/n_try)
    print('n2 > 20', len(n_detect2[n_detect2>20])/n_try)
    fig.savefig('gw_detect.pdf')
    plt.show(fig)


if __name__=='__main__':
    argv = sys.argv[1:]
    sys.exit(main(argv=argv))
