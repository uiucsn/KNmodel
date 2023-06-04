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
from astropy.coordinates import Distance, SkyCoord
import astropy.coordinates as coord
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
from gwemlightcurves.EjectaFits import DiUj2017, KrFo2019, CoDi2019
from kilopop.kilonovae import bns_kilonovae_population_distribution as s22p

np.random.RandomState(int(time.time()))

EOSNAME = "APR4_EPP"
MAX_MASS = 2.21  # specific to EoS model

detector_asd_links = dict(
    ligo='https://dcc.ligo.org/public/0165/T2000012/001/aligo_O4high.txt',
    virgo='https://dcc.ligo.org/public/0165/T2000012/001/avirgo_O4high_NEW.txt',
    kagra='https://dcc.ligo.org/public/0165/T2000012/001/kagra_80Mpc.txt'
)

def get_ejecta_mass(m1, m2):
    """Calculate ejecta mass of any remnant matter based on
    Dietrich & Ujevic (2017) or Foucart et. al. (2018) based on APR4
    equation of state.
    """
    m1, m2 = max(m1, m2), min(m1, m2)
    print(m1, m2)
    c_ns_1, m_b_1, _ = computeCompactness(m1, EOSNAME)
    c_ns_2, m_b_2, _ = computeCompactness(m2, EOSNAME)
    if m_b_2 == 0.0 or m_b_1 == 0.0:
        # treat as NSBH
        m_rem = computeDiskMass(m1, m2, 0., 0., eosname=EOSNAME)
    else:
        # treat as BNS
        m_rem = CoDi2019.calc_meje(np.array([m1]), np.array([c_ns_1]), np.array([m2]), np.array([c_ns_2]))[0]
        
    return m_rem

def get_range(detector):
    psd_url = detector_asd_links[detector]
    try:
        # if downloaded locally
        asd_fp = open(os.path.basename(parse.urlparse(psd_url).path), "rb")
    except FileNotFoundError:
        print(f"Downloading PSD for {detector}")
        asd_fp = io.BytesIO(request.urlopen(psd_url).read())
    freq, asd = np.loadtxt(asd_fp, unpack=True)
    psd = asd**2
    return partial(inspiral_range.range, freq, psd)


def get_correlated_series(n_events, upper_chol):
    """
    Get some correlated uniformly distributed random series between 0 and 1
    """
    rnd = np.random.uniform(0., 1., size=(n_events, 4))
    series = rnd @ upper_chol
    return series


def get_sim_dutycycles(n_events, upper_chol, h_duty, l_duty, v_duty, k_duty):
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
    k_series = series[3,:]

    h_on = duty_cycles[0,:]
    l_on = duty_cycles[1,:]
    v_on = duty_cycles[2,:]
    k_on = duty_cycles[3,:]

    h_on[h_series <= h_duty] = 1
    l_on[l_series <= l_duty] = 1
    v_on[v_series <= v_duty] = 1
    k_on[k_series <= k_duty] = 1

    h_on = h_on.astype(bool)
    l_on = l_on.astype(bool)
    v_on = v_on.astype(bool)
    k_on = k_on.astype(bool)

    return h_on, l_on, v_on, k_on


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
    parser.add_argument('--mass_distrib', choices=['mw','flat', 'msp', 'exg'], default='mw', help='Picky BNS mass distribution')
    parser.add_argument('--masskey1', type=float, action=MinZeroAction, default=1.4, help='Specify  Mass Keyword 1 (mw = mean, flat=lower bound)')
    parser.add_argument('--masskey2', type=float, action=MinZeroAction, default=0.09, help='Specify  Mass Keyword 2 (mw = sigma, flat=upper bound)')
    # Ryan's original value was -5.95 - the update comes from Alexandra Corsi's conservative estimate
    # 4.7d-6*4./3.*!pi*(170.)^3.*0.75*0.7 --> ~50
    # 3.2d-7*4./3.*!pi*(120.)^3.*0.75*0.7 --> 1
    # BTW: conservative and reasonable choice is 1.54d-6*4./3.*!pi*(120.)^3.*0.75*0.7 --> 5-6 events (median)
    parser.add_argument('--ntry', default=100, type=int, action=MinZeroAction, help='Set the number of MC samples')
    parser.add_argument('--box_size', default=400., action=MinZeroAction, type=float,\
            help='Specify the side of the box in which to simulate events')
    parser.add_argument('--sun_loss', default=0.61, help='The fraction not observed due to sun', type=float)
    parser.add_argument('--mean_lograte', default=-6.49, help='specify the lograthim of the mean BNS rate', type=float)
    parser.add_argument('--sig_lograte',  default=0.5, type=float, help='specify the std of the mean BNS rate')
    parser.add_argument('--hdutycycle', default=0.8, action=MinZeroAction, type=float, help='Set the Hanford duty cycle')
    parser.add_argument('--ldutycycle', default=0.8, action=MinZeroAction, type=float, help='Set the Livingston duty cycle')
    parser.add_argument('--vdutycycle', default=0.75, action=MinZeroAction, type=float, help='Set the Virgo duty cycle')
    parser.add_argument('--kdutycycle', default=0.4, action=MinZeroAction, type=float, help='Set the Kagra duty cycle')
    # duty factor motivation: https://dcc.ligo.org/public/0167/G2000497/002/G2000497_OpenLVEM_02Apr2020_kk_v2.pdf
    args = parser.parse_args(args=argv)
    return args


def main(argv=None):

    args = get_options(argv=argv)
    np.random.seed(seed=42)

    # setup time-ranges
    ligo_run_start = Time('2023-05-24T00:00:00.0')
    ligo_run_end   = Time('2024-11-24T00:00:00.0')
    jwst_cyc_start  = Time('2023-07-01T00:00:00.0')
    jwst_cyc_end    = Time('2024-06-30T00:00:00.0')
    eng_time       = 2.*u.week
    Range = namedtuple('Range', ['start', 'end'])
    ligo_run  = Range(start=ligo_run_start, end=ligo_run_end)
    jwst_cycle = Range(start=jwst_cyc_start,  end=jwst_cyc_end)
    latest_start = max(ligo_run.start, jwst_cycle.start)
    earliest_end = min(ligo_run.end, jwst_cycle.end)
    td = (earliest_end - latest_start) + eng_time
    fractional_duration = (td/(1.*u.year)).decompose().value

    box_size = args.box_size
    volume = box_size**3
    # create the mass distribution of the merging neutron star
    mass_distrib = args.mass_distrib

    min_mass = args.masskey1
    max_mass = args.masskey2

    # the two ligo detectors ahve strongly correlated duty cycles
    # they are both not very correlated with Virgo
    lvc_cor_matrix = np.array([[1., 0.8, 0.5, 0.1],
                               [0.8, 1., 0.5, 0.1],
                               [0.5, 0.5, 1., 0.1],
                               [0.1, 0.1, 0.1, 1.]])
    upper_chol = cholesky(lvc_cor_matrix)

    # setup duty cycles
    h_duty = args.hdutycycle
    l_duty = args.ldutycycle
    v_duty = args.vdutycycle
    k_duty = args.kdutycycle

    # setup event rates
    mean_lograte = args.mean_lograte
    sig_lograte  = args.sig_lograte
    n_try = args.ntry

    # define ranges
    ligo_range = get_range('ligo')
    virgo_range = get_range('virgo')
    kagra_range = get_range('kagra')

    def dotry(n):
        rate = 10.**(np.random.normal(mean_lograte, sig_lograte))
        n_events = np.around(rate*volume*fractional_duration).astype('int')
        if n_events == 0:
                return [], [], [], [], [], [], [], [], [], 0, 0, 0  # FIXME: fix to prevent unpacking error
        print(f"### Num trial = {n}; Num events = {n_events}")
        if mass_distrib == 'mw':
            mass1, mass2 = galactic_masses(n_events)
            # max_m, min_m = np.maximum(mass1, mass2), np.minimum(mass1, mass2)
            # print(get_ejecta_mass(max_m,min_m))
            ejecta_masses = np.array([ get_ejecta_mass(m1, m2) for m1, m2 in zip(mass1, mass2)])
        elif mass_distrib == 'exg':
            mass1, mass2 = extra_galactic_masses(n_events)
            # max_m, min_m = np.maximum(mass1, mass2), np.minimum(mass1, mass2)
            # print(get_ejecta_mass(max_m,min_m))
            ejecta_masses = np.array([ get_ejecta_mass(m1, m2) for m1, m2 in zip(mass1, mass2)])
        elif mass_distrib == 'msp':
            print("MSP population chosen, overriding mean_mass and sig_mass if supplied.")
            # numbers from https://arxiv.org/pdf/1605.01665.pdf
            # two modes, choose a random one each time
            mean_mass, sig_mass = random.choice([(1.393, 0.064), (1.807, 0.177)])
            mass1 = spstat.truncnorm.rvs(0, np.inf, mean_mass, sig_mass, n_events)
            mass2 = spstat.truncnorm.rvs(0, np.inf, mean_mass, sig_mass, n_events)
        else:
            print("Flat population chosen.")
            stars = s22p(population_size=n_events)
            mass1 = np.array([stars.compute_lightcurve_properties_per_kilonova(i)['mass1'] for i in range(n_events)])
            mass2 = np.array([stars.compute_lightcurve_properties_per_kilonova(i)['mass2'] for i in range(n_events)])
            ejecta_masses = np.array([stars.compute_lightcurve_properties_per_kilonova(i)['total_ejecta_mass'] for i in range(n_events)])

        # plt.hist(np.log10(ejecta_masses), density=True)
        # plt.xlabel('log10 mej')
        # # plt.xscale('log')
        # plt.xticks([-3, -2, -1, 0])
        # plt.axvspan(xmin=-2, xmax=-1, color='r', alpha=0.5)
        # plt.show()

        bns_range_ligo = np.array(
            [ligo_range(m1=m1, m2=m2) for m1, m2 in zip(mass1, mass2)]
        ) * u.Mpc
        bns_range_virgo = np.array(
            [virgo_range(m1=m1, m2=m2) for m1, m2 in zip(mass1, mass2)]
        ) * u.Mpc
        bns_range_kagra = np.array(
            [kagra_range(m1=m1, m2=m2) for m1, m2 in zip(mass1, mass2)]
        ) * u.Mpc
        bns_range_ligo = 190*u.Mpc
        bns_range_virgo = 115*u.Mpc
        bns_range_kagra = 10*u.Mpc

        tot_mass = mass1 + mass2

        av = np.random.exponential(1, n_events)*0.4
        #TODO: we need the extinction in r and h bands
        ar = av*0.748

        default_value= [0,]

        # simulate coordinates. Additional term ensures minimum distance of 0.05 Mpc
        x = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec + (0.05/(3**0.5)) * u.megaparsec
        y = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec + (0.05/(3**0.5)) * u.megaparsec
        z = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec + (0.05/(3**0.5)) * u.megaparsec
        dist = (x**2. + y**2. + z**2.)**0.5


        h_on, l_on, v_on, k_on = get_sim_dutycycles(n_events, upper_chol,
                                                    h_duty, l_duty, v_duty, k_duty)
        n_detectors_on = np.array(
            [sum(_) for _ in np.vstack((h_on, l_on, v_on, k_on)).T]
        )
        # which detectors observed
        dist_ligo_bool  = dist <= bns_range_ligo
        dist_virgo_bool = dist <= bns_range_virgo
        dist_kagra_bool = dist <= bns_range_kagra

        h_on_and_observed = h_on * dist_ligo_bool
        l_on_and_observed = l_on * dist_ligo_bool
        v_on_and_observed = v_on * dist_virgo_bool
        k_on_and_observed = k_on * dist_kagra_bool

        n_detectors_on_and_obs = np.sum(np.vstack(
            (h_on_and_observed, l_on_and_observed, v_on_and_observed,
             k_on_and_observed)).T,
            axis=1
        )

        two_det_obs = n_detectors_on_and_obs == 2
        three_det_obs = n_detectors_on_and_obs == 3
        four_det_obs = n_detectors_on_and_obs == 4

        # decide whether there is a kilnova based on remnant matter
        has_ejecta_bool = ejecta_masses > 0


        # # Get the actual values of the ejecta mass
        # ejecta_masses = [
        #     get_ejecta_mass(m1, m2) for m1, m2 in zip(mass1, mass2)
        # ]

        count = len(ejecta_masses)
        exp_count = (ejecta_masses < 0.01).sum() + (ejecta_masses > 0.09).sum()
        print(f'{(exp_count/n_events) * 100} % of the SEDs were extrapolated...')

        # Get random values for phi and cos theta as parameters for the model
        uniq_cos_theta = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 11 counts
        uniq_phi = np.array([15, 30, 45, 60, 75])

        cos_thetas = np.random.choice(uniq_cos_theta, size=n_events)
        phis = np.random.choice(uniq_phi, size=n_events)

        em_bool = []
        obsmag = []


        for i, (cos_theta, phi, mej, d) in enumerate(zip(cos_thetas, phis, ejecta_masses, dist)):

            #print(f"Sample parameters: cos_theta = {cos_theta}, phi = {phi}, ejecta_masses = {mej}, dist = {d}")


            r, dec, ra = coord.cartesian_to_spherical(x[i], y[i], z[i])
            coordinates = coord.SkyCoord(ra=ra, dec=dec)

            obj = SEDDerviedLC(mej = mej, phi = phi, cos_theta = cos_theta, dist=d, coord=coordinates, av = av[i])
            lcs = obj.buildJwstNircamLC()

            min_mag = min(lcs['f200w'])
            
            idx = lcs['f200w'] < 23
            
            if min_mag < 23:
                em_bool.append(True)
                obsmag.append((lcs['f200w'][idx])[0])
            else:
                em_bool.append(False)
                obsmag.append(min_mag)

        em_bool = np.array(em_bool)
        obsmag = np.array(obsmag)


        # whether this event was not affected by then sun
        detected_events = np.where(em_bool)
        sun_bool = np.random.random(len(detected_events[0])) >= args.sun_loss
        em_bool[detected_events] = sun_bool

        # print('ndet_GW', n_detectors_on_and_obs)
        # print("EM bool: ", em_bool)
        # print("Obs mag: ", obsmag)

        n2_gw_only = np.where(two_det_obs)[0]
        n2_gw = len(n2_gw_only)
        n2_good = np.where(two_det_obs & em_bool & has_ejecta_bool)[0]
        n2 = len(n2_good)
        # sanity check
        assert n2_gw >= n2, "GW events ({}) less than EM follow events ({})".format(n2_gw, n2)
        n3_gw_only = np.where(three_det_obs)[0]
        n3_gw = len(n3_gw_only)
        n3_good = np.where(three_det_obs & em_bool & has_ejecta_bool)[0]
        n3 = len(n3_good)
        # sanity check
        assert n3_gw >= n3, "GW events ({}) less than EM follow events ({})".format(n3_gw, n3)
        n4_gw_only = np.where(four_det_obs)[0]
        n4_gw = len(n4_gw_only)
        n4_good = np.where(four_det_obs & em_bool & has_ejecta_bool)[0]
        n4 = len(n4_good)
        # sanity check
        assert n4_gw >= n4, "GW events ({}) less than EM follow events ({})".format(n4_gw, n4)
        return dist[n2_good].value.tolist(), tot_mass[n2_good].tolist(),\
            dist[n3_good].value.tolist(), tot_mass[n3_good].tolist(),\
            dist[n4_good].value.tolist(), tot_mass[n4_good].tolist(),\
            obsmag[n2_good].tolist(), obsmag[n3_good].tolist(),\
            obsmag[n3_good].tolist(),\
            n2, n3, n4

    with schwimmbad.SerialPool() as pool:
        values = list(pool.map(dotry, range(n_try)))

    with open(f'ntry-{n_try}-values.pkl', 'wb') as f:
        pickle.dump(values, f)
    print("Finshed computation, plotting...")
    data_dump = dict()
    n_detect2 = []
    n_detect3 = []
    n_detect4 = []
    dist_detect2 = []
    mass_detect2 = []
    dist_detect3 = []
    mass_detect3 = []
    dist_detect4 = []
    mass_detect4 = []
    mag_detect2 = []
    mag_detect3 = []
    mag_detect4 = []
    for idx, (d2, m2, d3, m3, d4, m4, h2, h3, h4, n2, n3, n4) in enumerate(values):
        if n2 >= 0:
            n_detect2.append(n2)
            if n3>0:
                dist_detect2 += d2
                mass_detect2 += m2
                mag_detect2  += h2
        if n3>=0:
            n_detect3.append(n3)
            if n3 > 0:
                dist_detect3 += d3
                mass_detect3 += m3
                mag_detect3  += h3
        if n4>=0:
            n_detect4.append(n4)
            if n4 > 0:
                dist_detect4 += d4
                mass_detect4 += m4
                mag_detect4  += h4
        data_dump[f"{idx}"] = {"d2": d2, "m2": m2, "d3": d3,
                               "m3": m3, "d4": d4, "m4": m4,
                               "h2": h2, "h3": h3, "h4": h4,
                               "n2": n2, "n3": n3, "n4": n4}
    with open(f"data-dump-{args.mass_distrib}.pickle", "wb") as f:
        pickle.dump(data_dump, f)

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
    axes[0].axvline(round(mean_nevents), color='C0', linestyle='--', lw=2,
                    label=r'$\langle N\rangle = %d ;~ N_{95} = %d$' % (round(mean_nevents), ninetyfive_percent))
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
    axes[0].axvline(round(mean_nevents), color='C1', linestyle='--', lw=2,
                label=r'$\langle N\rangle = %d ;~ N_{95} = %d$' % (round(mean_nevents), ninetyfive_percent))
    axes[0].axvline(ninetyfive_percent, color='C1',
                linestyle='dotted', lw=1)
#vals, bins = np.histogram(n_detect4, bins=ebins, density=True)
    mean_nevents = np.mean(n_detect4)
    #vals*=norm
    #test = dict(zip(ebins, vals))
    axes[0].hist(n_detect4, density=True, histtype='stepfilled', color='C2', alpha=0.5, bins=ebins, zorder=1)
    axes[0].hist(n_detect4, density=True, histtype='step', color='C2', lw=3, bins=ebins, zorder=2)
    five_percent, ninetyfive_percent = np.percentile(n_detect4, 5), np.percentile(n_detect4, 95)
    axes[0].axvline(round(mean_nevents), color='C2', linestyle='--', lw=2,
                    label=r'$\langle N \rangle = %d ;~ N_{95} = %d$' % (round(mean_nevents), ninetyfive_percent))
    axes[0].axvline(ninetyfive_percent, color='C2',
                    linestyle='dotted', lw=1)
    axes[0].legend(frameon=False, fontsize='small', loc='upper right')
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
    # save number of detections
    with open(f'n-events-{args.mass_distrib}.pickle', 'wb') as f:
        res = dict(n_detect2=n_detect2, n_detect3=n_detect3, n_detect4=n_detect4,
                   dist_detect2=dist_detect2, dist_detect3=dist_detect3, dist_detect4=dist_detect4,
                   mass_detect2=mass_detect2, mass_detect3=mass_detect3, mass_detect4=mass_detect4,
                   mag_detect2=mag_detect2, mag_detect3=mag_detect3, mag_detect4=mag_detect4)
        pickle.dump(res, f)
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
        axes[1].legend(frameon=False, fontsize='small')
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
        axes[1].legend(frameon=False, fontsize='small')
    except ValueError:
        print("Could not create KDE since no 4-det detection")

    h_range = np.arange(15, 23, 0.1)
    kde = spstat.gaussian_kde(mag_detect2, bw_method='scott')
    ph = kde(h_range)
    axes[2].plot(h_range, ph, color='C0', linestyle='-', lw=3, zorder=4)
    axes[2].fill_between(h_range, np.zeros(len(h_range)), ph, color='C0', alpha=0.3, zorder=0)
    mean_h = np.mean(mag_detect2)
    axes[2].axvline(mean_h, color='C0', linestyle='--', lw=1.5, zorder=6, label=r'$\langle r \rangle = {:.1f}$ mag'.format(mean_h))

    kde = spstat.gaussian_kde(mag_detect3, bw_method='scott')
    ph = kde(h_range)
    axes[2].plot(h_range, ph, color='C1', linestyle='-', lw=3, zorder=2)
    axes[2].fill_between(h_range, np.zeros(len(h_range)), ph, color='C1', alpha=0.5, zorder=1)
    mean_h = np.mean(mag_detect3)
    axes[2].axvline(mean_h, color='C1', linestyle='--', lw=1.5, zorder=6, label=r'$\langle r \rangle = {:.1f}$ mag'.format(mean_h))
    axes[2].legend(frameon=False, fontsize='small')

    try:
        kde = spstat.gaussian_kde(mag_detect4, bw_method='scott')
        ph = kde(h_range)
        axes[2].plot(h_range, ph, color='C2', linestyle='-', lw=3, zorder=2)
        axes[2].fill_between(h_range, np.zeros(len(h_range)), ph, color='C1', alpha=0.5, zorder=1)
        mean_h = np.mean(mag_detect4)
        axes[2].axvline(mean_h, color='C2', linestyle='--', lw=1.5, zorder=6, label=r'$\langle H \rangle = {:.1f}$ mag'.format(mean_h))
        axes[2].legend(frameon=False, fontsize='small')
    except ValueError:
        print("Could not create KDE for h-mag since no 4 detector events found")

    axes[1].set_xlabel('Distance ($D$, Mpc)', fontsize='large')
    axes[1].set_ylabel('$P(D)$', fontsize='large')
    #if args.mass_distrib != 'msp':
    #    axes[0].set_title(f"Masses {args.mass_distrib}; {args.masskey1} -- {args.masskey2}")
    #else:
    #    axes[0].set_title("MSP bimodal mass @ 1.393 / 1.807 $M_{\odot}$")
    axes[0].set_xlabel('Number of Events ($N$)', fontsize='large')
    axes[0].set_ylabel('$P(N)$', fontsize='large')

    axes[2].set_xlabel('Apparent $r$, AB mag)', fontsize='large')
    axes[2].set_ylabel('$P(r)$', fontsize='large')
    axes[0].set_xlim(0, ebins.max())

    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(0, ymax)
    ymin, ymax = axes[2].get_ylim()
    axes[2].set_ylim(0, ymax)

    fig.legend(patches, legend_text,
               'upper center', frameon=False, ncol=3, fontsize='medium')
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=1.05)
    fig.savefig(f'gw_detect_{args.mass_distrib}_JWST_Cycle2_LVKRun04.pdf')
    plt.show()


if __name__=='__main__':
    argv = sys.argv[1:]
    sys.exit(main(argv=argv))
