#!/usr/bin/env python
import io
import pickle
import sys
import os
import json
import random
import time
import pandas as pd

from functools import partial
from urllib import parse, request
from tqdm import tqdm
from itertools import repeat

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
from multiprocessing import Pool, freeze_support
from scipy.linalg import cholesky
import scipy.integrate as scinteg
from sklearn.preprocessing import MinMaxScaler

from sed_to_lc import SEDDerviedLC
from dns_mass_distribution import Galaudage21, Farrow19
from dns_mass_distribution import MIN_MASS, MAX_MASS, M_TOV, EOS_interpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_phi
from rates_models import LVK_UG

import inspiral_range
import ligo.em_bright
import ligo.em_bright.computeDiskMass
from ligo.em_bright.computeDiskMass import computeCompactness, computeDiskMass
import lalsimulation as lalsim
from gwemlightcurves.EjectaFits import DiUj2017, CoDi2019
from kilopop.kilonovae import bns_kilonovae_population_distribution as s22p
from kilopop.kilonovae import bns_kilonova as saeev

#np.random.seed(seed=42)
disable_tqdm = True

# asd from https://emfollow.docs.ligo.org/userguide/capabilities.html
detector_asd_links_O4 = dict(
    ligo='https://dcc.ligo.org/public/0180/T2200043/003/aligo_O4high.txt',
    virgo='https://dcc.ligo.org/public/0180/T2200043/003/avirgo_O4high_NEW.txt',
    kagra='https://dcc.ligo.org/public/0180/T2200043/003/kagra_10Mpc.txt'
)

# asd from https://emfollow.docs.ligo.org/userguide/capabilities.html
detector_asd_links_O5 = dict(
    ligo='https://dcc.ligo.org/public/0180/T2200043/003/AplusDesign.txt',
    virgo='https://dcc.ligo.org/public/0180/T2200043/003/avirgo_O5low_NEW.txt',
    kagra='https://dcc.ligo.org/public/0180/T2200043/003/kagra_128Mpc.txt'
)

def compute_dyn_ej(m1, c1, m2, c2):

    a = -0.0719
    b = 0.2116
    d = -2.42
    n = -2.905

    mej_dyn = np.power(
        10.0,
        (
            ((a * (1.0 - 2.0 * c1) * m1) / (c1))
            + b * m2 * np.power((m1 / m2), n)
            + (d / 2.0)
        )
        + (
            ((a * (1.0 - 2.0 * c2) * m2) / (c2))
            + b * m1 * np.power((m2 / m1), n)
            + (d / 2.0)
        ),
    )

    # Imposing a maximum value for the dynamical ejecta
    max_mej_dyn = 0.09
    mej_dyn[mej_dyn > max_mej_dyn] = max_mej_dyn

    return mej_dyn


def compute_wind_ej(m1, m2, zetas):

    a = -31.335
    b = -0.9760
    c = 1.0474
    d = 0.05957

    # make sure these okay
    M_radius_1_dot_6 = EOS_interpolator(1.6)

    M_thresh = (2.38 - (3.606 * (M_TOV / M_radius_1_dot_6))) * M_TOV

    remnant_disk_mass = np.power(10.0, 
                                 a * (1.0 + b * np.tanh(
                                 (c - ((m1 + m2) /
                                  M_thresh)) / d)))

    remnant_disk_mass[remnant_disk_mass < 1.0e-3] = 1.0e-3
    mej_wind = zetas * remnant_disk_mass
    return mej_wind

def compute_compactness(m):

    G = 6.6743 * 10**-11 # m3 kg-1 s-2
    c = 3 * 10**8       # m s^-1
    M_sun = 1.9891 * 10**30 # kg

    R = EOS_interpolator(m) * 1000 # from km to m

    compactness = (G * m * M_sun) / (c**2 * R)

    return compactness

def get_ejecta_mass(m1, m2):

    # Different EOS
    c_ns_1 = compute_compactness(m1)
    c_ns_2 = compute_compactness(m2)

    n_events = len(m1)
    zetas = np.random.uniform(low=0.1, high=0.4, size=n_events)

    # treat as BNS
    m_dyn = compute_dyn_ej(m1, c_ns_1, m2, c_ns_2)
    m_wind = compute_wind_ej(m1, m2, zetas)

    # Check for prompt collapse to a BH
    M_radius_1_dot_6 = EOS_interpolator(1.6)
    M_thresh = (2.38 - (3.606 * (M_TOV / M_radius_1_dot_6))) * M_TOV

    m_total = m1 + m2
    m_dyn = np.where(m_total < M_thresh, m_dyn, 0)
    m_wind = np.where(m_total < M_thresh, m_wind, 0)

    return m_dyn, m_wind


def get_range(detector, ligo_run):
    if ligo_run == 'O4':
        psd_url = detector_asd_links_O4[detector]
    elif ligo_run == 'O5':
        psd_url = detector_asd_links_O5[detector]
    try:
        # if downloaded locally
        asd_fp = open(os.path.basename(parse.urlparse(psd_url).path), "rb")
    except FileNotFoundError:
        print(f"Downloading PSD for {detector}")
        asd_fp = io.BytesIO(request.urlopen(psd_url).read())
    freq, asd = np.loadtxt(asd_fp, unpack=True)
    psd = asd**2
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html#ggab955e4603c588fe19b39e47870a7b69cac65622993fd7f475a0ad423f35992906
    return partial(inspiral_range.range, freq, psd, approximant="TaylorF2")


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
    parser.add_argument('--output_csv', default='critical_point.csv', help='Output CSV file')
    #parser.add_argument('--ligo_run', choices=['O2','O3','O4','O5','hst32'], default='O4', help='Pick LIGO observing run')
    parser.add_argument('--mass_distrib', choices=['Galaudage21','Farrow19','flat'], default='Galaudage21', help='Pick BNS mass distribution')
    parser.add_argument('--rate_model', choices=['LVK_UG_O4'], default='LVK_UG_O4', help='Pick BNS merger rate model')
    parser.add_argument('--ntry', default=500, type=int, action=MinZeroAction, help='Set the number of MC samples')
    parser.add_argument('--detection_passband', default='desr', help='Pick detection passband. Should be from https://sncosmo.readthedocs.io/en/stable/bandpass-list.html')
    parser.add_argument('--detection_threshold', default=23, type=float, help='Pick detection threshold in detection passband.')
    parser.add_argument('--sun_loss', default=0.5, help='The fraction not observed due to sun', type=float)
    parser.add_argument('--bns_ligo_range', default= 150, help = 'Set the bns detection range for the two ligo detectors')
    parser.add_argument('--bns_virgo_range', default= 70, help = 'Set the bns detection range for the virgo detectors')
    parser.add_argument('--bns_kagra_range', default= 5, help = 'Set the bns detection range for the kagra detectors')

    # TODO: Add argument for overlap between survey and ligo run
    # duty factor motivation: https://dcc.ligo.org/public/0167/G2000497/002/G2000497_OpenLVEM_02Apr2020_kk_v2.pdf
    args = parser.parse_args(args=argv)
    return args

def run_trial(ligo_observing_run, 
              x, y, z,                                      # Coordinates
              mass1, mass2, omegas, dist,                   # Inputs for GW
              cos_thetas, phis, mej_dyn_arr, mej_wind_arr,  # Inputs for SED model
              detection_band, detection_threshold, av,         # EM counterpart threshold
              tot_ejecta_masses, 
              upper_chol, 
              n_events, n_years,
              args):

    if ligo_observing_run == 'O4':
        print("Configuring parameters for O4...")

        # setup duty cycles
        h_duty = 0.7
        l_duty = 0.7
        v_duty = 0.47
        k_duty = 0.27

    elif ligo_observing_run == 'O5':
        print("Configuring parameters for O5...")

        # setup duty cycles
        h_duty = 0.7
        l_duty = 0.7
        v_duty = 0.7
        k_duty = 0.7

    # define ranges
    ligo_range = get_range('ligo', ligo_observing_run)
    virgo_range = get_range('virgo', ligo_observing_run)
    kagra_range = get_range('kagra', ligo_observing_run)

    bns_range_ligo = np.array([])
    bns_range_virgo = np.array([])
    bns_range_kagra = np.array([])

    n1_bool = np.array(([False] * n_events))
    n2_bool = np.array(([False] * n_events))
    n3_bool = np.array(([False] * n_events))
    n4_bool = np.array(([False] * n_events))

    gw1_bool = np.array(([False] * n_events))
    gw2_bool = np.array(([False] * n_events))
    gw3_bool = np.array(([False] * n_events))
    gw4_bool = np.array(([False] * n_events))

    for m1, m2, o in tqdm(zip(mass1, mass2, omegas), total=n_events, disable=disable_tqdm):

        bns_range_ligo = np.append(bns_range_ligo, [ligo_range(m1=m1, m2=m2, inclination = o)])
        bns_range_virgo = np.append(bns_range_virgo,[virgo_range(m1=m1, m2=m2, inclination = o)])
        bns_range_kagra = np.append(bns_range_kagra, [kagra_range(m1=m1, m2=m2, inclination = o)])

    bns_range_ligo = bns_range_ligo*u.Mpc
    bns_range_virgo = bns_range_virgo*u.Mpc
    bns_range_kagra = bns_range_kagra*u.Mpc


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

    one_det_obs = n_detectors_on_and_obs == 1 
    two_det_obs = n_detectors_on_and_obs == 2
    three_det_obs = n_detectors_on_and_obs == 3
    four_det_obs = n_detectors_on_and_obs == 4

    # decide whether there is a kilonova based on remnant matter
    has_ejecta_bool = tot_ejecta_masses > 0

    # arrays to store peaks mags in different pbs
    peak_u = []
    peak_g = []
    peak_r = []
    peak_i = []
    peak_z = []
    peak_y = []

    em_bool = np.array([], dtype=bool)
    discovery_mags = np.array([])
    discovery_phases = np.array([])
    peak_mags = np.array([])
    discovery_windows = np.array([])

    ra_arr = np.array([])
    dec_arr = np.array([])
    d_Mpc = np.array([])

    scaling_factors = np.array([])
            

    for i, (cos_theta, phi, mej_dyn, mej_wind, d) in tqdm(enumerate(zip(cos_thetas, phis, mej_dyn_arr, mej_wind_arr, dist)), total=n_events, disable=disable_tqdm):

        #print(f"Sample parameters: mass1 = {mass1[i]}, mass2 = {mass2[i]}, cos_theta = {cos_theta}, phi = {phi}, ejecta_mass_dyn = {mej_dyn}, ejecta_mass_wind = {mej_wind}, dist = {d}")

        r, dec, ra = coord.cartesian_to_spherical(x[i], y[i], z[i])

        ra_arr = np.append(ra_arr, [ra.value])
        dec_arr = np.append(dec_arr, [dec.value])
        d_Mpc = np.append(d_Mpc, [d.value])

        coordinates = coord.SkyCoord(ra=ra, dec=dec)

        p = np.arange(0.3, 20.1, 0.2) 

        obj = SEDDerviedLC(mej_dyn = mej_dyn, mej_wind = mej_wind, phi = phi, cos_theta = cos_theta, dist=d, coord=coordinates, av =av[i])
        lcs = obj.getAppMagsInPassbands([detection_band], lc_phases=p)

        # Add peaks to data
        peaks = obj.getPeakAppMagsInPassbands(['lsstu','lsstg','lsstr','lssti','lsstz','lssty'],lc_phases=p)
        peak_u.append(peaks['lsstu'])
        peak_g.append(peaks['lsstg'])
        peak_r.append(peaks['lsstr'])
        peak_i.append(peaks['lssti'])
        peak_z.append(peaks['lsstz'])
        peak_y.append(peaks['lssty'])

        scaling_factors = np.append(scaling_factors, [obj.scaling_factor])

        min_mag = min(lcs[detection_band])

        # Minimum magnitude is the peak value
        peak_mags = np.append(peak_mags, [min_mag])
        
        idx = lcs[detection_band] < detection_threshold
        
        if min_mag < detection_threshold:
            em_bool = np.append(em_bool, [True])
            discovery_mag = (lcs[detection_band][idx])[0]
            discovery_phase = (p[idx])[0]
            discovery_mags = np.append(discovery_mags, [discovery_mag])
            discovery_phases = np.append(discovery_phases, [discovery_phase])
            discovery_window = (p[idx])[-1] - (p[idx])[0] + 0.2
            discovery_windows = np.append(discovery_windows, [discovery_window])
        else:
            em_bool = np.append(em_bool, [False])
            discovery_mags = np.append(discovery_mags, [np.nan])
            discovery_phases = np.append(discovery_phases, [np.nan])
            discovery_windows = np.append(discovery_windows, [np.nan])

    # whether this event was not affected by then sun
    detected_events = np.where(em_bool)
    sun_bool = np.random.random(len(detected_events[0])) >= args.sun_loss
    em_bool[detected_events] = sun_bool

    n1_gw_only = np.where(one_det_obs)[0]
    n1_gw = len(n1_gw_only)
    gw1_bool[n1_gw_only] = True
    n1_good = np.where(one_det_obs & em_bool & has_ejecta_bool)[0]
    n1 = len(n1_good)
    n1_bool[n1_good] = True
    # sanity check
    assert n1_gw >= n1, "GW events ({}) less than EM follow events ({})".format(n1_gw, n1)

    n2_gw_only = np.where(two_det_obs)[0]
    n2_gw = len(n2_gw_only)
    gw2_bool[n2_gw_only] = True
    n2_good = np.where(two_det_obs & em_bool & has_ejecta_bool)[0]
    n2 = len(n2_good)
    n2_bool[n2_good] = True
    # sanity check
    assert n2_gw >= n2, "GW events ({}) less than EM follow events ({})".format(n2_gw, n2)

    n3_gw_only = np.where(three_det_obs)[0]
    n3_gw = len(n3_gw_only)
    gw3_bool[n3_gw_only] = True
    n3_good = np.where(three_det_obs & em_bool & has_ejecta_bool)[0]
    n3 = len(n3_good)
    n3_bool[n3_good] = True
    # sanity check
    assert n3_gw >= n3, "GW events ({}) less than EM follow events ({})".format(n3_gw, n3)

    n4_gw_only = np.where(four_det_obs)[0]
    n4_gw = len(n4_gw_only)
    gw4_bool[n4_gw_only] = True
    n4_good = np.where(four_det_obs & em_bool & has_ejecta_bool)[0]
    n4 = len(n4_good)
    n4_bool[n4_good] = True
    # sanity check
    assert n4_gw >= n4, "GW events ({}) less than EM follow events ({})".format(n4_gw, n4)

    # 2 or more incident GW detection
    gw_indices = np.where(gw2_bool | gw3_bool | gw4_bool)[0]

    # 2 or more incident GW detection + detectable counterpart
    kn_indices = np.where(n2_bool | n3_bool | n4_bool)[0]

    # Return the time to the first detection
    print(gw_indices, kn_indices)


    if len(gw_indices) == 0:
        # Flag for no GW events
        first_gw_day = -1
    else:
        first_gw_day = gw_indices[0] / n_events * (365 * n_years)


    if len(kn_indices) == 0:
        # Flag for no KN events
        first_kn_day = -1
    else:
        first_kn_day = kn_indices[0] / n_events * (365 * n_years)

    return first_gw_day, first_kn_day


def get_earliest_detection_time(rate, args):

    n_years = 5

    box_size = 910

    # Simulate for 10 years
    Range = namedtuple('Range', ['start', 'end'])
    ligo_run_start = Time('2020-10-1T00:00:00.0')
    ligo_run_end   = Time(f'202{n_years}-10-1T00:00:00.0')
    survey_cyc_start = Time('2020-10-1T00:00:00.0')
    survey_cyc_end = Time(f'202{n_years}-10-1T00:00:00.0')
    eng_time       = 2.*u.week

    ligo_run  = Range(start=ligo_run_start, end=ligo_run_end)
    survey_cycle = Range(start=survey_cyc_start,  end=survey_cyc_end)
    latest_start = max(ligo_run_start, survey_cyc_start)
    earliest_end = min(ligo_run_end, survey_cyc_end)
    td = (earliest_end - latest_start) + eng_time
    fractional_duration = (td/(1.*u.year)).decompose().value

    # Simulation volume
    volume = box_size**3

    # the two ligo detectors ahve strongly correlated duty cycles
    # they are both not very correlated with Virgo
    lvc_cor_matrix = np.array([[1., 0.56, 0.56, 0.56],
                               [0.56, 1., 0.58, 0.58],
                               [0.56, 0.58, 1., 0.56],
                               [0.56, 0.58, 0.56, 1.]])
    upper_chol = cholesky(lvc_cor_matrix)

    # create the mass distribution of the merging neutron star
    mass_distrib = args.mass_distrib

    # Photometric detections
    detection_band = args.detection_passband
    detection_threshold = args.detection_threshold

    n_events = np.around(rate*volume*fractional_duration*(10**-9)).astype('int')
    n_events = max(n_events,1)
    print("N events:", n_events, flush=True)
    cos_thetas = np.random.uniform(0, 1, size=n_events)
    phis = np.random.uniform(15, 75, size=n_events)

    thetas = np.rad2deg(np.arccos(cos_thetas))
    omegas = np.minimum(thetas, 180 - thetas)

    if mass_distrib == 'Galaudage21':

        mass1, mass2 = Galaudage21(n_events)
        mej_dyn_arr, mej_wind_arr = get_ejecta_mass(mass1, mass2)

    elif mass_distrib == 'Farrow19':

        mass1, mass2 = Farrow19(n_events)
        mej_dyn_arr, mej_wind_arr = get_ejecta_mass(mass1, mass2)


    tot_mass = mass1 + mass2
    tot_ejecta_masses = mej_dyn_arr + mej_wind_arr

    av = np.random.exponential(0.334, n_events)*0.334

    # simulate coordinates. Additional term ensures minimum distance of 0.05 Mpc
    x = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec 
    y = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec 
    z = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec 
    dist = (x**2. + y**2. + z**2.)**0.5 + (0.05 * u.megaparsec)

    ##############################

    # Simulate for O4
    O4_first_gw_day, O4_first_kn_day = run_trial("O4", 
              x, y, z,                                      # Coordinates
              mass1, mass2, omegas, dist,                   # Inputs for GW
              cos_thetas, phis, mej_dyn_arr, mej_wind_arr,  # Inputs for SED model
              detection_band, detection_threshold, av,         # EM counterpart threshold
              tot_ejecta_masses, 
              upper_chol, 
              n_events, n_years,
              args)
    
    # Simulate for O5
    O5_first_gw_day, O5_first_kn_day = run_trial("O5", 
              x, y, z,                                      # Coordinates
              mass1, mass2, omegas, dist,                   # Inputs for GW
              cos_thetas, phis, mej_dyn_arr, mej_wind_arr,  # Inputs for SED model
              detection_band, detection_threshold, av,      # EM counterpart threshold
              tot_ejecta_masses, 
              upper_chol, 
              n_events, n_years,
              args)
    
    return rate, O4_first_gw_day, O4_first_kn_day, O5_first_gw_day, O5_first_kn_day



def main(argv=None):

    args = get_options(argv=argv)

    n_trials = args.ntry

    # Sample the BNS merger rate distribution for each trial
    rates_array = [LVK_UG(1)[0][0] for _ in range(n_trials)]


    with Pool(30) as pool:
        values = list(pool.starmap(get_earliest_detection_time, zip(rates_array, repeat(args))))


    df = pd.DataFrame(values, columns=['Rates','O4 Days to GW', 'O4 Days to KN', 'O5 Days to GW', 'O5 Days to KN'])
    df.to_csv(args.output_csv)
    print(df)

if __name__=='__main__':
    argv = sys.argv[1:]
    sys.exit(main(argv=argv))
