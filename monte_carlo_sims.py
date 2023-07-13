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
from interpolate_bulla_sed import uniq_cos_theta, uniq_phi

import inspiral_range
import ligo.em_bright
import ligo.em_bright.computeDiskMass
from ligo.em_bright.computeDiskMass import computeCompactness, computeDiskMass
import lalsimulation as lalsim
from gwemlightcurves.EjectaFits import DiUj2017, KrFo2019, CoDi2019
from kilopop.kilonovae import bns_kilonovae_population_distribution as s22p
from kilopop.kilonovae import bns_kilonova as saeev

np.random.RandomState(int(time.time()))

EOSNAME = "APR4_EPP"
MAX_MASS = 2.05  # specific to EoS model
MIN_MASS = 1

detector_asd_links = dict(
    ligo='https://dcc.ligo.org/public/0165/T2000012/001/aligo_O4high.txt',
    virgo='https://dcc.ligo.org/public/0165/T2000012/001/avirgo_O4high_NEW.txt',
    kagra='https://dcc.ligo.org/public/0165/T2000012/001/kagra_80Mpc.txt'
)
    
def get_ejecta_mass(m1, m2):

    merger = saeev(mass1=m1, mass2=m2)
    merger.map_to_kilonova_ejecta()
    return merger.param7, merger.param10

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
    parser.add_argument('--trials_dir', default='trial-mw-100', help='Directory to store simulation results')
    parser.add_argument('--mass_distrib', choices=['mw','flat', 'exg'], default='exg', help='Pick BNS mass distribution')
    parser.add_argument('--rate_model', choices=['Nitz23','Abbott23'], default='Abbott23', help='Pick BNS merger rate model')
    parser.add_argument('--ntry', default=100, type=int, action=MinZeroAction, help='Set the number of MC samples')
    parser.add_argument('--detection_passband', default='f200w', help='Pick detection passband. Should be from https://sncosmo.readthedocs.io/en/stable/bandpass-list.html')
    parser.add_argument('--detection_threshold', default=23, help='Pick detection threshold in detection passband.')
    parser.add_argument('--box_size', default=400., action=MinZeroAction, type=float, help='Specify the side of the box in which to simulate events')
    parser.add_argument('--sun_loss', default=0.61, help='The fraction not observed due to sun', type=float)
    parser.add_argument('--hdutycycle', default=0.8, action=MinZeroAction, type=float, help='Set the Hanford duty cycle')
    parser.add_argument('--ldutycycle', default=0.8, action=MinZeroAction, type=float, help='Set the Livingston duty cycle')
    parser.add_argument('--vdutycycle', default=0.75, action=MinZeroAction, type=float, help='Set the Virgo duty cycle')
    parser.add_argument('--kdutycycle', default=0.4, action=MinZeroAction, type=float, help='Set the Kagra duty cycle')
    parser.add_argument('--ligo_run_start', default='2023-05-24T00:00:00.0', help = 'Set the run start date for ligo')
    parser.add_argument('--ligo_run_end', default='2024-11-24T00:00:00.0', help = 'Set the run end date for ligo')
    parser.add_argument('--survey_start', default='2023-07-01T00:00:00.0', help = 'Set the start date for your survey of choice')
    parser.add_argument('--survey_end', default='2024-06-30T00:00:00.0', help = 'Set the end date for your survey of choice')
    parser.add_argument('--bns_ligo_range', default= 150, help = 'Set the bns detection range for the two ligo detectors')
    parser.add_argument('--bns_virgo_range', default= 70, help = 'Set the bns detection range for the virgo detectors')
    parser.add_argument('--bns_kagra_range', default= 0, help = 'Set the bns detection range for the kagra detectors')

    # TODO: Add argument for overlap between survey and ligo run
    # duty factor motivation: https://dcc.ligo.org/public/0167/G2000497/002/G2000497_OpenLVEM_02Apr2020_kk_v2.pdf
    args = parser.parse_args(args=argv)
    return args


def main(argv=None):

    args = get_options(argv=argv)
    np.random.seed(seed=42)

    # setup time-ranges
    ligo_run_start = Time(args.ligo_run_start)
    ligo_run_end   = Time(args.ligo_run_end)
    survey_cyc_start  = Time(args.survey_start)
    survey_cyc_end    = Time(args.survey_end)
    eng_time       = 2.*u.week

    Range = namedtuple('Range', ['start', 'end'])
    ligo_run  = Range(start=ligo_run_start, end=ligo_run_end)
    survey_cycle = Range(start=survey_cyc_start,  end=survey_cyc_end)
    latest_start = max(ligo_run.start, survey_cycle.start)
    earliest_end = min(ligo_run.end, survey_cycle.end)
    td = (earliest_end - latest_start) + eng_time
    fractional_duration = (td/(1.*u.year)).decompose().value

    box_size = args.box_size
    volume = box_size**3

    # the two ligo detectors ahve strongly correlated duty cycles
    # they are both not very correlated with Virgo
    lvc_cor_matrix = np.array([[1., 0.8, 0.5, 0.1],
                               [0.8, 1., 0.5, 0.1],
                               [0.5, 0.5, 1., 0.1],
                               [0.1, 0.1, 0.1, 1.]])
    upper_chol = cholesky(lvc_cor_matrix)

    # create the mass distribution of the merging neutron star
    mass_distrib = args.mass_distrib

    # setup duty cycles
    h_duty = args.hdutycycle
    l_duty = args.ldutycycle
    v_duty = args.vdutycycle
    k_duty = args.kdutycycle

    # setup event rates
    n_try = args.ntry
    rate_model = args.rate_model

    # Create the dir
    trials_dir = args.trials_dir
    os.mkdir(f"{trials_dir}")

    # Save args file
    with open(f'{trials_dir}/mc_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Photometric detections
    detection_band = args.detection_passband
    detection_threshold = args.detection_threshold

    # define ranges
    ligo_range = get_range('ligo')
    virgo_range = get_range('virgo')
    kagra_range = get_range('kagra')



    def dotry(n):
        
        trial_df = pd.DataFrame()

        if rate_model == "Nitz23":
            rate = np.random.uniform(52, 508)
        elif rate_model == "Abbott23":
            rate = np.random.uniform(10, 1700)


        n_events = np.around(rate*volume*fractional_duration*(10**-9)).astype('int')

        em_bool = np.array([], dtype=bool)
        discovery_mags = np.array([])
        peak_mags = np.array([])
    
        ra_arr = np.array([])
        dec_arr = np.array([])
        d_Mpc = np.array([])

        trial_number = np.array(([n] * n_events))
        n2_bool = np.array(([False] * n_events))
        n3_bool = np.array(([False] * n_events))
        n4_bool = np.array(([False] * n_events))

        scaling_factors = np.array([])

        print(f"### Num trial = {n}; Num events = {n_events}")
        if mass_distrib == 'mw':

            mass1, mass2 = galactic_masses(n_events)
            mej_dyn_arr = np.array([])
            mej_wind_arr = np.array([])

            for m1, m2 in zip(mass1, mass2):

                mej_dyn, mej_wind = get_ejecta_mass(m1, m2)
                mej_dyn_arr = np.append(mej_dyn_arr, [mej_dyn])
                mej_wind_arr = np.append(mej_wind_arr, [mej_wind])

        elif mass_distrib == 'exg':

            mass1, mass2 = extra_galactic_masses(n_events)
            mej_dyn_arr = np.array([])
            mej_wind_arr = np.array([])

            for m1, m2 in zip(mass1, mass2):

                mej_dyn, mej_wind = get_ejecta_mass(m1, m2)
                mej_dyn_arr = np.append(mej_dyn_arr, [mej_dyn])
                mej_wind_arr = np.append(mej_wind_arr, [mej_wind])

        elif mass_distrib == 'flat':

            stars = s22p(population_size=n_events)

            mass1 = np.array([stars.compute_lightcurve_properties_per_kilonova(i)['mass1'] for i in range(n_events)])
            mass2 = np.array([stars.compute_lightcurve_properties_per_kilonova(i)['mass2'] for i in range(n_events)])

            mej_dyn_arr = np.array([stars.compute_lightcurve_properties_per_kilonova(i)['dynamical_ejecta_mass'] for i in range(n_events)])
            mej_wind_arr = np.array([stars.compute_lightcurve_properties_per_kilonova(i)['secular_ejecta_mass'] for i in range(n_events)])

        bns_range_ligo = np.array(
            [ligo_range(m1=m1, m2=m2) for m1, m2 in zip(mass1, mass2)]
        ) * u.Mpc
        bns_range_virgo = np.array(
            [virgo_range(m1=m1, m2=m2) for m1, m2 in zip(mass1, mass2)]
        ) * u.Mpc
        bns_range_kagra = np.array(
            [kagra_range(m1=m1, m2=m2) for m1, m2 in zip(mass1, mass2)]
        ) * u.Mpc

        bns_range_ligo = args.bns_ligo_range * u.Mpc
        bns_range_virgo = args.bns_virgo_range * u.Mpc
        bns_range_kagra = args.bns_kagra_range * u.Mpc

        tot_mass = mass1 + mass2
        tot_ejecta_masses = mej_dyn_arr + mej_wind_arr

        av = np.random.exponential(0.334, n_events)*0.334

        # simulate coordinates. Additional term ensures minimum distance of 0.05 Mpc
        x = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec 
        y = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec 
        z = np.random.uniform(-box_size/2., box_size/2., n_events)*u.megaparsec 
        dist = (x**2. + y**2. + z**2.)**0.5 + (0.05 * u.megaparsec)


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

        # decide whether there is a kilonova based on remnant matter
        has_ejecta_bool = tot_ejecta_masses > 0

        cos_thetas = np.random.choice(uniq_cos_theta, size=n_events)
        phis = np.random.choice(uniq_phi, size=n_events)

        for i, (cos_theta, phi, mej_dyn, mej_wind, d) in tqdm(enumerate(zip(cos_thetas, phis, mej_dyn_arr, mej_wind_arr, dist)), total=n_events):

            #print(f"Sample parameters: mass1 = {mass1[i]}, mass2 = {mass2[i]}, cos_theta = {cos_theta}, phi = {phi}, ejecta_mass_dyn = {mej_dyn}, ejecta_mass_wind = {mej_wind}, dist = {d}")

            r, dec, ra = coord.cartesian_to_spherical(x[i], y[i], z[i])

            ra_arr = np.append(ra_arr, [ra.value])
            dec_arr = np.append(dec_arr, [dec.value])
            d_Mpc = np.append(d_Mpc, [d.value])

            coordinates = coord.SkyCoord(ra=ra, dec=dec)

            p = np.arange(0.3, 7.6, 0.2)

            obj = SEDDerviedLC(mej_dyn = mej_dyn, mej_wind = mej_wind, phi = phi, cos_theta = cos_theta, dist=d, coord=coordinates, av =av[i])
            lcs = obj.getAppMagsInPassbands([detection_band], lc_phases=p)


            scaling_factors = np.append(scaling_factors, [obj.scaling_factor])

            min_mag = min(lcs[detection_band])

            # Minimum magnitude is the peak value
            peak_mags = np.append(peak_mags, [min_mag])
            
            idx = lcs[detection_band] < detection_threshold
            
            if min_mag < detection_threshold:
                em_bool = np.append(em_bool, [True])
                discovery_mag = (lcs[detection_band][idx])[0]
                discovery_mags = np.append(discovery_mags, [discovery_mag])
            else:
                em_bool = np.append(em_bool, [False])
                discovery_mags = np.append(discovery_mags, [np.nan])
        plt.show()
        # whether this event was not affected by then sun
        detected_events = np.where(em_bool)
        sun_bool = np.random.random(len(detected_events[0])) >= args.sun_loss
        em_bool[detected_events] = sun_bool

        n2_gw_only = np.where(two_det_obs)[0]
        n2_gw = len(n2_gw_only)
        n2_good = np.where(two_det_obs & em_bool & has_ejecta_bool)[0]
        n2 = len(n2_good)
        n2_bool[n2_good] = True
        # sanity check
        assert n2_gw >= n2, "GW events ({}) less than EM follow events ({})".format(n2_gw, n2)

        n3_gw_only = np.where(three_det_obs)[0]
        n3_gw = len(n3_gw_only)
        n3_good = np.where(three_det_obs & em_bool & has_ejecta_bool)[0]
        n3 = len(n3_good)
        n3_bool[n3_good] = True
        # sanity check
        assert n3_gw >= n3, "GW events ({}) less than EM follow events ({})".format(n3_gw, n3)

        n4_gw_only = np.where(four_det_obs)[0]
        n4_gw = len(n4_gw_only)
        n4_good = np.where(four_det_obs & em_bool & has_ejecta_bool)[0]
        n4 = len(n4_good)
        n4_bool[n4_good] = True
        # sanity check
        assert n4_gw >= n4, "GW events ({}) less than EM follow events ({})".format(n4_gw, n4)

        # Create a data frame with all the information
        trial_df['trial_number'] = trial_number
        trial_df['m1'] = mass1
        trial_df['m2'] = mass2
        trial_df['total_mass'] = tot_mass
        trial_df['mej_dyn'] = mej_dyn_arr
        trial_df['mej_wind'] = mej_wind_arr
        trial_df['cos_theta'] = cos_thetas
        trial_df['phi'] = phis
        trial_df['a_v'] = av
        trial_df['dist'] = d_Mpc
        trial_df['ra'] = ra_arr
        trial_df['dec'] = dec_arr
        trial_df['n_detectors_on_and_obs'] = n_detectors_on_and_obs
        trial_df['em_bool'] = em_bool
        trial_df['peak_mag'] = peak_mags
        trial_df['discovery_mag'] = discovery_mags
        trial_df['scaling_factor'] = scaling_factors
        trial_df['two_detector_event'] = n2_bool
        trial_df['three_detector_event'] = n3_bool
        trial_df['four_detector_event'] = n4_bool

        print(f"Number of:\n2 detector events: {n2}\n3 detector events: {n3}\n4 detector events: {n4}")

        return dist[n2_good].value.tolist(), tot_mass[n2_good].tolist(),\
            dist[n3_good].value.tolist(), tot_mass[n3_good].tolist(),\
            dist[n4_good].value.tolist(), tot_mass[n4_good].tolist(),\
            discovery_mags[n2_good].tolist(), discovery_mags[n3_good].tolist(),\
            discovery_mags[n4_good].tolist(),\
            peak_mags[n2_good].tolist(), peak_mags[n3_good].tolist(),\
            peak_mags[n4_good].tolist(),\
            n2, n3, n4, trial_df

    with schwimmbad.SerialPool() as pool:
        values = list(pool.map(dotry, range(n_try)))

    with open(f'{trials_dir}/raw_mc_data.pkl', 'wb') as f:
        pickle.dump(values, f)
    print("Finished computation...")
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
    mag_peak2 = []
    mag_peak3 = []
    mag_peak4 = []
    df_list = []

    for idx, (d2, m2, d3, m3, d4, m4, h2, h3, h4, p2, p3, p4, n2, n3, n4, df) in enumerate(values):

        df_list.append(df)
        if n2 >= 0:
            n_detect2.append(n2)
            if n3>0:
                dist_detect2 += d2
                mass_detect2 += m2
                mag_detect2  += h2
                mag_peak2 += p2
        if n3>=0:
            n_detect3.append(n3)
            if n3 > 0:
                dist_detect3 += d3
                mass_detect3 += m3
                mag_detect3  += h3
                mag_peak3 += p3
        if n4>=0:
            n_detect4.append(n4)
            if n4 > 0:
                dist_detect4 += d4
                mass_detect4 += m4
                mag_detect4  += h4
                mag_peak4 += p4
        data_dump[f"{idx}"] = {"d2": d2, "m2": m2, "d3": d3,
                               "m3": m3, "d4": d4, "m4": m4,
                               "h2": h2, "h3": h3, "h4": h4,
                               "n2": n2, "n3": n3, "n4": n4,
                               "p2": p2, "p3": p3, "p4": p4,}
        
    with open(f"{trials_dir}/data_dump.pickle", "wb") as f:
        pickle.dump(data_dump, f)
    
    with open(f'{trials_dir}/plotting_data.pickle', 'wb') as f:
        res = dict(n_detect2=n_detect2, n_detect3=n_detect3, n_detect4=n_detect4,
                    dist_detect2=dist_detect2, dist_detect3=dist_detect3, dist_detect4=dist_detect4,
                    mass_detect2=mass_detect2, mass_detect3=mass_detect3, mass_detect4=mass_detect4,
                    mag_detect2=mag_detect2, mag_detect3=mag_detect3, mag_detect4=mag_detect4,
                    mag_peak2 =mag_peak2, mag_peak3=mag_peak3, mag_peak4=mag_peak4)
        pickle.dump(res, f)

    df_master = pd.concat(df_list, ignore_index=True)
    df_master.to_csv(f"{trials_dir}/trials_df.csv")


if __name__=='__main__':
    argv = sys.argv[1:]
    sys.exit(main(argv=argv))
