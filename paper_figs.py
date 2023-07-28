import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from kilopop.kilonovae import bns_kilonova as saeev
from kilopop.kilonovae import bns_kilonovae_population_distribution as s22p
from tqdm import tqdm

from interpolate_bulla_sed import BullaSEDInterpolator
from dns_mass_distribution import extra_galactic_masses, galactic_masses
from monte_carlo_sims import get_ejecta_mass
from sed_to_lc import mej_dyn_grid_high, mej_dyn_grid_low, mej_wind_grid_high, mej_wind_grid_low

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import Distance, SkyCoord
from astropy.io import ascii
from matplotlib import cm

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_mej_dyn, uniq_mej_wind, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands

np.random.seed(seed=42)

def makeLinearScalingLawsPlot():

    i2 = BullaSEDInterpolator(from_source=False)
    i2.computeFluxScalingLinearLaws(plot=True)
    i2.computeFluxScalingPowerLaws(plot=True)

def makeDnsMassHistograms():

    n = 1000

    fig, ax = plt.subplots(1, 2)

    m1_exg, m2_exg = extra_galactic_masses(n)
    m1_mw, m2_mw = galactic_masses(n)

    ax[0].hist(m1_exg, alpha= 0.5, histtype=u'step', label=r'$m_{recycled}$')
    ax[0].hist(m2_exg, alpha= 0.5, histtype=u'step', label=r'$m_{slow}$')
    ax[0].legend()

    ax[1].hist(m1_mw, alpha= 0.5, histtype=u'step', label=r'$m_{1}$')
    ax[1].hist(m2_mw, alpha= 0.5, histtype=u'step', label=r'$m_{2}$')
    ax[1].legend()

    fig.savefig(f'paper_figures/dns_mass_dist.pdf')

def makeMejEjectaPlot():

    n = 200

    mej_wind_arr_exg = np.array([])
    mej_dyn_arr_exg = np.array([])

    mej_wind_arr_mw = np.array([])
    mej_dyn_arr_mw = np.array([])

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)

    m1_exg, m2_exg = extra_galactic_masses(n)
    m1_mw, m2_mw = galactic_masses(n)

    for m1, m2 in tqdm(zip(m1_exg, m2_exg), total=n):

        mej_dyn, mej_wind = get_ejecta_mass(m1, m2)
        mej_dyn_arr_exg = np.append(mej_dyn_arr_exg, [mej_dyn])
        mej_wind_arr_exg = np.append(mej_wind_arr_exg, [mej_wind])

    for m1, m2 in tqdm(zip(m1_mw, m2_mw), total=n):

        mej_dyn, mej_wind = get_ejecta_mass(m1, m2)
        mej_dyn_arr_mw = np.append(mej_dyn_arr_mw, [mej_dyn])
        mej_wind_arr_mw = np.append(mej_wind_arr_mw, [mej_wind])
    
    mej_total_exg = mej_wind_arr_exg + mej_dyn_arr_exg
    mej_total_mw = mej_wind_arr_mw + mej_dyn_arr_mw

    log_mej_total_exg = np.log10(mej_total_exg)
    log_mej_total_mw = np.log10(mej_total_mw)

    scatter = ax[0][0].scatter(m1_exg, m2_exg, c=log_mej_total_exg)
    ax[0][0].set_xlabel(r'$M_{recycled}$')
    ax[0][0].set_ylabel(r'$M_{slow}$')

    ax[0][0].set_xlim(left=0.8, right=2.3)
    ax[0][0].set_ylim(bottom=0.8, top=2.3)
    
    ax[0][1].scatter(m1_mw, m2_mw, c=log_mej_total_mw)
    ax[0][1].set_xlabel(r'$M_1$')
    ax[0][1].set_ylabel(r'$M_2$')

    ax[0][1].set_xlim(left=0.8, right=2.3)
    ax[0][1].set_ylim(bottom=0.8, top=2.3)

    plt.colorbar(scatter, ax=ax.ravel().tolist())

    ax[1][0].hist(log_mej_total_exg, density=True)
    ax[1][0].set_xlabel(r'$m_{ej}$')
    ax[1][0].set_ylabel(r'$Count$')
    
    ax[1][1].hist(log_mej_total_mw, density=True)
    ax[1][1].set_xlabel(r'$m_{ej}$')
    ax[1][1].set_ylabel(r'$Count$')
    
    fig.savefig(f'paper_figures/mej_dist.pdf')

    plt.show()

def makeGW170817SedSurfacePlot():

        # Pass band stuff
        bands = ['g','r','i']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
        mej_wind = 0.05
        mej_dyn = 0.005
        phi = 30
        cos_theta = 0.9

        # coordinates for GW170817
        c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        d = 43*u.Mpc

        av = 0.0

        # LC from sed
        GW170817 = SEDDerviedLC(mej_dyn=mej_dyn, mej_wind = mej_wind, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
        lcs = GW170817.getAppMagsInPassbands(lsst_bands)
        GW170817.makeSedPlot()
        plt.savefig(f'paper_figures/GW170817SED.pdf')


def makeGW170817PhotometryPlot():

    # Pass band stuff
    bands = ['g','r','i']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
    mej_wind = 0.05
    mej_dyn = 0.005
    phi = 30
    cos_theta = 0.9

    # coordinates for GW170817
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 43*u.Mpc

    av = 0.0

    # LC from sed
    GW170817 = SEDDerviedLC(mej_dyn=mej_dyn, mej_wind = mej_wind, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
    lcs = GW170817.getAppMagsInPassbands(lsst_bands)
    
    # table from https://iopscience.iop.org/article/10.3847/2041-8213/aa8fc7#apjlaa8fc7t2
    data = pd.read_csv('data/gw170817photometry.csv', delimiter='\t' )  
    data['mag'] = [float(re.findall("\d+\.\d+", i)[0]) for i in data['Mag [AB]']]

    plt.scatter(data[data['Filter'] == 'g']['MJD'], data[data['Filter'] == 'g']['mag'] + 2, label='g + 2',c=colors[0])
    plt.scatter(data[data['Filter'] == 'r']['MJD'], data[data['Filter'] == 'r']['mag'], label='r',c=colors[1]) 
    plt.scatter(data[data['Filter'] == 'i']['MJD'], data[data['Filter'] == 'i']['mag'] - 2, label='i - 2', c=colors[2])

    plt.plot(phases, lcs[f'lsstg'] + 2, label = f'lsstg + 2', c=colors[0])
    plt.plot(phases, lcs[f'lsstr'], label = f'lsstr', c=colors[1])
    plt.plot(phases, lcs[f'lssti'] - 2, label = f'lssti - 2', c=colors[2])

    plt.xlabel('Phase')
    plt.ylabel('Apparent Mag')

    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(linestyle="--")

    plt.title(f'Interpolated Data: mej_total = {mej_dyn + mej_wind} phi = {phi} cos theta = {cos_theta}')
    plt.savefig(f'paper_figures/GW170817LC.pdf')

def makeTrialsEjectaScatter():
     
    df = pd.read_csv('500_exg_Abbott/trials_df.csv')

    mej_wind = df['mej_wind']
    mej_dyn = df['mej_dyn']

    plt.scatter(mej_wind, mej_dyn, marker='.')
    plt.xlabel('mej wind')
    plt.ylabel('mej dyn')
    plt.axvspan(mej_wind_grid_low,  mej_wind_grid_high, alpha=0.25, color='orange')
    plt.axhspan(mej_dyn_grid_low, mej_dyn_grid_high, alpha=0.25, color='orange')
    plt.savefig('paper_figures/mej_scatter.pdf')


def makeTrialsAvPlot():

    df = pd.read_csv('500_exg_Abbott/trials_df.csv')

    av = df['a_v']

    plt.hist(av, density=True, bins=50)
    plt.xlabel(r'$A_V$')
    plt.ylabel('Density')
    plt.savefig('paper_figures/av_hist.pdf')


if __name__ == '__main__':

    #makeLinearScalingLawsPlot()
    #makeDnsMassHistograms()
    #makeMejEjectaPlot()
    #makeGW170817PhotometryPlot()
    #makeGW170817SedSurfacePlot()
    #makeTrialsEjectaScatter()
    makeTrialsAvPlot()
