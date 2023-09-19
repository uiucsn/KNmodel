import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import numpy as np
import pandas as pd
import re
import io
from urllib import parse, request

from kilopop.kilonovae import bns_kilonova as saeev
from kilopop.kilonovae import bns_kilonovae_population_distribution as s22p
from tqdm import tqdm

from interpolate_bulla_sed import BullaSEDInterpolator
from scipy.interpolate import SmoothBivariateSpline
from dns_mass_distribution import extra_galactic_masses, galactic_masses
from monte_carlo_sims import get_ejecta_mass, get_range, MAX_MASS, MIN_MASS
from sed_to_lc import mej_dyn_grid_high, mej_dyn_grid_low, mej_wind_grid_high, mej_wind_grid_low

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import Distance, SkyCoord
from astropy.io import ascii
from matplotlib import cm

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_mej_dyn, uniq_mej_wind, phases, lmbd, uniq_cos_theta, uniq_phi
from sed_to_lc import SEDDerviedLC, lsst_bands

np.random.seed(seed=42)


def makeScalingLawsPlot():

    i = BullaSEDInterpolator(from_source=False)
    i.computeFluxScalingLaws(plot=True)

def makeDnsMassHistograms():

    n = 100000

    m1_exg, m2_exg = extra_galactic_masses(n)
    m1_mw, m2_mw = galactic_masses(n)

    bins = np.arange(MIN_MASS, MAX_MASS, 0.05)

    plt.hist(m1_exg, histtype=u'step', label=r'Exg $m_{recycled}$', linewidth=3, density=True, bins=bins)
    plt.hist(m2_exg, histtype=u'step', label=r'Exg $m_{slow}$', linewidth=3,  density=True, bins=bins)


    plt.hist(m1_mw, histtype=u'step', label=r'MW $m_{1}$', linewidth=3, linestyle='dashed',  density=True, bins=bins)
    plt.hist(m2_mw, histtype=u'step', label=r'MW $m_{2}$', linewidth=3, linestyle='dashed',  density=True, bins=bins)
    
    plt.legend()
    plt.xlabel(r"$\mathrm{M_{sun}}$", fontsize='x-large')
    plt.ylabel("Relative count", fontsize='x-large')

    plt.tight_layout()
    plt.savefig(f'paper_figures/dns_mass_dist.pdf')

def makeTrialsEjectaScatter():
     
    df = pd.read_csv('O5-LSST-r-24mag-Nitz/trials_df.csv')

    mej_wind = df['mej_wind']
    mej_dyn = df['mej_dyn']

    plt.scatter(mej_wind, mej_dyn, marker='.')
    plt.xlabel(r'$\mathrm{m_{ej}^{wind}}$')
    plt.ylabel(r'$\mathrm{m_{ej}^{dyn}}$')
    plt.axvspan(mej_wind_grid_low,  mej_wind_grid_high, alpha=0.25, color='orange')
    plt.axhspan(mej_dyn_grid_low, mej_dyn_grid_high, alpha=0.25, color='orange')
    plt.tight_layout()
    plt.savefig('paper_figures/mej_scatter.pdf')

def makeTrialsEjectaHistogram():
     
    df = pd.read_csv('O4-DECam-r-23mag-Abbott/trials_df.csv')

    mej_wind = df['mej_wind']
    mej_dyn = df['mej_dyn']

    sns.kdeplot(x=mej_wind, y=mej_dyn, cmap="Reds", fill=True, bw_adjust=1.5, x_limit=(0, ), y_limit=(0,))


    plt.xlabel(r'$\mathrm{m_{ej}^{wind}}$', fontsize='x-large')
    plt.ylabel(r'$\mathrm{m_{ej}^{dyn}}$', fontsize='x-large')


    plt.axvline(mej_wind_grid_low, color='black', linestyle='dotted')
    plt.axvline(mej_wind_grid_high, color='black', linestyle='dotted')
    plt.axhline(mej_dyn_grid_low, color='black', linestyle='dotted')
    plt.axhline(mej_dyn_grid_high, color='black', linestyle='dotted')
    
    plt.savefig('paper_figures/mej_scatter_hist.pdf')
    plt.tight_layout()

    plt.show()


def makeTrialsAvPlot():

    df = pd.read_csv('O4-DECam-r-23mag-Nitz/trials_df.csv')

    av = df['a_v']
    bins = np.arange(0, np.max(av), step=0.01)

    plt.hist(av, density=True, bins=bins,  histtype=u'step', linewidth=3)
    plt.xlabel(r'$A_V$', fontsize='x-large')
    plt.ylabel('Density', fontsize='x-large')

    plt.tight_layout()
    plt.savefig('paper_figures/av_hist.pdf')

def makeInterceptSurface():

    df_linear = pd.read_csv("data/m3_linear_scaling_laws.csv")

    phi = df_linear['phi']
    cos_theta = df_linear['cos_theta']
    c = df_linear['intercept']

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Surface for intercept (c) values", fontsize='x-large')
    #fig.set_size_inches(12.75, 10)

    ax1 = fig.add_subplot(projection='3d')

    ax1.plot(phi, cos_theta, c, 'ro')

    func = SmoothBivariateSpline(phi, cos_theta, c)
    print('c', func.get_residual())

    x_grid = np.arange(start=14.5, stop=76, step=0.1)
    y_grid = np.arange(start=-0.05, stop=1.1, step=0.01)
    z_grid = func(x_grid, y_grid).T

    xx, yy = np.meshgrid(x_grid, y_grid)

    ax1.plot_surface(xx, yy, z_grid, cmap=cm.plasma,rstride=1, cstride=1,  edgecolor='none')

    ax1.set_xlabel(r'$\Phi$', fontsize='x-large')
    ax1.set_ylabel(r'cos $\Theta$', fontsize='x-large')
    #ax1.set_zlabel('c', fontsize='x-large', labelpad=15)

    fig.savefig('paper_figures/c_surface.pdf', bbox_inches='tight')



def makeSlopeSurface():

    df_linear = pd.read_csv("data/m3_linear_scaling_laws.csv")

    phi = df_linear['phi']
    cos_theta = df_linear['cos_theta']
    m = df_linear['slope']

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Surface for slope (m) values", fontsize='x-large')
    #fig.set_size_inches(12.75, 10)
    
    ax1 = fig.add_subplot(projection='3d')

    ax1.plot(phi, cos_theta, m, 'ro')

    func = SmoothBivariateSpline(phi, cos_theta, m)
    print('m', func.get_residual())

    x_grid = np.arange(start=14.5, stop=76, step=0.1)
    y_grid = np.arange(start=-0.05, stop=1.1, step=0.01)
    z_grid = func(x_grid, y_grid).T

    xx, yy = np.meshgrid(x_grid, y_grid)

    ax1.plot_surface(xx, yy, z_grid, cmap=cm.plasma,rstride=1, cstride=1,  edgecolor='none')

    ax1.set_xlabel(r'$\Phi$', fontsize='x-large')
    ax1.set_ylabel(r'cos $\Theta$', fontsize='x-large')
    #ax1.set_zlabel('m', fontsize='x-large', labelpad=15)

    fig.savefig('paper_figures/m_surface.pdf', bbox_inches='tight')

def makeExponentSurface():

    df_linear = pd.read_csv("data/m3_power_scaling_laws.csv")

    phi = df_linear['phi']
    cos_theta = df_linear['cos_theta']
    n = df_linear['exponent']

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Surface for exponent (n) values", fontsize='x-large')
    #fig.set_size_inches(12.75, 10)

    ax1 = fig.add_subplot(projection='3d')

    ax1.plot(phi, cos_theta, n, 'ro')

    func = SmoothBivariateSpline(phi, cos_theta, n)
    print('n', func.get_residual())

    x_grid = np.arange(start=14.5, stop=76, step=0.1)
    y_grid = np.arange(start=-0.05, stop=1.1, step=0.01)
    z_grid = func(x_grid, y_grid).T

    xx, yy = np.meshgrid(x_grid, y_grid)

    ax1.plot_surface(xx, yy, z_grid, cmap=cm.plasma,rstride=1, cstride=1,  edgecolor='none')

    ax1.set_xlabel(r'$\Phi$', fontsize='x-large')
    ax1.set_ylabel(r'cos $\Theta$', fontsize='x-large')
    ax1.zaxis.set_major_formatter('{x:2<2.3f}')
    #ax1.set_zlabel('n', fontsize='x-large', labelpad=15)

    fig.savefig('paper_figures/n_surface.pdf', bbox_inches='tight')

def makeGW170817PhotometryPlotVillar():


    # Pass band stuff
    bands = ['g','r','i']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
    mej_wind = 10**-2.27
    mej_dyn = 10**-1.28
    phi = 49.5
    cos_theta = 0.7337298645

    # coordinates for GW170817
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 43*u.Mpc

    av = 0.0

    # LC from sed
    GW170817 = SEDDerviedLC(mej_dyn=mej_dyn, mej_wind = mej_wind, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
    lcs = GW170817.getAppMagsInPassbands(lsst_bands)
    

    from astropy.io import ascii
    data = ascii.read('data/Villar_170817.txt') 
    data = data[data['Used'] == "*"]

    data_g = data[data['Band']=='g']
    data_r = data[data['Band']=='r']
    data_i = data[data['Band']=='i']

    plt.errorbar(data_g['Phase'].data, np.array(data_g['Mag'].data, dtype=float) + 2, yerr=np.array(data_g['e_Mag'].data, dtype=float) , label='g + 2',c=colors[0], fmt='o')
    plt.errorbar(data_r['Phase'].data, np.array(data_r['Mag'].data, dtype=float), yerr=np.array(data_r['e_Mag'].data, dtype=float), label='r',c=colors[1], fmt='o')
    plt.errorbar(data_i['Phase'].data, np.array(data_i['Mag'].data, dtype=float) - 2, yerr=np.array(data_i['e_Mag'].data, dtype=float), label='i - 2',c=colors[2], fmt='o')

    plt.plot(phases[:60], lcs[f'lsstg'][:60] + 2, label = f'lsstg + 2', c=colors[0])
    plt.plot(phases[:60], lcs[f'lsstr'][:60], label = f'lsstr', c=colors[1])
    plt.plot(phases[:60], lcs[f'lssti'][:60] - 2, label = f'lssti - 2', c=colors[2])

    plt.xlabel('Phase (Days)', fontsize='x-large')
    plt.ylabel('Apparent Mag', fontsize='x-large')

    #plt.axhline(y=24, label = "Rubin 10s exposure", linestyle='dotted', color='red')

    plt.xlim(left= 0, right=7)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(linestyle="--")

    plt.tight_layout()
    plt.savefig(f'paper_figures/GW170817LC_villar.pdf')


def makeBNSRangePlot():

    ligo_range_O4 = get_range('ligo', 'O4')
    virgo_range_O4 = get_range('virgo', 'O4')
    kagra_range_O4 = get_range('kagra', 'O4')

    ligo_range_O5 = get_range('ligo', 'O5')
    virgo_range_O5 = get_range('virgo', 'O5')
    kagra_range_O5 = get_range('kagra', 'O5')

    print(f'Min O4 \nLIGO {ligo_range_O4(m1=1,m2=1, inclination=10)}\nVirgo {virgo_range_O4(m1=1,m2=1)}\nKagra {kagra_range_O4(m1=1,m2=1)}')
    print(f'Max O4 \nLIGO {ligo_range_O4(m1=2.05,m2=2.05, inclination=10)}\nVirgo {virgo_range_O4(m1=2.05,m2=2.05)}\nKagra {kagra_range_O4(m1=2.05,m2=2.05)}')

    print(f'Min O5 \nLIGO {ligo_range_O5(m1=1,m2=1)}\nVirgo {virgo_range_O5(m1=1,m2=1)}\nKagra {kagra_range_O5(m1=1,m2=1)}')
    print(f'Max O5 \nLIGO {ligo_range_O5(m1=2.05,m2=2.05)}\nVirgo {virgo_range_O5(m1=2.05,m2=2.05)}\nKagra {kagra_range_O5(m1=2.05,m2=2.05)}')

    masses = np.arange(1, 2.06, 0.1)

    lo4_array = np.zeros((len(masses), len(masses)))
    vo4_array = np.zeros((len(masses), len(masses)))
    ko4_array = np.zeros((len(masses), len(masses))) 

    lo5_array = np.zeros((len(masses), len(masses)))
    vo5_array = np.zeros((len(masses), len(masses)))
    ko5_array = np.zeros((len(masses), len(masses))) 

    # for i, m1 in enumerate(masses):
    #     for j, m2 in enumerate(masses):

    #         lo4_array[i][j] = ligo_range_O4(m1 = m1, m2 = m2)
    #         vo4_array[i][j] = virgo_range_O4(m1 = m1, m2 = m2)
    #         ko4_array[i][j] = kagra_range_O4(m1 = m1, m2 = m2)

    #         lo5_array[i][j] = ligo_range_O5(m1 = m1, m2 = m2)
    #         vo5_array[i][j] = virgo_range_O5(m1 = m1, m2 = m2)
    #         ko5_array[i][j] = kagra_range_O5(m1 = m1, m2 = m2)

    # cmap=cm.get_cmap('plasma')
    # max_range_O4 = max([np.max(lo4_array), np.max(vo4_array), np.max(ko4_array)])
    # min_range_O4 = min([np.min(lo4_array), np.min(vo4_array), np.min(ko4_array)])

    # h, w = 12, 5.5

    # fig, axes = plt.subplots(nrows=3, ncols=1)
    # fig.set_size_inches(w, h)
    # normalizer=Normalize(0, max_range_O4)
    # im=cm.ScalarMappable(norm=normalizer, cmap=cmap)

    # axes[0].imshow(lo4_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower",norm=normalizer, cmap=cmap)
    # axes[0].set_xlabel(r"$m_{1} (M_{\odot})$", fontsize='x-large')
    # axes[0].set_ylabel("LIGO Range\n\n" + r"$m_{2} (M_{\odot})$", fontsize='x-large')

    # axes[1].imshow(vo4_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower",norm=normalizer, cmap=cmap)
    # axes[1].set_xlabel(r"$m_{1} (M_{\odot})$", fontsize='x-large')
    # axes[1].set_ylabel("Virgo Range\n\n" + r"$m_{2} (M_{\odot})$", fontsize='x-large')

    # axes[2].imshow(ko4_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower",norm=normalizer, cmap=cmap)
    # axes[2].set_xlabel(r"$m_{1} (M_{\odot})$", fontsize='x-large')
    # axes[2].set_ylabel("Virgo Range\n\n" + r"$m_{2} (M_{\odot})$", fontsize='x-large')

    # cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    # cbar.set_label('Horizon distances (MPc)', fontsize='x-large')

    # plt.savefig('paper_figures/O4_range.pdf')
    # plt.show()

    # max_range_O5 = max([np.max(lo5_array), np.max(vo5_array), np.max(ko5_array)])
    # min_range_O5 = min([np.min(lo5_array), np.min(vo5_array), np.min(ko5_array)])

    # fig, axes = plt.subplots(nrows=3, ncols=1)
    # fig.set_size_inches(w, h)
    # normalizer=Normalize(0, max_range_O5)
    # im=cm.ScalarMappable(norm=normalizer,  cmap=cmap)

    # axes[0].imshow(lo5_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower",norm=normalizer, cmap=cmap)
    # axes[0].set_xlabel(r"$m_{1} (M_{\odot})$", fontsize='x-large')
    # axes[0].set_ylabel("LIGO Range\n\n" + r"$m_{2} (M_{\odot})$", fontsize='x-large')

    # axes[1].imshow(vo5_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower",norm=normalizer,  cmap=cmap)
    # axes[1].set_xlabel(r"$m_{1} (M_{\odot})$", fontsize='x-large')
    # axes[1].set_ylabel("Virgo Range\n\n" + r"$m_{2} (M_{\odot})$", fontsize='x-large')


    # axes[2].imshow(ko5_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower",norm=normalizer,  cmap=cmap)
    # axes[2].set_xlabel(r"$m_{1} (M_{\odot})$", fontsize='x-large')
    # axes[2].set_ylabel("KAGRA Range\n\n" + r"$m_{2} (M_{\odot})$", fontsize='x-large')

    # cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    # cbar.set_label('Horizon distances (MPc)', fontsize='x-large')

    # plt.savefig('paper_figures/O5_range.pdf')
    # plt.show()


if __name__ == '__main__':

    #makeScalingLawsPlot()
    #makeDnsMassHistograms()
    #makeTrialsEjectaScatter()
    #makeTrialsEjectaHistogram()
    makeTrialsAvPlot()
    #makeInterceptSurface()
    #makeSlopeSurface()
    #makeExponentSurface()
    #makeGW170817PhotometryPlotVillar()
    #makeBNSRangePlot()



