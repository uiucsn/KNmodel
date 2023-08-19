import matplotlib.pyplot as plt
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
from monte_carlo_sims import get_ejecta_mass, detector_asd_links_O4, detector_asd_links_O5, get_range
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

    n = 10000

    # fig, ax = plt.subplots(1, 2)
    # fig.set_size_inches(10, 5)

    m1_exg, m2_exg = extra_galactic_masses(n)
    m1_mw, m2_mw = galactic_masses(n)



    plt.hist(m1_exg, histtype=u'step', label=r'$exg m_{recycled}$', linewidth=3, density=True)
    plt.hist(m2_exg, histtype=u'step', label=r'$exg m_{slow}$', linewidth=3,  density=True)


    plt.hist(m1_mw, histtype=u'step', label=r'$mw m_{1}$', linewidth=3, linestyle='dashed',  density=True)
    plt.hist(m2_mw, histtype=u'step', label=r'$mw m_{2}$', linewidth=3, linestyle='dashed',  density=True)
    
    plt.legend()
    plt.xlabel(r"$\mathrm{M_{sun}}$", fontsize=12)
    plt.ylabel("Relative count", fontsize=12)

    plt.tight_layout()
    plt.savefig(f'paper_figures/dns_mass_dist.pdf')

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
    plt.tight_layout()
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
    plt.tight_layout()
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

    data = data[data['Instrument'] == 'DECam']

    data['mag'] = [float(re.findall("\d+\.\d+", i)[0]) for i in data['Mag [AB]']]

    plt.scatter(data[data['Filter'] == 'g']['MJD'], data[data['Filter'] == 'g']['mag'] + 2, label='g + 2',c=colors[0])
    plt.scatter(data[data['Filter'] == 'r']['MJD'], data[data['Filter'] == 'r']['mag'], label='r',c=colors[1]) 
    plt.scatter(data[data['Filter'] == 'i']['MJD'], data[data['Filter'] == 'i']['mag'] - 2, label='i - 2', c=colors[2])

    plt.plot(phases[:60], lcs[f'lsstg'][:60] + 2, label = f'lsstg + 2', c=colors[0])
    plt.plot(phases[:60], lcs[f'lsstr'][:60], label = f'lsstr', c=colors[1])
    plt.plot(phases[:60], lcs[f'lssti'][:60] - 2, label = f'lssti - 2', c=colors[2])

    plt.xlabel('Phase')
    plt.ylabel('Apparent Mag')

    plt.axhline(y=24, label = "LSST 10s exposure", linestyle='dotted', color='red')
    #plt.axhline(y=24, label = "LSST 10s exposure")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(linestyle="--")



    plt.title(f'Interpolated Data: mej_total = {mej_dyn + mej_wind} phi = {phi} cos theta = {cos_theta}')
    plt.tight_layout()
    plt.savefig(f'paper_figures/GW170817LC.pdf')

def makeTrialsEjectaScatter():
     
    df = pd.read_csv('500_exg_Abbott/trials_df.csv')

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
     
    df = pd.read_csv('500_exg_Abbott/trials_df.csv')

    mej_wind = df['mej_wind']
    mej_dyn = df['mej_dyn']

    sns.kdeplot(x=mej_wind, y=mej_dyn, cmap="Reds", fill=True, bw_adjust=1.5)


    plt.xlabel(r'$\mathrm{m_{ej}^{wind}}$')
    plt.ylabel(r'$\mathrm{m_{ej}^{dyn}}$')

    plt.axvline(mej_wind_grid_low, color='black', linestyle='dotted')
    plt.axvline(mej_wind_grid_high, color='black', linestyle='dotted')
    plt.axhline(mej_dyn_grid_low, color='black', linestyle='dotted')
    plt.axhline(mej_dyn_grid_high, color='black', linestyle='dotted')
    
    plt.savefig('paper_figures/mej_scatter_hist.pdf')

    plt.show()


def makeTrialsAvPlot():

    df = pd.read_csv('500_exg_Abbott/trials_df.csv')

    av = df['a_v']

    plt.hist(av, density=True, bins=50,  histtype=u'step')
    plt.xlabel(r'$A_V$')
    plt.ylabel('Density')
    plt.savefig('paper_figures/av_hist.pdf')

def makeInterceptSurface():

    df_linear = pd.read_csv("data/m3_linear_scaling_laws.csv")

    phi = df_linear['phi']
    cos_theta = df_linear['cos_theta']
    c = df_linear['intercept']

    fig = plt.figure()
    fig.suptitle("Surface for intercept values")

    ax1 = fig.add_subplot(projection='3d')

    ax1.plot(phi, cos_theta, c, 'ro')

    func = SmoothBivariateSpline(phi, cos_theta, c)
    print('c', func.get_residual())

    x_grid = np.arange(start=14.5, stop=76, step=0.1)
    y_grid = np.arange(start=-0.05, stop=1.1, step=0.01)
    z_grid = func(x_grid, y_grid).T

    xx, yy = np.meshgrid(x_grid, y_grid)

    ax1.plot_surface(xx, yy, z_grid, cmap=cm.plasma,rstride=1, cstride=1,  edgecolor='none')

    ax1.set_xlabel('Phi')
    ax1.set_ylabel('cos theta')
    ax1.set_zlabel('c')

    plt.tight_layout()
    fig.savefig('paper_figures/c_surface.pdf', bbox_inches='tight')



def makeSlopeSurface():

    df_linear = pd.read_csv("data/m3_linear_scaling_laws.csv")

    phi = df_linear['phi']
    cos_theta = df_linear['cos_theta']
    m = df_linear['slope']

    fig = plt.figure()
    fig.suptitle("Surface for slope values")
    ax1 = fig.add_subplot(projection='3d')

    ax1.plot(phi, cos_theta, m, 'ro')

    func = SmoothBivariateSpline(phi, cos_theta, m)
    print('m', func.get_residual())

    x_grid = np.arange(start=14.5, stop=76, step=0.1)
    y_grid = np.arange(start=-0.05, stop=1.1, step=0.01)
    z_grid = func(x_grid, y_grid).T

    xx, yy = np.meshgrid(x_grid, y_grid)

    ax1.plot_surface(xx, yy, z_grid, cmap=cm.plasma,rstride=1, cstride=1,  edgecolor='none')

    ax1.set_xlabel('Phi')
    ax1.set_ylabel('cos theta')
    ax1.set_zlabel('m')

    plt.tight_layout()
    fig.savefig('paper_figures/m_surface.pdf', bbox_inches='tight')

def makeExponentSurface():

    df_linear = pd.read_csv("data/m3_power_scaling_laws.csv")

    phi = df_linear['phi']
    cos_theta = df_linear['cos_theta']
    n = df_linear['exponent']

    fig = plt.figure()
    fig.suptitle("Surface for exponent values")

    ax1 = fig.add_subplot(projection='3d')

    ax1.plot(phi, cos_theta, n, 'ro')

    func = SmoothBivariateSpline(phi, cos_theta, n)
    print('n', func.get_residual())

    x_grid = np.arange(start=14.5, stop=76, step=0.1)
    y_grid = np.arange(start=-0.05, stop=1.1, step=0.01)
    z_grid = func(x_grid, y_grid).T

    xx, yy = np.meshgrid(x_grid, y_grid)

    ax1.plot_surface(xx, yy, z_grid, cmap=cm.plasma,rstride=1, cstride=1,  edgecolor='none')

    ax1.set_xlabel('Phi')
    ax1.set_ylabel('cos theta')
    ax1.set_zlabel('n')

    plt.tight_layout()
    fig.savefig('paper_figures/n_surface.pdf', bbox_inches='tight')

def makeBlueKnLc():
    # lanthanide free component - more wind ejecta. Bright and Blue due to the low opacity.
    # Pass band stuff
    bands = ['g','r','i']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
    mej_wind = 2 * mej_wind_grid_low
    mej_dyn = mej_dyn_grid_high
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


    plt.plot(phases[:60], lcs[f'lsstg'][:60] + 2, label = f'lsstg + 2', c=colors[0])
    plt.plot(phases[:60], lcs[f'lsstr'][:60], label = f'lsstr', c=colors[1])
    plt.plot(phases[:60], lcs[f'lssti'][:60] - 2, label = f'lssti - 2', c=colors[2])

    plt.xlabel('Phase')
    plt.ylabel('Apparent Mag')

    plt.axhline(y=24, label = "LSST 10s exposure", linestyle='dotted', color='red')
    #plt.axhline(y=24, label = "LSST 10s exposure")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(linestyle="--")



    plt.title(f'Interpolated Data: mej_total = {mej_dyn + mej_wind} phi = {phi} cos theta = {cos_theta}')
    plt.tight_layout()
    plt.savefig(f'paper_figures/BlueKN.pdf')

def makeRedKnLc():
    # lanthanide rich component - more dynamical ejecta. Faint and red due to the high opacity of the the lanthanides
    # Pass band stuff
    bands = ['g','r','i']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Best fit parameters for GW 170817 - https://iopscience.iop.org/article/10.3847/1538-4357/ab5799
    mej_wind = mej_wind_grid_high
    mej_dyn = 2 * mej_dyn_grid_low
    phi = 30
    cos_theta = 0.9

    # coordinates for GW170817
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 43*u.Mpc

    av = 0.0

    # LC from sed
    GW170817 = SEDDerviedLC(mej_dyn=mej_dyn, mej_wind = mej_wind, phi = phi, cos_theta = cos_theta, dist=d, coord=c, av = av)
    lcs = GW170817.getAppMagsInPassbands(lsst_bands)

    plt.plot(phases[:60], lcs[f'lsstg'][:60] + 2, label = f'lsstg + 2', c=colors[0])
    plt.plot(phases[:60], lcs[f'lsstr'][:60], label = f'lsstr', c=colors[1])
    plt.plot(phases[:60], lcs[f'lssti'][:60] - 2, label = f'lssti - 2', c=colors[2])

    plt.xlabel('Phase')
    plt.ylabel('Apparent Mag')

    plt.axhline(y=24, label = "LSST 10s exposure", linestyle='dotted', color='red')
    #plt.axhline(y=24, label = "LSST 10s exposure")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(linestyle="--")



    plt.title(f'Interpolated Data: mej_total = {mej_dyn + mej_wind} phi = {phi} cos theta = {cos_theta}')
    plt.tight_layout()
    plt.savefig(f'paper_figures/RedKN.pdf')

def makeGW170817PhotometryPlotVillar():


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
    

    from astropy.io import ascii
    data = ascii.read('data/Villar_170817.txt') 
    data = data[data['Used'] == "*"]

    data_g = data[data['Band']=='g']
    data_r = data[data['Band']=='r']
    data_i = data[data['Band']=='i']



    plt.errorbar(data_g['Phase'].data, np.array(data_g['Mag'].data, dtype=float) + 2, yerr=np.array(data_g['e_Mag'].data, dtype=float) , label='g + 2',c=colors[0], fmt='.')
    plt.errorbar(data_r['Phase'].data, np.array(data_r['Mag'].data, dtype=float), yerr=np.array(data_r['e_Mag'].data, dtype=float), label='r',c=colors[1], fmt='.')
    plt.errorbar(data_i['Phase'].data, np.array(data_i['Mag'].data, dtype=float) - 2, yerr=np.array(data_i['e_Mag'].data, dtype=float), label='i - 2',c=colors[2], fmt='.')

    plt.plot(phases[:60], lcs[f'lsstg'][:60] + 2, label = f'lsstg + 2', c=colors[0])
    plt.plot(phases[:60], lcs[f'lsstr'][:60], label = f'lsstr', c=colors[1])
    plt.plot(phases[:60], lcs[f'lssti'][:60] - 2, label = f'lssti - 2', c=colors[2])

    plt.xlabel('Phase')
    plt.ylabel('Apparent Mag')

    plt.axhline(y=24, label = "Rubin 10s exposure", linestyle='dotted', color='red')

    plt.xlim(left= 0, right=7)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(linestyle="--")



    plt.title(f'Interpolated Data: mej_total = {mej_dyn + mej_wind} phi = {phi} cos theta = {cos_theta}')
    plt.tight_layout()
    plt.savefig(f'paper_figures/GW170817LC_villar.pdf')

def makeO4PSD():

    for detector in detector_asd_links_O4:
        print(f"Downloading PSD for O4 {detector}")
        psd_url = detector_asd_links_O4[detector]
        asd_fp = io.BytesIO(request.urlopen(psd_url).read())
        freq, asd = np.loadtxt(asd_fp, unpack=True)
        psd = asd**2
        plt.plot(freq, psd, label=f"O4 {detector}")
    
    
    plt.xscale('log')
    plt.yscale('log')

    plt.title('LVK PSD for run O4')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Strain Noise')

    plt.xlim(10,10**4)

    plt.legend()

    plt.savefig('paper_figures/O4_psd.pdf')
    plt.show()

def makeO5PSD():

    for detector in detector_asd_links_O5:

        print(f"Downloading PSD for O5 {detector}")
        psd_url = detector_asd_links_O5[detector]
        asd_fp = io.BytesIO(request.urlopen(psd_url).read())
        freq, asd = np.loadtxt(asd_fp, unpack=True)
        psd = asd**2
        plt.plot(freq, psd, label=f"O5 {detector}")
    

    plt.xscale('log')
    plt.yscale('log')

    plt.title('LVK PSD for run O5')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Strain Noise')

    plt.xlim(10,10**4)

    plt.legend()

    plt.savefig('paper_figures/O5_psd.pdf')
    plt.show()

def makeBNSRangePlot():

    ligo_range_O4 = get_range('ligo', 'O4')
    virgo_range_O4 = get_range('virgo', 'O4')
    kagra_range_O4 = get_range('kagra', 'O4')

    ligo_range_O5 = get_range('ligo', 'O5')
    virgo_range_O5 = get_range('virgo', 'O5')
    kagra_range_O5 = get_range('kagra', 'O5')

    masses = np.arange(1, 2.06, 0.05)

    lo4_array = np.zeros((len(masses), len(masses)))
    vo4_array = np.zeros((len(masses), len(masses)))
    ko4_array = np.zeros((len(masses), len(masses))) 

    lo5_array = np.zeros((len(masses), len(masses)))
    vo5_array = np.zeros((len(masses), len(masses)))
    ko5_array = np.zeros((len(masses), len(masses))) 

    for i, m1 in enumerate(masses):
        for j, m2 in enumerate(masses):

            lo4_array[i][j] = ligo_range_O4(m1 = m1, m2 = m2)
            vo4_array[i][j] = virgo_range_O4(m1 = m1, m2 = m2)
            ko4_array[i][j] = kagra_range_O4(m1 = m1, m2 = m2)

            lo5_array[i][j] = ligo_range_O5(m1 = m1, m2 = m2)
            vo5_array[i][j] = virgo_range_O5(m1 = m1, m2 = m2)
            ko5_array[i][j] = kagra_range_O5(m1 = m1, m2 = m2)


    plt.imshow(lo4_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower")
    plt.xlabel(r"$m_{1}$")
    plt.ylabel(r"$m_{2}$")
    plt.title('LIGO range - O4')
    plt.colorbar(label = 'Distance (Mpc)')
    plt.tight_layout()
    plt.savefig('paper_figures/Ligo_O4_range.pdf')
    plt.show()

    plt.imshow(ko4_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower")
    plt.xlabel(r"$m_{1}$")
    plt.ylabel(r"$m_{2}$")
    plt.title('Kagra range - O4')
    plt.colorbar(label = 'Distance (Mpc)')
    plt.tight_layout()
    plt.savefig('paper_figures/Kagra_O4_range.pdf')
    plt.show()

    plt.imshow(vo4_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower")
    plt.xlabel(r"$m_{1}$")
    plt.ylabel(r"$m_{2}$")
    plt.title('Virgo range - O4')
    plt.colorbar(label = 'Distance (Mpc)')
    plt.tight_layout()
    plt.savefig('paper_figures/Virgo_O4_range.pdf')
    plt.show()

    plt.imshow(lo5_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower")
    plt.xlabel(r"$m_{1}$")
    plt.ylabel(r"$m_{2}$")
    plt.title('LIGO range - O5')
    plt.colorbar(label = 'Distance (Mpc)')
    plt.tight_layout()
    plt.savefig('paper_figures/Ligo_O5_range.pdf')
    plt.show()

    plt.imshow(ko5_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower")
    plt.xlabel(r"$m_{1}$")
    plt.ylabel(r"$m_{2}$")
    plt.title('Kagra range - O5')
    plt.colorbar(label = 'Distance (Mpc)')
    plt.tight_layout()
    plt.savefig('paper_figures/Kagra_O5_range.pdf')
    plt.show()

    plt.imshow(vo5_array, extent=[min(masses),max(masses),min(masses),max(masses)], origin="lower")
    plt.xlabel(r"$m_{1}$")
    plt.ylabel(r"$m_{2}$")
    plt.title('Virgo range - O5')
    plt.colorbar(label = 'Distance (Mpc)')
    plt.tight_layout()
    plt.savefig('paper_figures/Virgo_O5_range.pdf')
    plt.show()


if __name__ == '__main__':

    #makeScalingLawsPlot()
    #makeDnsMassHistograms()
    #makeMejEjectaPlot()
    #makeGW170817PhotometryPlot()
    #makeGW170817SedSurfacePlot()
    #makeTrialsEjectaScatter()
    #makeTrialsEjectaHistogram()
    #makeTrialsAvPlot()
    #makeInterceptSurface()
    #makeSlopeSurface()
    #makeExponentSurface()
    #makeRedKnLc()
    #makeBlueKnLc()
    #makeGW170817PhotometryPlotVillar()
    #makeO4PSD()
    #makeO5PSD()
    makeBNSRangePlot()



