import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, Distance
import afterglowpy as grb
import sncosmo
from tqdm import tqdm

from interpolate_bulla_sed import BullaSEDInterpolator
from interpolate_bulla_sed import uniq_cos_theta, uniq_mej_dyn, uniq_mej_wind, uniq_phi, phases, lmbd
from sed_to_lc import SEDDerviedLC, lsst_bands

# register new bandpasses
    # STAR-X: http://star-x.xraydeep.org/observatory/#inst_reqs
    # rough cut off based on central and total coverage
sncosmo.register(sncosmo.Bandpass([165., 200.], [1., 1.], name='starx::180', wave_unit=u.nm), 'starx::180')
sncosmo.register(sncosmo.Bandpass([200., 250.], [1., 1.], name='starx::225', wave_unit=u.nm), 'starx::225')
sncosmo.register(sncosmo.Bandpass([250., 300.], [1., 1.], name='starx::275', wave_unit=u.nm), 'starx::275')
sncosmo.register(sncosmo.Bandpass([165., 300.], [1., 1.], name='starx::total', wave_unit=u.nm), 'starx::total')

    #UVEX: https://www.uvex.caltech.edu/page/about
sncosmo.register(sncosmo.Bandpass([1390., 1900.], [1., 1.], name='uvex::fuv', wave_unit=u.AA), 'uvex::fuv')
sncosmo.register(sncosmo.Bandpass([2030., 2700.], [1., 1.], name='uvex::nuv', wave_unit=u.AA), 'uvex::nuv')

# for checking the early time enhancement in IR
# phases = np.linspace(0.1, 5, 100)
# lmbd = np.linspace(1000, 2300, 131)*u.nm.to(u.AA)

# take the lmbd and phases from the KN grid and use it in afterglowpy
#t_s = phases*u.d.to(u.s)
#nu = (const.c / (lmbd*u.AA)).to(u.Hz)

# goal: take in a KN object + afterglow params and get the new combined SED
class AfterglowAddition():

    # add all parameters
    def __init__(self, KN, E0=10**52.96, thetaCore=0.066, n0=10**-2.7, p=2.17, epsilon_e=10**-1.4, 
                 epsilon_B=10**-4, time = phases, wav = lmbd, addKN = True):
        
        # initialize with either KN object or sed file
        if isinstance(KN, SEDDerviedLC):
            # get the sed and viewing angle from the obj
            self.KNsed = KN.sed
            theta_v = np.arccos(KN.cos_theta)
            self.host_ebv = KN.host_ebv
            self.mw_ebv = KN.mw_ebv

        # TODO: allow initialization without KN


        elif isinstance(KN, str):
            # get the viewing angle from the file name
            cos_theta = float(KN.split("_")[3])
            theta_v = np.arccos(cos_theta)
        
            # get the KNsed
            table = pd.read_csv(KN, delimiter=' ', names = ['Phase', 'Wavelength', 'Flux'])
            self.KNsed = np.array(table['Flux']).reshape(len(phases), len(lmbd))

        elif isinstance(KN, float):
            theta_v = KN
            addKN = False

        # to get the initial sed, d_L must = 10pc
        self.grb_params = { # use the same params I used for nsf fig
            'jetType':     grb.jet.Gaussian,   # Gaussian jet!! - not flat
            'specType':    0,                  # Basic Synchrotron Spectrum
            'thetaObs':    theta_v,            # Viewing angle in radians
            'E0':          E0,                 # Isotropic-equivalent energy in erg
            'thetaCore':   thetaCore,               # Half-opening angle in radians
            'thetaWing':  0.47,               # Wing angle in radians,  min(10*thetaCore, np.pi/2)
            'n0':          n0,                 # circumburst density in cm^{-3}
            'p':           p,               # electron energy distribution index
            'epsilon_e':   epsilon_e,           # epsilon_e
            'epsilon_B':   epsilon_B,             # epsilon_B
            'xi_N':        1.0,                # Fraction of electrons accelerated
            'd_L':         10*u.pc.to(u.cm),   # Luminosity distance in cm
        }
        self.phases = time # in days
        self.lmbd = wav # in Angstrom - as with Bulla grid

        self.t_s = self.phases*u.d.to(u.s)
        self.nu = (const.c / (self.lmbd*u.AA)).to(u.Hz)

        self.sed = self.getSed()
        if addKN:
            self.sed += self.KNsed

    # construct an afterglow SED in the same shape as the KN SED
    # TODO: add this to a pool? for each frequency of interest
    def getSed(self):
        Flmbda = np.empty((len(self.nu), len(self.t_s)))
        #print(Flmbda.shape, flush=True)
        for i, n in enumerate(self.nu):
            Fnu = (grb.fluxDensity(self.t_s, n, **self.grb_params)*u.mJy).to(u.erg/u.s/u.cm**2/u.Hz) # gives the flux density in mJy
            Fl = (Fnu*(n**2)/const.c).to(u.erg/u.s/u.cm**2/u.AA)
            Flmbda[i][:] = Fl.value

        return np.array(Flmbda).transpose() #+ self.KN.sed

    # from ved, adapted to use the afterglow SED
        # remove False extinction part
    def getAbsMagsInPassbands(self, passbands, apply_extinction = True, apply_redshift = False):

        lcs = {}
        
        for passband in passbands:
            source_name = f"test_{passband}"

            #print(lc_phases.shape, lmbd.shape, self.sed.shape, flush=True)
            source = sncosmo.TimeSeriesSource(phase=self.phases, wave=self.lmbd, flux = self.sed, name=source_name, zero_before=True)

            model = sncosmo.Model(source)

            if apply_extinction:

                # add host galaxy extinction E(B-V)
                model.add_effect(sncosmo.CCM89Dust(), 'host', 'rest')
                model.set(hostebv = self.host_ebv)

                # add MW extinction to observing frame
                model.add_effect(sncosmo.F99Dust(), 'mw', 'obs')
                model.set(mwebv=self.mw_ebv)

            if apply_redshift:

                # Adding redshift based on distance: https://docs.astropy.org/en/stable/api/astropy.coordinates.Distance.html#astropy.coordinates.Distance.z
                z = self.distance.z
                model.set(z=z)

            abs_mags = model.bandmag(band=passband, time = self.phases, magsys="ab")
            lcs[passband] = abs_mags

        return lcs
    
        # same as sed_to_lc
    def getAppMagsInPassbands(self, passbands, apply_extinction = False, apply_redshift = False):

        # Get abs mags first
        lcs = self.getAbsMagsInPassbands(passbands, apply_extinction = apply_extinction, apply_redshift= apply_redshift)

        # Add the distance modulus using the KN dist
        for passband in passbands:
            lcs[passband] += self.KN.distance.distmod.value

        return lcs

if __name__ == "__main__":
    
    def checkSEDs(seds, labels):

        plt.figure(figsize=(8,6))
        # plot the SED
        for i in range(len(phases)):

            for j, sed in enumerate(seds):
                plt.plot(lmbd, sed[i], label = labels[j])
        
            plt.title(f"phase = {phases[i]}")
            plt.xlabel("wavelength (AA)")
            plt.ylabel("Flambda (erg/s/cm^2/AA)")
            plt.legend()
            plt.show()

    def checkLCs(to_plot, labels, passbands = lsst_bands):
       
        #band = 'lssti'
        fig = plt.figure(figsize=(8,6))

        for i, obj in enumerate(to_plot):
            for j, band in enumerate(passbands):
                plt.plot(phases, obj.getAbsMagsInPassbands(passbands)[band], label=labels[i]+" "+band, color=f'C{j}')

        plt.title(f"{band}")
        plt.xlabel("time (days)")
        plt.ylabel("Absolute magnitude")
        plt.gca().invert_yaxis()
        plt.legend()
        fig.savefig(f'img/UV_IR.png')
        plt.show()

    def plotDistributions():

        # evenly spaced samples of a few parameters
        #logE0 = np.linspace(50, 56, 7)
        #logn = np.linspace(-5, 5, 11)
        theta = np.linspace(10, 0, 6)
        cos_theta = np.cos(theta*u.deg.to(u.rad))

        # GW170817 object on axis - holding these constant while changing afterglow probably not reasonable
            # I assume they effect each other
        # mej_dyn = 10**-2.27
        # mej_wind = 10**-1.28
        # phi = 49.5
        # #cos_theta = 1
        # c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        # d = 40*u.Mpc

        # GW170817 object on axis
        mej_wind = 0.05
        mej_dyn = 0.001
        phi = 30
        #theta = 7
        #cos_theta = np.cos(theta*u.deg.to(u.rad))
        c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        d = 40*u.Mpc

        #KN = SEDDerviedLC(mej_dyn, mej_wind, phi, cos_theta, dist=d, coord=c, av=0.0)

        # p_grb = { # these probably need to be changed
        #     'inclination_EM': 0,
        #     'log10_E0': 49.3,
        #     'thetaCore': 0.05,
        #     'thetaWing': 0.01,
        #     'log10_n0':-2,
        #     'p':2.250,
        #     'log10_epsilon_e':-1, 
        #     'log10_epsilon_B':-3.,
        #     'luminosity_distance': 40,}
        
        plt.figure(figsize=(8,6))
        cmap = cm['jet']

        for i, ct in tqdm(enumerate(cos_theta)): 
            KN = SEDDerviedLC(mej_dyn, mej_wind, phi, ct, dist=d, coord=c, av=0.0)      
            afterglow = AfterglowAddition(KN) # use typical values
            
            #t = np.arccos(ct)*u.rad.to(u.deg)
            plt.plot(phases, afterglow.getAbsMagsInPassbands(lsst_bands, apply_extinction=False)['lssti'],
                     label=r"$cos \theta_v$ = " + str(np.round(ct, 3)), color=cmap(i/len(cos_theta)))
            plt.plot(phases, KN.getAbsMagsInPassbands(lsst_bands, apply_extinction=False)['lssti'],
                     label=r"$cos \theta_v$ = " + str(np.round(ct, 3)), color=cmap(i/len(cos_theta)), linestyle='--')
        plt.title(f"Viewing between 0 and 10 degrees")
        #plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='Parameter Value')

        plt.xlabel("time (days)")
        plt.ylabel("M")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig("img/aft_theta.png")
        plt.show()

    def compareWithFile():

        # choose a few options for parameters
            # uniq_mej_dyn  = np.array([0.001, 0.005, 0.01, 0.02])  
            # uniq_mej_wind =  np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13])
            # uniq_phi = np.array([15, 30, 45, 60, 75])
            # uniq_cos_theta = np.array([0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        mej_dyn = np.array([0.001, 0.005, 0.005])
        mej_wind = np.array([0.01, 0.07, 0.09])
        phi = np.array([30, 45, 60])
        cos_theta = np.array([0.1, 0.5, 0.9])

        c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
        d = 40*u.Mpc            

        for i in range(len(mej_dyn)):
            md = mej_dyn[i]
            mw = mej_wind[i]
            p = phi[i]
            ct = cos_theta[i]

            KN = SEDDerviedLC(md, mw, p, ct, dist=d, coord=c, av=0.0)

            sed_dir = './SEDs/SIMSED.BULLA-BNS-M3-3COMP/'
            file = f'sed_cos_theta_{ct}_mejdyn_{md}_mejwind_{mw}0_phi_{p}.txt' 

            table = pd.read_csv(sed_dir+file, delimiter=' ', names = ['Phase', 'Wavelength', 'Flux'])
            file_sed = np.array(table['Flux']).reshape(len(phases), len(lmbd))

            
            for j, band in enumerate(lsst_bands):
                plt.plot(phases, KN.getAbsMagsInPassbands(lsst_bands)[band], color = f'C{j}', alpha=0.5)   
                
            # insert the new sed and recalc
            KN.sed = file_sed
            for j, band in enumerate(lsst_bands):
                plt.plot(phases, KN.getAbsMagsInPassbands(lsst_bands)[band], color = f'C{j}', linestyle='--')
            
            plt.xlabel("time (days)")
            plt.ylabel("M")
            plt.gca().invert_yaxis()
            plt.legend()
            plt.show()


def make_Ryan2020_F2():
    grb_params = { # use the same params I used for nsf fig
            'jetType':     grb.jet.Gaussian,   # Gaussian jet!! - not flat
            'specType':    0,                  # Basic Synchrotron Spectrum
            'thetaObs':    0,            # Viewing angle in radians
            'E0':          1e52,                 # Isotropic-equivalent energy in erg
            'thetaCore':   0.1,               # Half-opening angle in radians
            'thetaWing':   1,               # Wing angle in radians
            'n0':          1e-3,                 # circumburst density in cm^{-3}
            'p':           2.2,               # electron energy distribution index
            'epsilon_e':   1e-1,           # epsilon_e
            'epsilon_B':   1e-2,             # epsilon_B
            'xi_N':        1.0,                # Fraction of electrons accelerated
            'd_L':         3.09e26,   # Luminosity distance in cm
        }

    grb_params2 = { # use the same params I used for nsf fig
                'jetType':     grb.jet.TopHat,   # 
                'specType':    0,                  # Basic Synchrotron Spectrum
                'thetaObs':    0.16,            # Viewing angle in radians
                'E0':          1e52,                 # Isotropic-equivalent energy in erg
                'thetaCore':   0.1,               # Half-opening angle in radians
                'thetaWing':   1,               # Wing angle in radians
                'n0':          1e-3,                 # circumburst density in cm^{-3}
                'p':           2.2,               # electron energy distribution index
                'epsilon_e':   1e-1,           # epsilon_e
                'epsilon_B':   1e-2,             # epsilon_B
                'xi_N':        1.0,                # Fraction of electrons accelerated
                'd_L':         3.09e26,   # Luminosity distance in cm
            }

    fig, ax = plt.subplots(1,1)
    t = np.logspace(4, 8)
    for n in [1e9, 1e18]:
        Fnu = grb.fluxDensity(t, n, **grb_params)
        Fnu2 = grb.fluxDensity(t, n, **grb_params2)
        ax.plot(t, Fnu)
        ax.plot(t, Fnu2)
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.savefig("test_aftpy.png")

    
def makeTest():
    t_day = np.arange(0.0001, 20, 0.01)

    params_grb = { # from Troja 2020
    'E0': 10**52.9,
    'thetaCore': 0.07,
    'n0':10**-2.7,
    'p':2.17,
    'epsilon_e':10**-1.4, 
    'epsilon_B':10**-4.,
    }

    mej_wind = 1e-3
    mej_dyn = 1e-3
    phi = 60

    
    #theta = 7
    #cos_theta = np.cos(theta*u.deg.to(u.rad))
    c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
    d = 40*u.Mpc

    fig = plt.figure()
    for i, angle in enumerate(np.linspace(0, 8, 5)):
        ct = np.cos(angle*u.deg.to(u.rad))
        KN = SEDDerviedLC(mej_dyn, mej_wind, phi, ct, dist=d, coord=c, av=0.0)      
        afterglow = AfterglowAddition(KN, **params_grb) # use typical values
    
        band = 'ztfi'
        mag = KN.getAbsMagsInPassbands([band,], lc_phases=t_day, apply_extinction=False)
        plt.plot(t_day, mag[band], label=f'{int(angle)} deg', color=f'C{i}')

        mag_grb = afterglow.getAbsMagsInPassbands([band,], lc_phases=t_day, apply_extinction=False)
        plt.plot(t_day, mag_grb[band], color=f'C{i}', linestyle='--')

    plt.legend()
    plt.gca().invert_yaxis()
    plt.ylim(-5, -20)
    plt.xlim(0.1, 20)
    plt.xlabel('Time t [days]', fontsize=18)
    plt.ylabel('$m$ [mag]', fontsize=18)
    plt.xscale('log')
    fig.savefig('img/compare.png')
    plt.show()
# call stuff here
#compareWithFile()

#GW170817 object
# mej_wind = 0.05
# mej_dyn = 0.005
# phi = 30
# theta = 4*u.deg.to(u.rad)
# cos_theta = np.cos(theta) #1

# c = SkyCoord(ra = "13h09m48.08s", dec = "−23deg22min53.3sec")
# d = 40*u.Mpc
# KN = SEDDerviedLC(mej_dyn, mej_wind, phi, cos_theta, dist=d, coord=c, av=0.0)

# # afterglow
# E0 = 10**52.9 #erg
# n0 = # #cm**-3    
# #aftKN = AfterglowAddition(KN, E0, n0)
# afterglow = AfterglowAddition(KN, E0, n0, False)

# sed_dir = './SEDs/SIMSED.BULLA-BNS-M3-3COMP/'

# cos_theta = 0.0
# mej_dyn = 0.001
# mej_wind = 0.010
# phi = 0
# file = f'sed_cos_theta_{cos_theta}_mejdyn_{mej_dyn}_mejwind_{mej_wind}0_phi_{phi}.txt'

#KN = SEDDerviedLC(mej_dyn, mej_wind, phi, cos_theta, dist=d, coord=c, av=0.0)
# aftKN = AfterglowAddition(KN, E0, n0)
#afterglow = AfterglowAddition(KN, addKN=False)

# afterglow_f = AfterglowAddition(sed_dir + file, E0, n0, False)
# aftKN_f = AfterglowAddition(sed_dir + file, E0, n0)

# KN_f = SEDDerviedLC(mej_dyn, mej_wind, phi, cos_theta, dist=d, coord=c, av=0.0)
# KN_f.sed = afterglow_f.KNsed # replace calc'd sed with file sed so i can be plotted

#print(KN.sed == KN_f)
#checkLCs([afterglow, KN], ["afterglow", "KN"], passbands=['f125w', 'f160w', 'f200w', 'uvot::uvw2', 'uvot::uvw1'])
#checkSEDs([afterglow_f.sed, aftKN_f.KNsed, afterglow.sed, KN.sed], ["afterglow f", "KN f", "afterglow", "KN"])
#plotDistributions() 
#makeTest()   






