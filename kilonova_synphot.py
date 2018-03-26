#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Routines to get synthetic photometry from Dan Kasen's Kilonova model
- Gautham Narayan (gnarayan@stsci.edu), 20180325
"""
import sys
import os
import numpy as np
import astropy
import astropy.constants
import astropy.coordinates as c
import astropy.units as u
import astropy.table as at
import h5py
import bisect
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
cdbs = os.getenv('PYSYN_CDBS')
if cdbs is None:
    cdbs = '~/work/synphot/'
    cdbs = os.path.expanduser(cdbs)
    os.putenv('PYSYN_CDBS', cdbs)
import pysynphot as S
import webbpsf as W


# define some constants
SPEED_OF_LIGHT = astropy.constants.c.cgs.value
TELESCOPE_AREA = 25.0 * 10000 # cm^2 -- Area of the telescope has to be in centimeters because pysynphot...
S.setref(area=TELESCOPE_AREA)
SEC_TO_DAY = u.second.to(u.day)
CM_TO_ANGSTROM = u.centimeter.to(u.angstrom)
ANGSTROM_TO_CM = u.angstrom.to(u.centimeter)
FNU_TO_MJY = (u.erg/(u.centimeter**2)/u.second/u.Hertz).to(u.microjansky)
ANGSTROM_TO_MICRON = u.angstrom.to(u.micron)
MPC_TO_CM = u.megaparsec.to(u.centimeter)
DISTANCE = [50, 120, 200]
TMAX = 90

class Kilonova(object):
    def __init__(self):
        """
        Read Dan's Kilonova spectral model and return the base arrays
        """
        name = 'spectrum.h5'
        fin    = h5py.File(name,'r')

        # frequency in Hz
        nu    = np.array(fin['nu'],dtype='d')
        # array of time in seconds
        times = np.array(fin['time'])
        # covert time to days
        times *= SEC_TO_DAY

        # specific luminosity (ergs/s/Hz)
        # this is a 2D array, Lnu[times][nu]
        Lnu_all   = np.array(fin['Lnu'],dtype='d')

        self._times   = times
        self._nu      = nu
        self._Lnu_all = Lnu_all

    def get_model(self, phase):
        """
        Get the flam spectrum for some specific phase
        """
        it  = bisect.bisect(self._times, phase)
        it -= 1 # I think Dan's array indexing is off by 1
        Lnu = self._Lnu_all[it,:]
        # if you want thing in Flambda (ergs/s/Angstrom)
        lam  = SPEED_OF_LIGHT/self._nu*CM_TO_ANGSTROM
        Llam = Lnu*self._nu**2.0/SPEED_OF_LIGHT/CM_TO_ANGSTROM
        return lam, Llam

    def get_norm_model(self, phase, distance):
        """
        Get the flam spectrum for some specific phase and distance
        """
        dist = c.Distance(distance*u.megaparsec)
        z = dist.z
        lam, flam = self.get_model(phase)
        lamz = lam*(1.+z)
        fnorm = flam/(4*np.pi*(distance*MPC_TO_CM)**2.)
        return lamz, fnorm


def main():
    # save the figures
    figdir = 'Figures'

    # just listing the wide filters
    nircam_bandpasses = 'F070W,F090W,F115W,F150W,F200W,F277W,F356W,F444W'
    miri_bandpasses   = 'F560W,F770W,F1000W,F1130W,F1280W,F1500W,F1800W,F2100W,F2550W'
    nircam_bandpasses = nircam_bandpasses.split(',')
    miri_bandpasses   = miri_bandpasses.split(',')

    # configure the instruments
    nircam = W.NIRCam()
    miri   = W.MIRI()

    # load the bandpasses
    bandpasses = {}
    for bp in nircam_bandpasses:
        nircam.filter = bp
        bpmodel = nircam._getSynphotBandpass(nircam.filter)
        bandpasses[bp] = bpmodel
    for bp in miri_bandpasses:
        miri.filter = bp
        bpmodel = miri._getSynphotBandpass(miri.filter)
        bandpasses[bp] = bpmodel

    # we just need a couple of bandpasses for testing
    use_bandpasses = nircam_bandpasses + miri_bandpasses[0:3]

    # init the kilonova model and create some arrays to store output
    kn = Kilonova()

    # do this for a few distances
    for j, dmpc in enumerate(DISTANCE):
        time    = []
        rfphase = []
        flux    = {}

        if j == 0:
            fig = plt.figure(figsize=(8,15))
            ax = fig.add_subplot(1,1,1)

        for i, phase in enumerate(kn._times):
            # get the kilonova model and spectrum for this phase and distance
            lam, flam   = kn.get_model(phase)
            lamz, fnorm = kn.get_norm_model(phase, dmpc)
            name = 'kilonova_{:+.2f}'.format(phase)
            spec = S.ArraySpectrum(wave=lamz, flux=fnorm, waveunits='angstrom', fluxunits='flam', name=name)

            # get synthetic mags in each passband
            for bp in use_bandpasses:
                passband = bandpasses[bp]
                obs  = S.Observation(spec, passband, force='taper')
                try:
                    mag = obs.effstim('abmag')
                except ValueError as e:
                    mag = np.nan
                thispb = flux.get(bp)
                if thispb is None:
                    thispb = [mag,]
                else:
                    thispb.append(mag)
                flux[bp] = thispb

            # keep a track of rest-frame phase + observer frame days (should make much difference at these distances)
            dist = c.Distance(dmpc*u.megaparsec)
            z = dist.z
            rfphase.append(phase)
            time.append(phase*(1.+z))

            # write output photometry tables
            if i % 5 == 0:
                # convert flam -> fnu -> mJy (YUCK)
                lam_micron = lamz *ANGSTROM_TO_MICRON
                f_nu = (((lamz * ANGSTROM_TO_CM)**2.) / SPEED_OF_LIGHT) * fnorm * (CM_TO_ANGSTROM / ANGSTROM_TO_MICRON)
                f_mjy = f_nu * FNU_TO_MJY
                table_name = 'Tables/kilnova_orig_{}Mpc_p{:+.2f}.txt'.format(dmpc, phase)
                this_spec = at.Table([lam_micron, f_mjy], names=['wave_micron','flux_mjy'])
                this_spec.sort('wave_micron')
                this_spec.write(table_name, format='ascii.fixed_width', delimiter=' ',overwrite='True')

                # plot output spectral sequence
                if j == 0:
                    fplot = flam/flam.mean()
                    ax.plot(lam*ANGSTROM_TO_MICRON, fplot+180 - i, 'k-')

        # finalize spectral sequence plot
        if j==0:
            ax.tick_params(axis='both', which='major', labelsize='large')
            ax.set_xlabel(r'Rest Wavelength ($\mu$m)', fontsize='xx-large')
            ax.set_ylabel(r'Relative F$_{\lambda}$ + constant', fontsize='xx-large')
            ax.set_xlim(0.5, 9.5)
            fig.tight_layout(rect=[0,0,1,0.96])
            plt.savefig('{}/kilonova_spec.pdf'.format(figdir))
            plt.close(fig)

        # dump output mag tables
        arrays = [rfphase, time,] + [flux[bp] for bp in use_bandpasses]
        names  = ['rfphase', 'ofphase'] + [bp for bp in use_bandpasses]
        out = at.Table(arrays, names=names)
        out.write('Tables/kilonova_phottable_{}Mpc.txt'.format(dmpc), delimiter=' ', format='ascii.fixed_width', overwrite=True)

        # plot up the lightcurves
        npbs = len(use_bandpasses)
        color=iter(plt.cm.tab20(np.linspace(0,1,npbs)))
        with PdfPages('{}/kilonova_phot_{}Mpc.pdf'.format(figdir, dmpc)) as pdf:
            fig = plt.figure(figsize=(10,10))
            for i, bp in enumerate(use_bandpasses):

                # save four passbands per page
                if i%4 == 0 and i > 0:
                    fig.suptitle('Kilonova Synthetic Photometry {} Mpc'.format(dmpc), fontsize='xx-large')
                    fig.tight_layout(rect=[0,0,1,0.93])
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig = plt.figure(figsize=(10,10))

                # plot up a passband
                ax = fig.add_subplot(2,2,i%4+1)
                thiscol = next(color)
                ax.plot(out['ofphase'], out[bp], marker='o', linestyle='-', lw=0.5, label=bp, color=thiscol)
                ax.tick_params(axis='both', which='major', labelsize='large')
                ax.set_ylabel('{} (AB mag)'.format(bp), fontsize='xx-large')
                ax.set_xlabel('Observer-frame Phase (Days)', fontsize='xx-large')
                ax.legend(loc='upper right', frameon=False)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim((ymax, ymin))

                # finalize lightcurve plot
                if i == npbs-1:
                    fig.suptitle('Kilonova Synthetic Photometry {} Mpc'.format(dmpc), fontsize='xx-large')
                    fig.tight_layout(rect=[0,0,1,0.93])
                    pdf.savefig(fig)
                    plt.close(fig)


if __name__=='__main__':
    sys.exit(main())
