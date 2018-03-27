import pysynphot as S
import astropy.table as at
import webbpsf as W
nc = W.NIRCam()
nc.filter = 'F115W'
bp = nc._getSynphotBandpass(nc.filter)
f = at.Table.read('Tables/kilnova_orig_50Mpc_p+5.25.txt', format='ascii')
spec = S.ArraySpectrum(f['wave_micron'],f['flux_mjy'], waveunits='micron', fluxunits='mjy')
spec2 = S.ArraySpectrum(f['wave_micron'],f['flux_mjy']*1E5, waveunits='micron', fluxunits='mjy')
TELESCOPE_AREA = 25.0 * 10000
S.setref(area=TELESCOPE_AREA)
obs = S.Observation(spec, bp,force='taper')
obs2 = S.Observation(spec2, bp,force='taper')
print('WRONG FOR PYSYNPHOT, RIGHT FOR ETC:', obs.effstim('abmag'))
print('RIGHT FOR PYSYNPHOT, WRONG FOR ETC:',obs2.effstim('abmag'))
