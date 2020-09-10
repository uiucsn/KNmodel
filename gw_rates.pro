seed = 1

mw_dist = 1


;Assume O3 starts Feb 1, 2019 with 2 weeks of engineering in November
;Cycle 26 runs Oct 1, 2018 - Sept 30, 2019.
;That means 241 days of overlap of O3 and C26, with 255 days including engineering.

duration = 255/365.25

bns_ligo_horizon = 120.
bns_virgo_horizon = 65.
chirp_scale = 2.66

n_events = 10000
size = 500
x = randomu(seed, n_events)*size-size/2.
y = randomu(seed, n_events)*size-size/2.
z = randomu(seed, n_events)*size-size/2.

if (mw_dist eq 1) then begin & $
  mean_mass = 1.33 & $
  sig_mass  = 0.09 & $
  mass1 = randomn(seed, n_events)*sig_mass + mean_mass & $
  mass2 = randomn(seed, n_events)*sig_mass + mean_mass & $
  tot_mass = mass1+mass2 & $
endif

if (mw_dist eq 0) then begin & $
  minmass = 1 & $
  maxmass = 3 & $
  mass1 = randomu(seed, n_events)*(maxmass-minmass) + minmass & $
  mass2 = randomu(seed, n_events)*(maxmass-minmass) + minmass & $
  tot_mass = mass1+mass2 & $
endif

h_on = fltarr(n_events)
l_on = fltarr(n_events)
v_on = fltarr(n_events)

h_duty = 0.6
l_duty = 0.6
v_duty = 0.6

h_on[where(randomu(seed, n_events) gt 1-h_duty)] = 1
l_on[where(randomu(seed, n_events) gt 1-l_duty)] = 1
v_on[where(randomu(seed, n_events) gt 1-v_duty)] = 1

dist = sqrt(x^2.+y^2.+z^2.)

mean_lograte = -5.95
sig_lograte  = 0.55

rate = 10^(randomn(seed, n_events)*sig_lograte + mean_lograte)
;rate = 3.2e-7 ;Mpc^-3/year
;rate = 4e-6 ;Mpc^-3/year

rate_full_volume = round(rate*size^3.*duration)

n_try = 1000
n_detect2 = fltarr(n_try)
n_detect3 = fltarr(n_try)
dist_detect = fltarr(n_try, n_events) - 1

for i = 0, n_try - 1 do begin & $

  index = randomu(seed, rate_full_volume[i])*n_events & $
  keep2 = where( $
    ((dist[index] lt bns_ligo_horizon*tot_mass[index]/chirp_scale) and (h_on[index] eq 1) and (l_on[index] eq 1)) or $
    ((dist[index] lt bns_ligo_horizon*tot_mass[index]/chirp_scale) and (v_on[index] eq 1) and ((h_on[index] eq 1) or (l_on[index] eq 1))), n2) & $

  n_detect2[i] = n2 & $

  keep3 = where((dist[index] lt bns_virgo_horizon*tot_mass[index]/chirp_scale) and (h_on[index]+l_on[index]+v_on[index] eq 3), n3) & $

  n_detect3[i] = n3 & $

  if (n2 gt 0) then dist_detect[i,index[keep2]] = dist[index[keep2]] & $
end

plothist,dist_detect[where(dist_detect gt 0)]

;plothist,n_detect2

;plothist,n_detect3

print, n_elements(where(n_detect2 gt 3))/float(n_try)
print, n_elements(where(n_detect2 gt 5))/float(n_try)
print, n_elements(where(n_detect2 gt 20))/float(n_try)
