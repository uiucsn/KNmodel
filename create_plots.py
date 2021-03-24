import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spstat

pops = ('flat', 'mw')


#res = {pop: pickle.load(open(f'n-events-{pop}.pickle', 'rb')) for pop in pops}
#res = {pop: pickle.load(open(f'n-events-hst-29-{pop}.pickle', 'rb')) for pop in pops}
# res = {pop: pickle.load(open(f'hst_h_band/n-events-hst-29-30-31-{pop}.pickle', 'rb')) for pop in pops}
res = {pop: pickle.load(open(f'hst_r_band/n-events-hst-29-30-31-{pop}.pickle', 'rb')) for pop in pops}

n_kn = 2

all_2_det_events = np.hstack([res[pop]['n_detect2'] for pop in pops])
all_3_det_events = np.hstack([res[pop]['n_detect3'] for pop in pops])
all_4_det_events = np.hstack([res[pop]['n_detect4'] for pop in pops])

all_2_det_dist = np.hstack([res[pop]['dist_detect2'] for pop in pops])
all_3_det_dist = np.hstack([res[pop]['dist_detect3'] for pop in pops])
all_4_det_dist = np.hstack([res[pop]['dist_detect4'] for pop in pops])

all_2_det_mags = np.hstack([res[pop]['rmah_detect2'] for pop in pops])
all_3_det_mags = np.hstack([res[pop]['rmah_detect3'] for pop in pops])
all_4_det_mags = np.hstack([res[pop]['rmah_detect4'] for pop in pops])
# all_2_det_mags = np.hstack([res[pop]['hmag_detect2'] for pop in pops])
# all_3_det_mags = np.hstack([res[pop]['hmag_detect3'] for pop in pops])
# all_4_det_mags = np.hstack([res[pop]['hmag_detect4'] for pop in pops])

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.5/0.7, 3.5))

ebins = np.arange(32)
patches = []
legend_text = []
for events, distances, mags, color, nn in zip(
            (all_2_det_events, all_3_det_events, all_4_det_events),
            (all_2_det_dist, all_3_det_dist, all_4_det_dist),
            (all_2_det_mags, all_3_det_mags, all_4_det_mags),
            ('C0', 'C1', 'C2'),
            (2, 3, 4)
        ):
    # P(N > 1)
    print(f"P(N >= 1 event detected) for {nn}-detector events: {np.sum(events >= 1)/len(events):.3f}")
    # event count distribution
    vals, _, _ = axes[0].hist(events, histtype='stepfilled',
                            bins=ebins, color=color, alpha=0.3,
                            density=True, zorder=0)
    axes[0].hist(events, density=True, histtype='step', color=color, lw=3,
                 bins=ebins, zorder=2)
    bin_centers = 0.5*(ebins[0:-1] + ebins[1:])
    mean_nevents = np.mean(events)
    five_percent, ninetyfive_percent = np.percentile(events, 5), np.percentile(events, 95)
    axes[0].axvline(round(mean_nevents), color=color, linestyle='--', lw=2,
                    label=r'$\langle N\rangle = %d ;~ N_{95} = %d$' % (round(mean_nevents), ninetyfive_percent))
    axes[0].axvline(ninetyfive_percent, color=color,
                    linestyle='dotted', lw=1)
    # distance distribution
    if len(distances) < 10: continue
    dist_range = np.arange(0, 400., 0.1)
    kde = spstat.gaussian_kde(distances, bw_method='scott')
    pdist = kde(dist_range)
    axes[1].plot(dist_range, pdist, color=color, linestyle='-', lw=3, zorder=4)
    patch = axes[1].fill_between(dist_range, np.zeros(len(dist_range)), pdist, color=color, alpha=0.3, zorder=0)
    patches.append(patch)
    legend_text.append('2 Detector Events')
    mean_dist = np.mean(distances)
    axes[1].axvline(
        mean_dist, color=color, linestyle='--', lw=1.5, zorder=6,
        label=r'$\langle D \rangle = {:.0f}$ Mpc'.format(mean_dist)
    )
    # magnitude distribution
    h_range = np.arange(15, 23, 0.1)
    kde = spstat.gaussian_kde(mags, bw_method='scott')
    ph = kde(h_range)
    axes[2].plot(h_range, ph, color=color, linestyle='-', lw=3, zorder=4)
    axes[2].fill_between(h_range, np.zeros(len(h_range)), ph, color=color,
                         alpha=0.3, zorder=0)
    mean_h = np.mean(mags)
    # axes[2].axvline(mean_h, color=color, linestyle='--', lw=1.5,
    #                 zorder=6, label=r'$\langle R \rangle = {:.1f}$ mag'.format(mean_h))
    axes[2].axvline(mean_h, color=color, linestyle='--', lw=1.5,
                    zorder=6, label=r'$\langle H \rangle = {:.1f}$ mag'.format(mean_h))

axes[0].set_yscale('log')
axes[0].legend(frameon=False, fontsize='small', loc='upper right')
axes[1].legend(frameon=False, fontsize='small')
axes[2].legend(frameon=False, fontsize='small')
plt.show()
#plt.savefig('hst_r_band/hst_gw_detect_hst_29_30_31.pdf')