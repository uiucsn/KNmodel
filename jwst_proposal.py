import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as spstat
import astropy.units as u
from astropy.coordinates import SkyCoord

np.random.seed(42)

def get_options(argv=None):
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_dir', required=True, help='Directory to store simulation results')

    args = parser.parse_args(args=argv)
    return args

def prepForJWST(df):

    sky_fraction = 0.39

    og_len = len(df)

    # Remove events too close to the galactic plane
    ra, dec = df['ra'].to_numpy(), df['dec'].to_numpy()
    sc = SkyCoord(ra=ra, dec=dec, unit='rad')
    b = sc.galactic.b.deg
    df = df[(b >= 10) | (b <= -10)]

    new_df = []

    # Randomly choose 39 percent of the events.
    for i in range(1000):
        trial_df = df[df['trial_number'] == i]
        idx = np.random.randint(1, len(trial_df), int(sky_fraction * len(trial_df)))
        trial_df = trial_df.iloc[idx]

        new_df.append(trial_df)

    new_df = pd.concat(new_df)

    print(len(new_df)/og_len * 100 , "percent of the events were retained")
    return new_df



def stat1(df):

    gw1_luminous = []
    gw2_luminous = []
    gw3_luminous = []
    gw4_luminous = []

    for i in range(1000):

        trial_df = df[df['trial_number'] == i]

        # Number of events in the trial with 1,2,3, or 4 GW detections
        gw1_df = trial_df[trial_df['gw1'] == True]
        gw2_df = trial_df[trial_df['gw2'] == True]
        gw3_df = trial_df[trial_df['gw3'] == True]
        gw4_df = trial_df[trial_df['gw4'] == True]

        # Find out how many of them leave behind a luminous remnant
        gw1_luminous.append(len(gw1_df[gw1_df['mej_dyn'] + gw1_df['mej_wind'] > 0]))
        gw2_luminous.append(len(gw2_df[gw2_df['mej_dyn'] + gw2_df['mej_wind'] > 0]))
        gw3_luminous.append(len(gw3_df[gw3_df['mej_dyn'] + gw3_df['mej_wind'] > 0]))
        gw4_luminous.append(len(gw4_df[gw4_df['mej_dyn'] + gw4_df['mej_wind'] > 0]))

    res = {
        1: np.array(gw1_luminous),
        2: np.array(gw2_luminous),
        3: np.array(gw3_luminous),
        4: np.array(gw4_luminous),
        'total': np.array(gw1_luminous) + np.array(gw2_luminous) + np.array(gw3_luminous) + np.array(gw4_luminous)
    }

    # Compute the statistic of the data collected

    res_5, res_95, res_median, res_mean = {}, {}, {}, {}
    
    for key in res:

        res_5[key] = np.percentile(res[key], 5)
        res_95[key] = np.percentile(res[key], 95)
        res_median[key] = np.median(res[key])
        res_mean[key] = np.mean(res[key])

    print('Result 1')
    print('# of events wth 4, 3, 2, 1 coincidental GW detection + ejecta')
    print("5th percentiles", res_5)
    print("Medians", res_median)
    print("95th percentiles", res_95)
    print("Means", res_mean)
    print()

    return res

def stat2(df):

    d_60 = []
    d_100 = []
    d_150 = []
    d_200 = []

    for i in range(1000):

        trial_df = df[df['trial_number'] == i]

        # Events must have GW detection on  at least one instrument
        trial_df = trial_df[(trial_df['gw1'] == True) | (trial_df['gw2'] == True) | (trial_df['gw3'] == True) | (trial_df['gw4'] == True)]

        # Number of events in the trial within 60, 100, 150, and 200 Mpc
        d60_df = trial_df[trial_df['dist'] < 60]
        d100_df = trial_df[trial_df['dist'] < 100]
        d150_df = trial_df[trial_df['dist'] < 150]
        d200_df = trial_df[trial_df['dist'] < 200]

        # Find out how many of them leave behind a luminous remnant
        d_60.append(len(d60_df[d60_df['mej_dyn'] + d60_df['mej_wind'] > 0]))
        d_100.append(len(d100_df[d100_df['mej_dyn'] + d100_df['mej_wind'] > 0]))
        d_150.append(len(d150_df[d150_df['mej_dyn'] + d150_df['mej_wind'] > 0]))
        d_200.append(len(d200_df[d200_df['mej_dyn'] + d200_df['mej_wind'] > 0]))

    res = {
        60: np.array(d_60),
        100: np.array(d_100),
        150: np.array(d_150),
        200: np.array(d_200),
    }

    # Compute the statistic of the data collected

    res_5, res_95, res_median, res_mean = {}, {}, {}, {}
    
    for key in res:

        res_5[key] = np.percentile(res[key], 5)
        res_95[key] = np.percentile(res[key], 95)
        res_median[key] = np.median(res[key])
        res_mean[key] = np.mean(res[key])

    print('Result 2')
    print("# of with a GW detection + ejecta at D < 60, 100, 150, 200 Mpc")
    print("5th percentiles", res_5)
    print("Medians", res_median)
    print("95th percentiles", res_95)
    print("Means", res_mean)
    print()

    return res

def stat3(df):

    successes1 = 0
    successes2 = 0
    successes3 = 0

    for i in range(1000):

        trial_df = df[df['trial_number'] == i]

        # GW + EM
        trial_df = trial_df[(trial_df['gw1'] == True) | (trial_df['gw2'] == True) | (trial_df['gw3'] == True) | (trial_df['gw4'] == True)]
        trial_df = trial_df[trial_df['mej_dyn'] + trial_df['mej_wind'] > 0]

        if len(trial_df) >= 1:
            successes1 += 1
        if len(trial_df) >= 2:
            successes2 += 1
        if len(trial_df) >= 3:
            successes3 += 1
    
    res = {
        1: successes1/1000 * 100,
        2: successes2/1000 * 100,
        3: successes3/1000 * 100,
    }
    
    print('Result 3')
    print(f'Chance of having >= 1,2,3 events with 1+ GW detection + some remnant (%)')
    print(res)
    print()



def stat4(df):

    # GW Detection + remnant
    df = df[(df['gw1'] == True) | (df['gw2'] == True) | (df['gw3'] == True) | (df['gw4'] == True)]
    df = df[df['mej_dyn'] + df['mej_wind'] > 0]


    dist = df['dist'].to_numpy()


    # Compute the statistic of the data collected
    res = {
        '5': np.percentile(dist, 5),
        '95': np.percentile(dist, 95),
        'median': np.median(dist),
        'mean': np.mean(dist)
    }

    print('Result 4')
    print('Distances of event distances with GW detection + ejecta')
    print(res)
    print()

    return res

def stat5(df):

    successes1 = 0
    successes2 = 0
    successes3 = 0

    for i in range(1000):

        trial_df = df[df['trial_number'] == i]

        # GW + EM
        trial_df = trial_df[(trial_df['gw2'] == True) | (trial_df['gw3'] == True) | (trial_df['gw4'] == True)]
        trial_df = trial_df[trial_df['mej_dyn'] + trial_df['mej_wind'] > 0]

        if len(trial_df) >= 1:
            successes1 += 1
        if len(trial_df) >= 2:
            successes2 += 1
        if len(trial_df) >= 3:
            successes3 += 1
    
    res = {
        1: successes1/1000 * 100,
        2: successes2/1000 * 100,
        3: successes3/1000 * 100,
    }
    
    print('Result 5')
    print(f'Chance of having >= 1,2,3 events with 2+ GW detection + some remnant (%)')
    print(res)
    print()

def getDistances(df):

    # GW Detection + remnant
    df = df[df['mej_dyn'] + df['mej_wind'] > 0]
    df1 = df[(df['gw1'] == True)]
    df2 = df[(df['gw2'] == True)]
    df3 = df[(df['gw3'] == True)]
    df4 = df[(df['gw4'] == True)]


    # Compute the statistic of the data collected
    res = {
        1: df1['dist'].to_numpy(),
        2: df2['dist'].to_numpy(),
        3: df3['dist'].to_numpy(),
        4: df4['dist'].to_numpy(),
    }

    return res

if __name__=='__main__':

    argv = sys.argv[1:]

    args = get_options(argv=argv)
    trials_dir = args.trials_dir


    df = pd.read_csv(f'{trials_dir}/trials_df.csv')
    df = prepForJWST(df)

    # Result 1 - number of KNe observable by JWST(*) wth 4, 3, 2 detectors
    res1 = stat1(df)
    dist = getDistances(df)

    # Result 2 - number of KNe observable by JWST with D < 60, 100, 150, 200 Mpc
    res2 = stat2(df)
    res3 = stat3(df)
    res4 = stat4(df)
    res5 = stat5(df)

    fig_kw = {'figsize':(8.5/0.7, 4)}
    fig, ax = plt.subplots(nrows=1, ncols=2, **fig_kw)


    # Plot results 1
    bins = np.arange(-0.5, max(res1['total']) + 1)
    for i in range(1,4):
        ax[0].hist(res1[i], bins=bins,  color = f"C{i - 1}", histtype='step', density = True, lw = 2)
        ax[0].hist(res1[i], bins=bins, color = f"C{i - 1}", histtype='stepfilled', density = True, alpha = 0.5)
    
        ax[0].axvline(np.mean(res1[i]), label=f"<N> = {np.mean(res1[i]):.1f}", color = f"C{i - 1}", linestyle = 'dashed', lw = 2)

    ax[0].hist(res1['total'], bins=bins, color = "black", density = True, histtype='step', lw = 3)
    ax[0].axvline(np.mean(res1['total']), label=f"Total <N> = {np.mean(res1['total']):.1f}", color = "black", linestyle = 'dashed', lw = 2)

    ax[0].set_xlabel('Number of luminous remnants (N)', fontsize = 'x-large')
    ax[0].set_ylabel('P(N)', fontsize = 'x-large')

    ax[0].legend()
    ax[0].set_yscale('log')

    # Plot results 1
    patches = []
    legend_text = []
    dist_range = np.arange(0, 260, 0.1)
    for i in range(1,4):

        kde = spstat.gaussian_kde(dist[i], bw_method='scott')
        pdist = kde(dist_range)

        ax[1].plot(dist_range, pdist, color = f"C{i - 1}", lw = 2)
        patch = ax[1].fill_between(dist_range, np.zeros(len(dist_range)), pdist, color=f"C{i - 1}", alpha=0.5, zorder=1)

        ax[1].axvline(np.mean(dist[i]), label=f"<D> = {np.mean(dist[i]):.0f} Mpc", color = f"C{i - 1}", linestyle = 'dashed', lw = 2)

        patches.append(patch)
        legend_text.append(f"{i} Detector")


    ax[1].set_xlabel('Distance (D, Mpc)' , fontsize = 'x-large')
    ax[1].set_ylabel('P(D)', fontsize = 'x-large')

    ax[1].legend()
    fig.legend(patches, legend_text,
                'upper center', frameon=False, ncol=4, fontsize='medium')

    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=1.05)


    plt.show()

    fig.savefig(f'{trials_dir}/{trials_dir[:-1]}_jwst_proposal.pdf')


    # ds = [60, 100, 150, 200]

    # # Plot results 1
    # bins = np.arange(-0.5, 20)
    # for i,key in enumerate(res2):
    #     plt.hist(res2[key], bins=bins, label=f"Distance < {key} Mpc", color = f"C{i}", histtype='step', lw = 3)
    #     #plt.hist(res1[key], bins=bins,  color = f"C{i}", histtype='stepfilled', alpha = 0.5)


    # plt.xlabel('Number of events within distance limit')
    # plt.ylabel('Number of trials')

    # plt.legend()
    # plt.yscale('log')

    # plt.show()