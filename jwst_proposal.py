import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def get_options(argv=None):
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_dir', required=True, help='Directory to store simulation results')

    args = parser.parse_args(args=argv)
    return args

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

    print('Stat 1 - Results')
    print("5th percentiles", res_5)
    print("Medians", res_median)
    print("95th percentiles", res_95)
    print("Means", res_mean)

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

    print('Stat 2 - Results')
    print("5th percentiles", res_5)
    print("Medians", res_median)
    print("95th percentiles", res_95)
    print("Means", res_mean)

    return res

def stat3(df):

    gw1_percent = []
    gw2_percent = []
    gw3_percent = []
    gw4_percent = []

    for i in range(1000):

        trial_df = df[df['trial_number'] == i]

        # Number of events in the trial with 1,2,3, or 4 GW detections
        gw1_df = trial_df[trial_df['gw1'] == True]
        gw2_df = trial_df[trial_df['gw2'] == True]
        gw3_df = trial_df[trial_df['gw3'] == True]
        gw4_df = trial_df[trial_df['gw4'] == True]

        # Find out how many of them leave behind a luminous remnant
        if len(gw1_df) != 0:
            gw1_percent.append(len(gw1_df[gw1_df['mej_dyn'] + gw1_df['mej_wind'] > 0]) / len(trial_df) * 100)
        else:
            gw1_percent.append(0)

        if len(gw2_df) != 0:
            gw2_percent.append(len(gw2_df[gw2_df['mej_dyn'] + gw2_df['mej_wind'] > 0]) / len(trial_df) * 100)
        else:
            gw2_percent.append(0)
        
        if len(gw3_df) != 0:
            gw3_percent.append(len(gw3_df[gw3_df['mej_dyn'] + gw3_df['mej_wind'] > 0]) / len(trial_df) * 100)
        else:
            gw3_percent.append(0)
        
        if len(gw4_df) != 0:
            gw4_percent.append(len(gw4_df[gw4_df['mej_dyn'] + gw4_df['mej_wind'] > 0]) / len(trial_df) * 100)
        else:
            gw4_percent.append(0)

    res = {
        1: np.array(gw1_percent),
        2: np.array(gw2_percent),
        3: np.array(gw3_percent),
        4: np.array(gw4_percent),
        'total': np.array(gw1_percent) + np.array(gw2_percent) + np.array(gw3_percent) + np.array(gw4_percent)
    }

    # Compute the statistic of the data collected

    res_5, res_95, res_median, res_mean = {}, {}, {}, {}
    
    for key in res:

        res_5[key] = np.percentile(res[key], 5)
        res_95[key] = np.percentile(res[key], 95)
        res_median[key] = np.median(res[key])
        res_mean[key] = np.mean(res[key])

    print('Stat 3 - Results')
    print("5th percentiles", res_5)
    print("Medians", res_median)
    print("95th percentiles", res_95)
    print("Means", res_mean)

    return res


def stat4(df):

    median_dist = []

    for i in range(1000):

        trial_df = df[df['trial_number'] == i]

        # Events must have GW detection on  at least one instrument
        trial_df = trial_df[(trial_df['gw1'] == True) | (trial_df['gw2'] == True) | (trial_df['gw3'] == True) | (trial_df['gw4'] == True)]

        if len(trial_df) != 0:
            median_dist.append(np.median(trial_df['dist'].to_numpy()))

    res = {
        'total': np.array(median_dist)
    }


    # Compute the statistic of the data collected
    res_5, res_95, res_median, res_mean = {}, {}, {}, {}
    
    for key in res:

        res_5[key] = np.percentile(res[key], 5)
        res_95[key] = np.percentile(res[key], 95)
        res_median[key] = np.median(res[key])
        res_mean[key] = np.mean(res[key])

    print('Stat 4 - Results')
    print("5th percentiles", res_5)
    print("Medians", res_median)
    print("95th percentiles", res_95)
    print("Means", res_mean)

    return res




if __name__=='__main__':

    argv = sys.argv[1:]

    args = get_options(argv=argv)
    trials_dir = args.trials_dir


    df = pd.read_csv(f'{trials_dir}/trials_df.csv')

    # Result 1 - number of KNe observable by JWST(*) wth 4, 3, 2 detectors
    res1 = stat1(df)

    # Plot results 1
    bins = np.arange(-0.5, 20)
    for i in range(1,5):
        plt.hist(res1[i], bins=bins, label=f"{i} Detector events", color = f"C{i}", histtype='step', lw = 3)
        plt.hist(res1[i], bins=bins,  color = f"C{i}", histtype='stepfilled', alpha = 0.5)

    plt.hist(res1['total'], bins=bins, label=f"Total", color = "black", histtype='step', lw = 3)


    plt.xlabel('Number of luminous remnants')
    plt.ylabel('Number of trials')

    plt.legend()
    plt.yscale('log')

    plt.show()

    # Result 2 - number of KNe observable by JWST with D < 60, 100, 150, 200 Mpc
    res2 = stat2(df)
    ds = [60, 100, 150, 200]

    # Plot results 1
    bins = np.arange(-0.5, 20)
    for i,key in enumerate(res2):
        plt.hist(res2[key], bins=bins, label=f"Distance < {key} Mpc", color = f"C{i}", histtype='step', lw = 3)
        #plt.hist(res1[key], bins=bins,  color = f"C{i}", histtype='stepfilled', alpha = 0.5)


    plt.xlabel('Number of events within distance limit')
    plt.ylabel('Number of trials')

    plt.legend()
    plt.yscale('log')

    plt.show()

    res3 = stat3(df)
    ds = [60, 100, 150, 200]

    # Plot results 1
    bins = np.arange(-0.5, 20)
    for i in range(1,5):
        plt.hist(res3[i], bins=bins, label=f"{i} Detector events", color = f"C{i}", histtype='step', lw = 3)
        plt.hist(res3[i], bins=bins,  color = f"C{i}", histtype='stepfilled', alpha = 0.5)

    plt.hist(res3['total'], bins=bins, label=f"Total", color = "black", histtype='step', lw = 3)

    plt.xlabel('Percentage of events with luminous remnant')
    plt.ylabel('Number of trials')

    plt.legend()
    plt.yscale('log')

    plt.show()


    res4 = stat4(df)



    plt.hist(res4['total'], label=f"Total", color = "black", histtype='step', lw = 3)

    plt.xlabel('Median distance of luminous remnant')
    plt.ylabel('Number of trials')

    plt.legend()
    plt.yscale('log')

    plt.show()