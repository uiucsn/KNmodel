import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

GW170817_chirp = 1.1977
GW170817_dist = 43

def get_options(argv=None):
    '''
    Get commandline options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_dir', required=True, help='Directory to store simulation results')
    parser.add_argument('--obs_run', required=True, choices=['O4','O5'], help='Observing run')

    args = parser.parse_args(args=argv)
    return args

def getChirpMass(m1, m2):

    m_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    return m_chirp

if __name__=='__main__':

    argv = sys.argv[1:]

    args = get_options(argv=argv)
    trials_dir = args.trials_dir
    obs_run = args.obs_run

    df = pd.read_csv(f'{trials_dir}/trials_df.csv')

    # Fig 1 - Number of BNS mergers detected by LVK

    gw1 = df.groupby('trial_number')['gw1'].sum()
    gw2 = df.groupby('trial_number')['gw2'].sum()
    gw3 = df.groupby('trial_number')['gw3'].sum()
    gw4 = df.groupby('trial_number')['gw4'].sum()

    gw_mergers = gw1 + gw2 + gw3 + gw4
    gw_mean = np.mean(gw_mergers)
    gw_median = np.median(gw_mergers)
    gw_5 = np.percentile(gw_mergers, 5)
    gw_95 = np.percentile(gw_mergers, 95)

    print('BNS mergers detected: ${',  gw_median, "}_{-", gw_median - gw_5, "}^{+", gw_95 - gw_median, "}$")

    bins = np.arange(0, np.max(gw_mergers))


    plt.hist(gw_mergers, histtype='step', bins=bins)
    plt.axvline(gw_mean, label = r'$\langle N_{{mergers}} \rangle = {:.1f}$'.format(gw_mean), c = 'black', linestyle='--')
    plt.axvline(gw_5, label = r'$\langle N_{{mergers}} \rangle_{5} = %.1f$ ' % (gw_5), c = 'red', linestyle='dotted')
    plt.axvline(gw_95, label = r'$\langle N_{{mergers}} \rangle_{95} = %.1f$ ' % (gw_95), c = 'red', linestyle='dotted')

    plt.xlabel(r'Number of BNS mergers detected ($N_{mergers}$)', fontsize='x-large')
    plt.ylabel('Count', fontsize='x-large')

    #plt.yscale('log')
    plt.tight_layout()

    plt.legend()
    plt.savefig(f'{trials_dir}/BNS_mergers_hist.pdf')
    plt.show()

    # Fig 2 - Chirp masses distribution

    gw1_df = df[df['gw1'] == True]
    gw2_df = df[df['gw2'] == True]
    gw3_df = df[df['gw3'] == True]
    gw4_df = df[df['gw4'] == True]

    chirp_1 = getChirpMass(gw1_df['m1'],gw1_df['m2']).to_numpy() 
    chirp_2 = getChirpMass(gw2_df['m1'],gw2_df['m2']).to_numpy() 
    chirp_3 = getChirpMass(gw3_df['m1'],gw3_df['m2']).to_numpy() 
    chirp_4 = getChirpMass(gw4_df['m1'],gw4_df['m2']).to_numpy() 

    chirp_masses = np.concatenate((chirp_1, chirp_2, chirp_3, chirp_4))
    bins = np.arange(min(chirp_masses) - 0.1, max(chirp_masses)+ 0.1, 0.01) 

    plt.hist(chirp_masses, histtype='step', bins = bins)
    plt.axvline(np.mean(chirp_masses), label = r'$\langle M_{{chirp}} \rangle = %.1f {M}_{\odot }$ ' % (np.mean(chirp_masses)), c = 'black', linestyle='--')
    plt.axvline(1.1977, label= r'GW170817  $M_{chirp} \sim {1.2}\,{M}_{\odot }$', c = 'red', linestyle='dotted')

    plt.xlabel(r'$M_{chirp} ({M}_{\odot})$',fontsize='x-large')
    plt.ylabel('Count', fontsize='x-large')

    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{trials_dir}/chirp_masses.pdf')
    plt.show()

    # Figure 3 - Inclination  vs chirp mass
    cos_theta1 = gw1_df['cos_theta']
    cos_theta2 = gw2_df['cos_theta']
    cos_theta3 = gw3_df['cos_theta']    
    cos_theta4 = gw4_df['cos_theta']   

    cos_theta = np.concatenate((cos_theta1, cos_theta2, cos_theta3, cos_theta4))


    thetas = np.rad2deg(np.arccos(cos_theta))
    omegas = np.minimum(thetas, 180 - thetas)

    sns.kdeplot(chirp_masses, omegas ,  cmap="Reds", levels=15)


    plt.xlabel(r'$M_{chirp} ({M}_{\odot})$', fontsize='x-large')
    plt.ylabel(r'Inclination (degrees)',fontsize='x-large')

    plt.tight_layout()
    plt.savefig(f'{trials_dir}/chirp_inclination.pdf')
    plt.show()


    # Figure 4 - dist  vs chirp mass
    dist1 = gw1_df['dist']
    dist2 = gw2_df['dist']
    dist3 = gw3_df['dist']    
    dist4 = gw4_df['dist']     

    dist = np.concatenate((dist1, dist2, dist3, dist4))

    sns.kdeplot(chirp_masses, dist, cmap="Reds", levels=15)
    plt.scatter(GW170817_chirp, GW170817_dist, marker='*',c='black', label='GW170817')


    plt.xlabel(r'$M_{chirp} ({M}_{\odot})$', fontsize='x-large')
    plt.ylabel(r'Distance (Mpc)',fontsize='x-large')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{trials_dir}/chirp_distance.pdf')
    plt.show()

    # Figure 5 - ejecta mass histogram
    no_mej = df.groupby('trial_number')
    no_mej_fracs = []

    for i in range(len(no_mej)):

        temp = df[df['trial_number'] == i]

        N = len(temp)
        M = len(temp.loc[(temp['mej_wind']==0) & (temp['mej_dyn']==0)])
        f = M/N * 100
        no_mej_fracs.append(f)

    print(r"%","of mergers producing zero ejecta ${", np.mean(no_mej_fracs), "}_{-", np.mean(no_mej_fracs) - np.percentile(no_mej_fracs, 5) ,"}^{+", np.percentile(no_mej_fracs, 95) - np.mean(no_mej_fracs), "}$" )

    peak_u = df['peak_u'].to_numpy()
    peak_g = df['peak_g'].to_numpy()
    peak_r = df['peak_r'].to_numpy()
    peak_i = df['peak_i'].to_numpy()
    peak_z = df['peak_z'].to_numpy()
    peak_y = df['peak_y'].to_numpy()

    peak_u[peak_u==np.inf] = np.nan
    peak_g[peak_g==np.inf] = np.nan
    peak_r[peak_r==np.inf] = np.nan
    peak_i[peak_i==np.inf] = np.nan
    peak_z[peak_z==np.inf] = np.nan
    peak_y[peak_y==np.inf] = np.nan

    print(max(peak_u),max(peak_i), max(peak_y))
    bins = np.arange(min(min(peak_u),min(peak_i), min(peak_z), min(peak_y)) - 1, max(max(peak_u),max(peak_g),max(peak_r),max(peak_i), max(peak_z), max(peak_y)) + 1, 0.5)
    plt.hist(peak_u, label = 'LSST u', histtype='step', color='C1', lw=3, bins=bins)
    #plt.hist(peak_g, label = 'LSST g', histtype='step', color='C2', lw=3, bins=bins)
    #plt.hist(peak_r, label = 'LSST r', histtype='step', color='C3', lw=3, bins=bins)
    plt.hist(peak_i, label = 'LSST i', histtype='step', color='C4', lw=3, bins=bins)
    #plt.hist(peak_z, label = 'LSST z', histtype='step', color='C5', lw=3, bins=bins)
    plt.hist(peak_y, label = 'LSST y', histtype='step', color='C6', lw=3, bins=bins)

    plt.hist(peak_u, histtype='stepfilled', color='C1', alpha=0.3, bins=bins)
    #plt.hist(peak_g, histtype='stepfilled', color='C2', alpha=0.5, bins=bins)
    #plt.hist(peak_r, histtype='stepfilled', color='C3', alpha=0.3, bins=bins)
    plt.hist(peak_i, histtype='stepfilled', color='C4', alpha=0.3, bins=bins)
    #plt.hist(peak_z, histtype='stepfilled', color='C5', alpha=0.5, bins=bins)
    plt.hist(peak_y, histtype='stepfilled', color='C6', alpha=0.3, bins=bins)

    plt.xlabel("Peak mag (AB)", fontsize='x-large')
    plt.ylabel("Count",fontsize='x-large')

    plt.yscale('log')
    plt.xlim(12, 40)

    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{trials_dir}/peak_mags.pdf')
    plt.show()