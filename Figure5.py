import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

GW170817_chirp = 1.1977
GW170817_dist = 41

GW190425_chirp = 1.44
GW190425_dist = 159 

def getChirpMass(m1, m2):

    m_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    return m_chirp

def fig1(df, name, c):



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

    bins = np.arange(-0.5, np.max(gw_mergers) + 1)


    plt.hist(gw_mergers, histtype='step',color = c, bins=bins, label=name)
    plt.axvline(gw_mean, label = r'$\langle N_{{mergers}} \rangle = {:.1f}$'.format(gw_mean), c = c, linestyle='--')

def fig2(df, name, c):
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

    plt.hist(chirp_masses, histtype='step', bins = bins, color=c, label=name)
    #plt.axvline(np.mean(chirp_masses), label = r'$\langle M_{{chirp}} \rangle = %.1f {M}_{\odot }$ ' % (np.mean(chirp_masses)), c = c, linestyle='--')

def fig3(df, name, c):

    gw1_df = df[df['gw1'] == True]
    gw2_df = df[df['gw2'] == True]
    gw3_df = df[df['gw3'] == True]
    gw4_df = df[df['gw4'] == True]


    dist1 = gw1_df['dist']
    dist2 = gw2_df['dist']
    dist3 = gw3_df['dist']    
    dist4 = gw4_df['dist']     

    chirp_1 = getChirpMass(gw1_df['m1'],gw1_df['m2']).to_numpy() 
    chirp_2 = getChirpMass(gw2_df['m1'],gw2_df['m2']).to_numpy() 
    chirp_3 = getChirpMass(gw3_df['m1'],gw3_df['m2']).to_numpy() 
    chirp_4 = getChirpMass(gw4_df['m1'],gw4_df['m2']).to_numpy() 

    chirp_masses = np.concatenate((chirp_1, chirp_2, chirp_3, chirp_4))
    dist = np.concatenate((dist1, dist2, dist3, dist4))

    sns.kdeplot(chirp_masses, dist,levels=[0.2, 0.5, 0.8], color=c)




if __name__=='__main__':


    df_farrow = pd.read_csv(f'O4-DECam-r-23mag-Farrow/trials_df.csv')
    df_galaudage = pd.read_csv(f'O4-DECam-r-23mag-Galaudage/trials_df.csv')

    fig1(df_farrow, "Farrow et al.", 'C0')
    fig1(df_galaudage, "Galaudage et al.", 'C1')

    plt.xlabel(r'Number of BNS mergers detected ($N_{mergers}$)', fontsize='x-large')
    plt.ylabel('Count', fontsize='x-large')

    plt.tight_layout()

    plt.legend()
    plt.savefig('paper_figures/BNS_mergers_hist.pdf')
    plt.show()

    # Fig 2 - Chirp masses distribution

    fig2(df_farrow, "Farrow et al.", 'C0')
    fig2(df_galaudage, "Galaudage et al.", 'C1')

    plt.axvline(GW170817_chirp, label= r'GW170817  $M_{chirp} \sim {1.2}\,{M}_{\odot }$', c = 'red', linestyle='dotted')
    plt.axvline(GW190425_chirp, label= r'GW190425  $M_{chirp} \sim {1.44}\,{M}_{\odot }$', c = 'black', linestyle='dotted')
    plt.xlabel(r'$M_{chirp} ({M}_{\odot})$',fontsize='x-large')
    plt.ylabel('Count', fontsize='x-large')

    plt.legend()

    plt.tight_layout()
    plt.savefig('paper_figures/chirp_masses.pdf')
    plt.show()

    # Figure 3 - dist  vs chirp mass

    fig3(df_farrow, "Farrow et al.", 'C0')
    fig3(df_galaudage, "Galaudage et al.", 'C1')



    colors = ['C0', 'C1']
    lines = [Line2D([0], [0], color=colors[0]), Line2D([0], [0], color=colors[1])]
    labels = ['Farrow et al.', 'Galaudage et al.']
    plt.legend(lines, labels)

    plt.errorbar(GW190425_chirp, GW190425_dist, xerr=[[0.002],[0.004]], yerr=[[72], [69]], marker='x', c='black', ecolor='black')
    plt.text(GW190425_chirp + 0.01, GW190425_dist + 0.5,'GW190425', c='black')

    plt.errorbar(GW170817_chirp, GW170817_dist,  xerr=[[0.002],[0.002]], yerr=[[3.1], [3.1]], marker='x', c='red', ecolor='red')
    plt.text(GW170817_chirp + 0.01, GW170817_dist + 0.5,'GW170817', c='red')


    plt.xlabel(r'$M_{chirp} ({M}_{\odot})$', fontsize='x-large')
    plt.ylabel(r'Distance (Mpc)',fontsize='x-large')


    plt.tight_layout()
    plt.savefig('paper_figures/chirp_distance.pdf')
    plt.show()



