import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rates_models import LVK_UG

day_interval = 50

def plot_days_to_KN_discovery_distribution(df, run, color):

    data = df[f"O{run} Days to KN"]
    data_med = int(np.median(data))
    data_5 = int(np.percentile(data, 5))
    data_95 = int(np.percentile(data, 95))

    run_label = f"O{run}"

    if run == "4":
        bins = np.arange(0,5 * 365, day_interval)
        bins = np.append(bins[:-1], 1900)
    else:
        bins = np.arange(0, max(data) + day_interval, day_interval)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6, 10))

    axs[0].hist(data, density=True, bins=bins, histtype='step', linewidth=2, fill=False, alpha=1, color=color)
    axs[0].hist(data, density=True, bins=bins, histtype='stepfilled', fill=True, alpha=0.5, color=color)

    axs[0].axvline(data_5, label=r"$P_{5\%} $=" + f"{data_5}", linestyle='dotted', ymin=0, ymax=1, color='black')
    axs[0].axvline(data_med, label=r"$P_{50\%} $=" + f"{data_med}", linestyle='dashed', ymin=0, ymax=1, color='black')
    axs[0].axvline(data_95, label=r"$P_{95\%} $=" + f"{data_95}", linestyle='dotted', ymin=0, ymax=1, color='black')

    axs[0].legend()

    #plt.title(f"LVK Observing Run {run}", fontsize='x-large')
    #axs[0].set_xlabel(rf"Days to first discoverable KN ($D_{{KN}}^{{{run_label}}}$)", fontsize='x-large')
    axs[0].set_ylabel(r"Density", fontsize='x-large')

    #plt.xlim(0, max(df["O5 Days to KN"]))

    axs[1].hist(data, density=True, bins=len(data), histtype='step', linewidth=2, fill=False, alpha=1, color=color, cumulative=True)
    axs[1].hist(data, density=True, bins=len(data), histtype='stepfilled', fill=True, alpha=0.5, color=color, cumulative=True)

    if run=='4':

        axs[1].axvline(744, 0, 1, label="Current O4 Duration", color='black', linestyle='dashed')
        axs[0].arrow(x=1790, y=0.00008, dx=50, dy=0, head_length=20, width=0.00002, fc ='black')
        axs[1].legend() 

    axs[1].grid()

    #plt.title(f"LVK Observing Run {run}", fontsize='x-large')
    axs[1].set_xlabel(rf"Days to first discoverable KN ($D_{{KN}}^{{{run_label}}}$)", fontsize='x-large')
    axs[1].set_ylabel(r"Density", fontsize='x-large')

    #plt.show()
    plt.tight_layout()
    plt.savefig(f'O4_O5_critical_plots/O{run}_KN_dist.pdf', bbox_inches = "tight")
    plt.close()


def plot_delta_days_distribution(df, color):

    O4_kn_days, O5_kn_days = df["O4 Days to KN"], df["O5 Days to KN"]
    data = O4_kn_days - O5_kn_days

    data_med = int(np.median(data))
    data_5 = int(np.percentile(data, 5))
    data_95 = int(np.percentile(data, 95))

    bins = np.arange(-500,  2000, day_interval)

    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(6, 10))


    axs[0].hist(data, density=True, histtype='step', linewidth=2, fill=False, bins=bins, alpha=1, color=color)
    axs[0].hist(data, density=True, histtype='stepfilled', fill=True, bins=bins, alpha=0.5, color=color)

    axs[0].axvline(data_5, label=r"$P_{5\%}=$" + f"{data_5}", linestyle='dotted', ymin=0, ymax=1, color='black')
    axs[0].axvline(data_med, label=r"$P_{50\%}=$" + f"{data_med}", linestyle='dashed', ymin=0, ymax=1, color='black')
    axs[0].axvline(data_95, label=r"$P_{95\%}=$" + f"{data_95}", linestyle='dotted', ymin=0, ymax=1, color='black')
    axs[0].axvline(730, label=r"2 year shutdown", linestyle='dashed', ymin=0, ymax=1, color='red')

    axs[0].legend()

    #axs[0].set_xlabel(r"Difference in days to first discoverable KN ($\Delta D_{KN}$)", fontsize='x-large')
    axs[0].set_ylabel(r"Density", fontsize='x-large')

    axs[1].hist(data, density=True, histtype='step', linewidth=2, fill=False, bins=len(data), alpha=1, color=color, cumulative=True)
    axs[1].hist(data, density=True, histtype='stepfilled', fill=True, bins=len(data), alpha=0.5, color=color, cumulative=True)
    
    axs[1].axvline(730, label=r"2 year shutdown", linestyle='dashed', ymin=0, ymax=1, color='red')

    axs[1].grid()

    axs[1].set_xlabel(r"Difference in days to first discoverable KN ($\Delta D_{KN}$)", fontsize='x-large')
    axs[1].set_ylabel(r"Density", fontsize='x-large')

    plt.tight_layout()
    plt.savefig("O4_O5_critical_plots/delta_t_distribution.pdf", bbox_inches = "tight")
    plt.close()


def rates_vs_days_to_kn(df):

    rates = df["Rate"]
    o4_days = df[f"O4 Days to KN"]
    o5_days = df[f"O5 Days to KN"]

    o4_kn_indices = np.where(o4_days.to_numpy()!=5 * 365)[0]
    o4_no_kn_indices = np.where(o4_days.to_numpy()==5 * 365)[0]

    plt.scatter(rates[o4_kn_indices], o4_days[o4_kn_indices], marker='.', alpha=0.5, label='O4 - KN detection', color='C4')
    plt.scatter(rates[o4_no_kn_indices], o4_days[o4_no_kn_indices], marker=r'$\uparrow$', s=18, alpha=0.5, label='O4 - No KN detection', color='C4')

    plt.scatter(rates, o5_days, marker='.', alpha=0.5, label='O5 - KN detection', color='C2')

    plt.axhline(744, 0, 1, label="Current O4 Duration", color='black', linestyle='dashed')

    plt.legend()

    plt.xlabel(r"BNS Merger Rate ($Gpc^{-3} \cdot yr^{-1}$)", fontsize='x-large')
    plt.ylabel(r"$D_{KN}$", fontsize='x-large')
    
    plt.savefig("O4_O5_critical_plots/bns_rate_vs_days.pdf", bbox_inches = "tight")
    plt.close()

def rates_vs_delta_days_to_kn(df):

    rates = df["Rate"]
    O4_kn_days, O5_kn_days = df["O4 Days to KN"], df["O5 Days to KN"]
    data = O4_kn_days - O5_kn_days

    o4_days = df[f"O4 Days to KN"]
    o4_kn_indices = np.where(o4_days.to_numpy()!=5 * 365)[0]
    o4_no_kn_indices = np.where(o4_days.to_numpy()==5 * 365)[0]

    plt.scatter(rates[o4_kn_indices], data[o4_kn_indices], marker='.', alpha=0.5, color='gray', label='O4 - KN detection')
    plt.scatter(rates[o4_no_kn_indices], data[o4_no_kn_indices], marker=r'$\uparrow$', alpha=0.5, color='gray', s=18, label='O4 - No  KN detection')

    plt.legend()

    plt.xlabel(r"BNS Merger Rate ($Gpc^{-3} \cdot yr^{-1}$)", fontsize='x-large')
    plt.ylabel(r"$\Delta D_{KN}$", fontsize='x-large')
    
    plt.savefig("O4_O5_critical_plots/bns_rate_vs_delta_days_to_KN.pdf", bbox_inches = "tight")
    plt.close()

def plot_chirp_mass_dist(df):

    o4_days = df[f"O4 Days to KN"]
    o4_kn_indices = np.where(o4_days.to_numpy()!=5 * 365)[0]
    o4_no_kn_indices = np.where(o4_days.to_numpy()==5 * 365)[0]

    O4_m1, O4_m2 =  df[f"O4 m1"][o4_kn_indices], df[f"O4 m2"][o4_kn_indices]
    O5_m1, O5_m2 =  df[f"O5 m1"], df[f"O5 m2"]

    O4_chirp = (O4_m1 * O4_m2)**0.6/(O4_m1 + O4_m2)**0.2
    O5_chirp = (O5_m1 * O5_m2)**0.6/(O5_m1 + O5_m2)**0.2

    plt.hist(O4_chirp, density=True, histtype='step', linewidth=2, fill=False, bins=50, alpha=1, color="C4")
    plt.hist(O4_chirp, density=True, histtype='stepfilled', fill=True, bins=50, alpha=0.5, color="C4", label="O4")

    plt.hist(O5_chirp, density=True, histtype='step', linewidth=2, fill=False, bins=50, alpha=1, color="C2")
    plt.hist(O5_chirp, density=True, histtype='stepfilled', fill=True, bins=50, alpha=0.5, color="C2", label="O5")

    plt.legend()

    plt.xlabel(r"$M_{chirp}$", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    plt.savefig("O4_O5_critical_plots/m_chirp_dist.pdf", bbox_inches = "tight")
    plt.close()

def plot_peak_mag_dist(df):

    o4_days = df[f"O4 Days to KN"]
    o4_kn_indices = np.where(o4_days.to_numpy()!=5 * 365)[0]
    o4_no_kn_indices = np.where(o4_days.to_numpy()==5 * 365)[0]

    O4_peak_mag =  df[f"O4 peak mag"][o4_kn_indices]
    O5_peak_mag =  df[f"O5 peak mag"]

    plt.hist(O4_peak_mag, density=True, histtype='step', linewidth=2, fill=False, bins=50, alpha=1, color="C4")
    plt.hist(O4_peak_mag, density=True, histtype='stepfilled', fill=True, bins=50, alpha=0.5, color="C4", label="O4")

    plt.hist(O5_peak_mag, density=True, histtype='step', linewidth=2, fill=False, bins=50, alpha=1, color="C2")
    plt.hist(O5_peak_mag, density=True, histtype='stepfilled', fill=True, bins=50, alpha=0.5, color="C2", label="O5")

    plt.legend()

    plt.xlabel(r"Peak Mag (DECam r-band)", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    plt.savefig("O4_O5_critical_plots/peak_mag_dist.pdf", bbox_inches = "tight")
    plt.close()

def plot_distance_dist(df):

    o4_days = df[f"O4 Days to KN"]
    o4_kn_indices = np.where(o4_days.to_numpy()!=5 * 365)[0]
    o4_no_kn_indices = np.where(o4_days.to_numpy()==5 * 365)[0]

    O4_dist =  df[f"O4 dist"][o4_kn_indices]
    O5_dist =  df[f"O5 dist"]

    plt.hist(O4_dist, density=True, histtype='step', linewidth=2, fill=False, bins=50, alpha=1, color="C4")
    plt.hist(O4_dist, density=True, histtype='stepfilled', fill=True, bins=50, alpha=0.5, color="C4", label="O4")

    plt.hist(O5_dist, density=True, histtype='step', linewidth=2, fill=False, bins=50, alpha=1, color="C2")
    plt.hist(O5_dist, density=True, histtype='stepfilled', fill=True, bins=50, alpha=0.5, color="C2", label="O5")

    plt.legend()

    plt.xlabel(r"Luminosity Distance", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    plt.savefig("O4_O5_critical_plots/distance_dist.pdf", bbox_inches = "tight")
    plt.close()

def bns_rates_distribution(df):

    rate_exponents = np.arange(1e-3, 1e3, 0.01)
    rates = 10**rate_exponents
    samples, d = LVK_UG(n)


    fx = d.pdf(rate_exponents)
    Fx = d.cdf(rate_exponents)
    r_5 = 10**d.ppf(0.05)
    r_50 = 10**d.ppf(0.50)
    r_95 = 10**d.ppf(0.95)
    print(r_5, r_95)

    plt.plot(rates, fx, color='C0')
    plt.fill_between(rates, fx, color='C0', alpha=0.5)
    #plt.plot(rates, Fx, label=r"CDF")
    plt.axvline(x=r_5, label=rf'$P_{{5\%}}$ = {math.ceil(r_5)}', linestyle='dotted', color='black')
    plt.axvline(x=r_50, label=rf'$P_{{50\%}}$ = {math.ceil(r_50)}', linestyle='dashed', color='black')
    plt.axvline(x=r_95, label=rf'$P_{{95\%}}$ = {math.ceil(r_95)}', linestyle='dotted', color='black')
    plt.xlabel(r'BNS Merger Rate ($GPc^{-3} yr^{-1}$)', fontsize='x-large')
    plt.ylabel("Density", fontsize='x-large')
    plt.xscale('log')
    plt.xlim(left = 10**1.5, right = 10**3.3)
    plt.legend()

    plt.savefig("O4_O5_critical_plots/bns_rate_dist.pdf", bbox_inches = "tight")
    plt.close()

def plot_lvc_correlation_matrix():

    lvc_cor_matrix = np.array([[1., 0.59, 0.58, 0.56],
                               [0.59, 1., 0.70, 0.56],
                               [0.58, 0.70, 1., 0.56],
                               [0.56, 0.56, 0.56, 1.]])
    
    plt.imshow(lvc_cor_matrix)
    for i in range(lvc_cor_matrix.shape[0]):
        for j in range(lvc_cor_matrix.shape[1]):
            if i == j:
                text = plt.text(j, i, lvc_cor_matrix[i, j], ha="center", va="center", color="black")
            else:
                text = plt.text(j, i, lvc_cor_matrix[i, j], ha="center", va="center", color="w")
            
    plt.xticks(range(4),['LIGO H', 'LIGO L', 'Virgo', 'KAGRA'])
    plt.yticks(range(4),['LIGO H', 'LIGO L', 'Virgo', 'KAGRA'])
    plt.tight_layout()

    plt.savefig("O4_O5_critical_plots/correlations.pdf", bbox_inches = "tight")
    plt.close()

    
if __name__=="__main__":

    df = pd.read_csv("critical_point.csv")
    n = df.to_numpy().shape[0]

    # Count the number of zero KN detections
    o4_no_detections = len(np.where(df['O4 Days to KN'].to_numpy() == -1)[0])
    o5_no_detections = len(np.where(df['O5 Days to KN'].to_numpy() == -1)[0])

    print("O4 no detection", o4_no_detections/n * 100, " %")
    print("O5 no detection", o5_no_detections/n * 100, " %")

    df.replace(-1, 5 * 365, inplace=True)

    plot_days_to_KN_discovery_distribution(df, "4", "C4")
    plot_days_to_KN_discovery_distribution(df, "5", "C2")

    plot_delta_days_distribution(df, 'gray')

    rates_vs_days_to_kn(df)
    rates_vs_delta_days_to_kn(df)
    bns_rates_distribution(df)

    plot_chirp_mass_dist(df)
    plot_peak_mag_dist(df)
    plot_distance_dist(df)
    plot_lvc_correlation_matrix()