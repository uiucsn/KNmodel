import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

day_interval = 50

def plot_days_to_KN_discovery_distribution(df, run, color):

    data = df[f"O{run} Days to KN"]
    data_med = int(np.median(data))
    data_5 = int(np.percentile(data, 5))
    data_95 = int(np.percentile(data, 95))

    run_label = f"O{run}"


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
        axs[1].legend() 

    axs[1].grid()

    #plt.title(f"LVK Observing Run {run}", fontsize='x-large')
    axs[1].set_xlabel(rf"Days to first discoverable KN ($D_{{KN}}^{{{run_label}}}$)", fontsize='x-large')
    axs[1].set_ylabel(r"Density", fontsize='x-large')

    #plt.show()
    plt.tight_layout()
    plt.savefig(f'O{run}_KN_dist.pdf', bbox_inches = "tight")
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
    plt.savefig("delta_t_distribution.pdf", bbox_inches = "tight")
    plt.close()


def rates_vs_days_to_kn(df):

    rates = df["Rates"]
    o4_days = df[f"O4 Days to KN"]
    o5_days = df[f"O5 Days to KN"]

    o4_kn_indices = np.where(o4_days.to_numpy()!=5 * 365)[0]
    o4_no_kn_indices = np.where(o4_days.to_numpy()==5 * 365)[0]

    plt.scatter(rates[o4_kn_indices], o4_days[o4_kn_indices], marker='.', alpha=0.5, label='O4', color='C4')
    plt.scatter(rates[o4_no_kn_indices], o4_days[o4_no_kn_indices], marker=r'$\uparrow$', s=18, alpha=0.5, label='O4 - No KN', color='C4')

    plt.scatter(rates, o5_days, marker='.', alpha=0.5, label='O5', color='C2')

    plt.axhline(744, 0, 1, label="Current O4 Duration", color='black', linestyle='dashed')

    plt.legend()

    plt.xlabel(r"BNS Merger Rate ($Gpc^{-3} \cdot yr^{-1}$)", fontsize='x-large')
    plt.ylabel(r"$D_{KN}$", fontsize='x-large')
    
    plt.savefig("bns_rate_vs_days.pdf", bbox_inches = "tight")
    plt.close()

def rates_vs_delta_days_to_kn(df):

    rates = df["Rates"]
    O4_kn_days, O5_kn_days = df["O4 Days to KN"], df["O5 Days to KN"]
    data = O4_kn_days - O5_kn_days

    plt.scatter(rates, data, marker='.', alpha=0.5, color='gray')

    plt.xlabel(r"BNS Merger Rate ($Gpc^{-3} \cdot yr^{-1}$)", fontsize='x-large')
    plt.ylabel(r"$\Delta D_{KN}$", fontsize='x-large')
    
    plt.savefig("bns_rate_vs_delta_days_to_KN.pdf", bbox_inches = "tight")
    plt.close()


def bns_rates_distribution(df):

    data = df[f"Rates"]
    data_med = int(np.median(data))
    data_5 = int(np.percentile(data, 5))
    data_95 = int(np.percentile(data, 95))

    bins = 10**np.linspace(np.log10(min(data)), np.log10(max(data)), 30)

    plt.hist(data, density=True, histtype='step', fill=False, linewidth=2, bins=bins, alpha=1, color="C0")
    plt.hist(data, density=True, histtype='stepfilled', fill=True, bins=bins, alpha=0.5, color="C0")

    plt.axvline(data_5, label=r"$P_{5\%=}$" + f"{data_5}", linestyle='dotted', ymin=0, ymax=1, color='black')
    plt.axvline(data_med, label=r"$P_{50\%=}$" + f"{data_med}", linestyle='dashed', ymin=0, ymax=1, color='black')
    plt.axvline(data_95, label=r"$P_{95\%=}$" + f"{data_95}", linestyle='dotted', ymin=0, ymax=1, color='black')

    plt.xlabel(r"BNS Merger Rate ($Gpc^{-3} \cdot yr^{-1}$)", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    plt.xscale('log')
    plt.xlim()
    plt.legend()

    plt.savefig("bns_rate_dist.pdf", bbox_inches = "tight")


    
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