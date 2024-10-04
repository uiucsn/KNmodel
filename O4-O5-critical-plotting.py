import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_days_to_KN_discovery_distribution(df, run, color):

    data = df[f"O{run} Days to KN"]
    data_med = int(np.median(data))
    data_5 = int(np.percentile(data, 5))
    data_95 = int(np.percentile(data, 95))

    run_label = f"O{run}"

    plt.hist(data, density=True, histtype='step', fill=False, bins=25, alpha=1, color=color)
    plt.hist(data, density=True, histtype='stepfilled', fill=True, bins=25, alpha=0.5, color=color)

    plt.axvline(data_5, label=r"$P_{5\%} $=" + f"{data_5}", linestyle='dotted', ymin=0, ymax=1, color='black')
    plt.axvline(data_med, label=r"$P_{50\%} $=" + f"{data_med}", linestyle='dashed', ymin=0, ymax=1, color='black')
    plt.axvline(data_95, label=r"$P_{95\%} $=" + f"{data_95}", linestyle='dotted', ymin=0, ymax=1, color='black')

    plt.legend()

    plt.title(f"LVK Observing Run {run}", fontsize='x-large')
    plt.xlabel(rf"Days to first discoverable KN ($D_{{KN}}^{{{run_label}}}$)", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    #plt.xlim(0, max(df["O5 Days to KN"]))

    #plt.show()
    plt.savefig(f'O{run}_KN_dist.pdf', bbox_inches = "tight")
    plt.close()


def plot_days_to_KN_discovery_cumulative(df, run, color):

    data = df[f"O{run} Days to KN"]

    run_label = f"O{run}"

    plt.hist(data, density=True, histtype='step', fill=False, bins=25, alpha=1, color=color, cumulative=True)
    plt.hist(data, density=True, histtype='stepfilled', fill=True, bins=25, alpha=0.5, color=color, cumulative=True)

    plt.tight_layout()
    plt.grid()

    #plt.title(f"LVK Observing Run {run}", fontsize='x-large')
    plt.xlabel(rf"Days to first discoverable KN ($D_{{KN}}^{{{run_label}}}$)", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    #plt.xlim(0, max(df["O5 Days to KN"]))

    #plt.show()
    plt.savefig(f'O{run}_KN_cumulative.pdf', bbox_inches = "tight")
    plt.close()

def plot_delta_days_distribution(df, color):

    O4_kn_days, O5_kn_days = df["O4 Days to KN"], df["O5 Days to KN"]
    data = O4_kn_days - O5_kn_days

    data_med = int(np.median(data))
    data_5 = int(np.percentile(data, 5))
    data_95 = int(np.percentile(data, 95))

    plt.hist(data, density=True, histtype='step', fill=False, bins=25, alpha=1, color=color)
    plt.hist(data, density=True, histtype='stepfilled', fill=True, bins=25, alpha=0.5, color=color)

    plt.axvline(data_5, label=r"$P_{5\%}=$" + f"{data_5}", linestyle='dotted', ymin=0, ymax=1, color='black')
    plt.axvline(data_med, label=r"$P_{50\%}=$" + f"{data_med}", linestyle='dashed', ymin=0, ymax=1, color='black')
    plt.axvline(data_95, label=r"$P_{95\%}=$" + f"{data_95}", linestyle='dotted', ymin=0, ymax=1, color='black')
    plt.axvline(730, label=r"2 year shutdown", linestyle='dashed', ymin=0, ymax=1, color='red')

    plt.tight_layout()
    plt.legend()

    plt.xlabel(r"Difference in days to first discoverable KN ($\Delta D_{KN}$)", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    #plt.xlim(0, max(df["O5 Days to KN"]))

    #plt.show()
    plt.savefig("delta_t_distribution.pdf", bbox_inches = "tight")
    plt.close()

def plot_delta_days_cumulative(df, color):

    O4_kn_days, O5_kn_days = df["O4 Days to KN"], df["O5 Days to KN"]
    data = O4_kn_days - O5_kn_days

    data_med = int(np.median(data))
    data_5 = int(np.percentile(data, 5))
    data_95 = int(np.percentile(data, 95))

    plt.hist(data, density=True, histtype='step', fill=False, bins=25, alpha=1, color=color, cumulative=True)
    plt.hist(data, density=True, histtype='stepfilled', fill=True, bins=25, alpha=0.5, color=color, cumulative=True)

    plt.axvline(730, label=r"2 year shutdown", linestyle='dashed', ymin=0, ymax=1, color='red')

    plt.tight_layout()
    plt.grid()

    plt.xlabel(r"Difference in days to first discoverable KN ($\Delta D_{KN}$)", fontsize='x-large')
    plt.ylabel(r"Density", fontsize='x-large')

    #plt.xlim(0, max(df["O5 Days to KN"]))

    #plt.show()
    plt.savefig("delta_t_cumulative.pdf", bbox_inches = "tight")
    plt.close()


def rates_vs_days_to_kn(df):

    rates = df["Rates"]
    o4_days = df[f"O4 Days to KN"]
    o5_days = df[f"O5 Days to KN"]

    plt.scatter(rates, o4_days, marker='.', alpha=0.5, label='O4', color='C4')
    plt.scatter(rates, o5_days, marker='.', alpha=0.5, label='O5', color='C2')

    plt.legend()

    plt.xlabel(r"BNS Merger Rate ($Gpc^{-3} \cdot yr^{-1}$)", fontsize='x-large')
    plt.ylabel(r"Days to first discoverable KN ($D_{KN}$)", fontsize='x-large')
    
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

    plt.hist(data, density=True, histtype='step', fill=False, bins=25, alpha=1, color="C0")
    plt.hist(data, density=True, histtype='stepfilled', fill=True, bins=25, alpha=0.5, color="C0")

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
    plot_days_to_KN_discovery_cumulative(df, "4", "C4")
    #plot_days_to_GW_discovery_distribution(df, "4", "C4")

    plot_days_to_KN_discovery_distribution(df, "5", "C2")
    plot_days_to_KN_discovery_cumulative(df, "5", "C2")
    #plot_days_to_GW_discovery_distribution(df, "5", "C2")

    plot_delta_days_distribution(df, 'gray')
    plot_delta_days_cumulative(df, 'gray')

    rates_vs_days_to_kn(df)
    rates_vs_delta_days_to_kn(df)
    bns_rates_distribution(df)