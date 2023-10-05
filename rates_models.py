import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as stats

def LVK_UG(n):
    # BNS merger rate for LVK user guide
    rate_mean = np.log10(210)
    rate_95 = np.log10(210 + 240)
    rate_5 = np.log10(210 - 120)

    z_95 = 1.6449
    z_5 = -1.6449

    sigma1 = (rate_95 - rate_mean) / z_95
    sigma2 = (rate_5 - rate_mean) / z_5

    d = stats.norm(loc=rate_mean, scale=sigma1) 
    sample_exponents = d.rvs(size=n)

    sample_rates = 10**sample_exponents

    return sample_rates, d


if __name__ == '__main__':

    n = 100000
    rate_exponents = np.arange(1e-3, 1e3, 0.01)
    rates = 10**rate_exponents

    samples, d = LVK_UG(n)


    fx = d.pdf(rate_exponents)
    Fx = d.cdf(rate_exponents)
    r_5 = 10**d.ppf(0.05)
    r_95 = 10**d.ppf(0.95)
    print(r_5, r_95)

    plt.plot(rates, fx, label=r"PDF")
    #plt.plot(rates, Fx, label=r"CDF")
    plt.axvline(x=r_5, label=r'$\langle R \rangle_{5} = %.2f$' % (r_5))
    plt.axvline(x=r_95, label=r"$\langle R \rangle_{95} = %.2f$" % (r_95))
    plt.xlabel(r'Rate (R, $GPc^{-3} yr^{-1}$)')
    plt.ylabel('P(R)')
    plt.xscale('log')
    plt.xlim(left = 1, right = 10000)
    plt.legend()
    plt.show()

    s_5 = np.percentile(samples, 5)
    s_95 = np.percentile(samples, 95)
    bins = np.arange(0, max(samples), 10)

    plt.hist(samples,  histtype='step',  bins=bins)

    plt.axvline(x=r_5, label=r'$\langle R \rangle_{5} = %.2f$' % (s_5), c ='black', linestyle='--')
    plt.axvline(x=r_95, label=r"$\langle R \rangle_{95} = %.2f$" % (s_95), c ='black', linestyle='--')

    plt.xlabel(r'Rate (R, $GPc^{-3} yr^{-1}$)', fontsize='x-large')
    plt.ylabel('Count', fontsize='x-large')
    plt.legend()

    plt.xscale('log')
    plt.tight_layout()

    plt.show()
