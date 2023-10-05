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

    x = np.arange(-100,  100, 0.05)

    sigma1 = (rate_95 - rate_mean) / z_95
    sigma2 = (rate_5 - rate_mean) / z_5

    # Computing PDF
    A = (2/np.pi)**0.5 * 1/(sigma1 + sigma2)

    f_x_low = A * np.exp(-(x - rate_mean)**2/(2*sigma1**2))
    f_x_high = A * np.exp(-(x - rate_mean)**2/(2*sigma2**2))

    # Set negative rate probabilities to zero and re normalize to make sure sum of PDF is 1
    f_x = np.where(x < rate_mean, f_x_low, f_x_high)
    f_x = f_x/np.sum(f_x)


    # Find the CDF and build interpolator
    cdf = np.cumsum(f_x)
    inv_cdf = interpolate.interp1d(cdf, x)

    # Generate samples
    r = np.random.rand(n)
    rvs = inv_cdf(r) # exponents of the rates
    sampled_rates = 10**rvs
    all_rates = 10**x

    return sampled_rates, all_rates, f_x


if __name__ == '__main__':

    n = 10000
    r3, x3, fx3 = LVK_UG(n)


    plt.plot(x3, fx3, label=r"LVK user guide")
    plt.xlabel(r'Rate (R, $GPc^{-3} yr^{-1}$)')
    plt.ylabel('P(R)')
    plt.xscale('log')
    plt.xlim(left = 1, right = 10000)
    plt.legend()
    plt.show()

    d = stats.norm(
    loc=2.322219294733919, 
    scale= 0.2012239157647422  #  0.2012239157647422 
    ) 
    x = np.arange(1e-3, 1e3, 0.01)
    fx = d.pdf(x)
    print(10**d.ppf(0.05), 10**d.ppf(0.95))

    plt.plot(10**x, d.cdf(x), label=r"LVK user guide")
    plt.xlabel(r'Rate (R, $GPc^{-3} yr^{-1}$)')
    plt.ylabel('P(R)')
    plt.xscale('log')
    plt.xlim(left = 1, right = 10000)
    plt.legend()
    plt.show()


    bins = np.arange(0, max(max(r3), max(r3)), 10)
    plt.hist(r3, label=r"LVK User guide",  histtype='step',  bins=bins)
    plt.hist(10**d.rvs(size=n), label=r"LVK User guide",  histtype='step',  bins=bins)
    plt.xlabel(r'Rate (R, $GPc^{-3} yr^{-1}$)')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.legend()
    plt.show()

    #print(np.percentile(r3, 5), np.percentile(r3, 95))