import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

def Abbott23Rate(n):

    rate_mean = 105.5
    rate_95 = rate_mean + 190.2
    rate_5 = rate_mean - 83.9

    z_95 = 1.645
    z_5 = -1.645

    x = np.arange(-1e5,  1e5, 0.1)

    sigma1 = (rate_95 - rate_mean) / z_95
    sigma2 = (rate_5 - rate_mean) / z_5

    # Computing PDF
    A = (2/np.pi)**0.5 * 1/(sigma1 + sigma2)

    f_x_low = A * np.exp(-(x - rate_mean)**2/(2*sigma1**2))
    f_x_high = A * np.exp(-(x - rate_mean)**2/(2*sigma2**2))

    # Set negative rate probabilities to zero and re normalize to make sure sum of PDF is 1
    f_x = np.where(x < rate_mean, f_x_low, f_x_high)
    f_x = np.where(x < 0, 0, f_x)
    f_x = f_x / np.sum(f_x)

    # Find the CDF and build interpolator
    cdf = np.cumsum(f_x)
    inv_cdf = interpolate.interp1d(cdf, x)

    # Generate samples
    r = np.random.rand(n)
    rvs = inv_cdf(r)

    return rvs, x, f_x
    
def Nitz21Rate(n):
    

    # Constants needed
    rate_mean = 200
    rate_95 = rate_mean + 309
    rate_5 = rate_mean - 148

    x = np.arange(-1e5,  1e5, 0.1)

    z_95 = 1.645
    z_5 = -1.645

    sigma1 = (rate_95 - rate_mean) / z_95
    sigma2 = (rate_5 - rate_mean) / z_5

    # Computing PDF
    A = (2/np.pi)**0.5 * 1/(sigma1 + sigma2)

    f_x_low = A * np.exp(-(x - rate_mean)**2/(2*sigma1**2))
    f_x_high = A * np.exp(-(x - rate_mean)**2/(2*sigma2**2))

    # Set negative rate probabilities to zero and re normalize to make sure sum of PDF is 1
    f_x = np.where(x < rate_mean, f_x_low, f_x_high)
    f_x = np.where(x < 0, 0, f_x)
    f_x = f_x / np.sum(f_x)

    # Find the CDF and build interpolator
    cdf = np.cumsum(f_x)
    inv_cdf = interpolate.interp1d(cdf, x)

    # Generate samples
    r = np.random.rand(n)
    rvs = inv_cdf(r)

    return rvs, x, f_x

if __name__ == '__main__':
    n = 1
    r1, x1, fx1 = Abbott23Rate(n)
    r2, x2, fx2 = Nitz21Rate(n)
    print(r1[0])

    plt.plot(x1, fx1, label=r"Abbott - $105.5^{+190.2}_{-83.9}$")
    plt.plot(x2, fx2, label=r"Nitz - $200^{+309}_{-148}$")
    plt.xlabel(r'Rate (R, $GPc^{-3} yr^{-1}$)')
    plt.ylabel('P(R)')
    plt.xlim(left = 0, right = 1000)
    plt.legend()
    plt.show()

    bins = np.arange(0, max(max(r1), max(r2)), 10)
    plt.hist(r1, label=r"Abbott - $105.5^{+190.2}_{-83.9}$", histtype='step', bins=bins)
    plt.hist(r2, label=r"Nitz - $200^{+309}_{-148}$",  histtype='step',  bins=bins)
    plt.xlabel(r'Rate (R, $GPc^{-3} yr^{-1}$)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
