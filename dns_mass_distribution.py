import matplotlib.pyplot as plt
import numpy as np

def extra_galactic_masses(n):
    # https://iopscience.iop.org/article/10.3847/2041-8213/abe7f6/pdf

    # Stats for double gaussian distribution of Recycled stars
    m1_recycled = 1.34
    sd1_recycled = 0.02

    m2_recycled = 1.47
    sd2_recycled = 0.15

    f_recycled = 0.68

    # Stats for double gaussian distribution of slow stars
    m1_slow = 1.29
    sd1_slow = 0.09

    m2_slow = 1.8
    sd2_slow = 0.15

    f_slow = 0.5

    # Sampling from the two gaussian for recycled stars according to the weight fraction
    m_recycled_1 = np.random.normal(loc=m1_recycled, scale=sd1_recycled, size = int(f_recycled * n))
    m_recycled_2 = np.random.normal(loc=m2_recycled, scale=sd2_recycled, size = n - int(f_recycled * n))
    m_recycled = np.concatenate((m_recycled_1, m_recycled_2))
    np.random.shuffle(m_recycled)


    # Sampling from the two gaussian for slow stars according to the weight fraction
    m_slow_1 = np.random.normal(loc=m1_slow, scale=sd1_slow, size = int(f_slow * n))
    m_slow_2 = np.random.normal(loc=m2_slow, scale=sd2_slow, size = n - int(f_slow * n))
    m_slow = np.concatenate((m_slow_1, m_slow_2))
    np.random.shuffle(m_slow)

    return m_recycled, m_slow

def galactic_masses(n):
    # https://iopscience.iop.org/article/10.3847/1538-4357/ab12e3/pdf

    mean = 1.33
    std = 0.09

    m1 = np.random.normal(loc=mean, scale=std, size = n)
    m2 = np.random.normal(loc=mean, scale=std, size = n)

    np.random.shuffle(m1)
    np.random.shuffle(m2)

    return m1, m2


if __name__ == '__main__':

    m1, m2 = extra_galactic_masses(10000000)

    fig, ax = plt.subplots(2)

    ax[0].hist(m1, label='Recycled', density=True, alpha = 0.5,  bins=100)
    ax[0].set_xlabel('Mass (M_sun)')
    ax[0].set_ylabel('p(mass)')
    ax[0].legend()

    ax[1].hist(m2, label='Slow', density=True, alpha = 0.5, bins=100)
    ax[1].set_xlabel('Mass (M_sun)')
    ax[1].set_ylabel('p(mass)')
    ax[1].legend()
    
    plt.show()

    m1, m2 = galactic_masses(10000)

    plt.hist(m1, label='m1', density=True, alpha = 0.5,  bins=100, histtype='step')
    plt.hist(m2, label='m2', density=True, alpha = 0.5,  bins=100, histtype='step')
    plt.xlabel('Mass (M_sun)')
    plt.ylabel('p(mass)')
    plt.legend()
    plt.show()
