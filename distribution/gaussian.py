import numpy as np


def get_mean(data):
    return np.mean(data)

def get_variance(data):
    return np.var(data)

def get_gaussian_likelihood(x, mean, var):
    # avoid var < 0 (too concentrate to same value)
    if var <= 0:
        var = 10
    return (1/np.sqrt(2*np.pi*var))*np.exp(((-1)*(x-mean)**2)/(2*var))