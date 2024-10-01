import numpy as np
from math import sqrt, log, pi, cos, sin, acos, floor, ceil
from scipy.integrate import quad, trapezoid, cumulative_trapezoid

sqrt2pi = sqrt(2*pi)

def Gauss(x, mu=0.0, std=1.0):
    '''Gaussian pdf'''
    return np.exp(-0.5* ((x-mu)/std)**2) / (sqrt2pi * std)

def Unif(x, a:float, b:float):
    '''Uniform pdf between [a,b]'''
    return ((a <= x) & (x <= b)) / (b-a)

def Beta(x:np.ndarray, alpha:float, beta:float):
    '''Beta pdf (un-normalized)'''
    return np.power(x, alpha-1) * np.power(1 - x, beta-1)

def Cauchy(x, x0, a):
    '''Cauchy pdf'''
    return a / ((x - x0)**2 + a**2) / pi

def GaussMixture(x:np.ndarray, weights:np.ndarray, params:list[tuple]):
    '''Gaussian Mixture pdf'''
    # output y
    y = np.zeros_like(x)
    for i in range(len(weights)):
        y += weights[i] * Gauss(x, params[i][0], params[i][1])
    return y

def Gauss2D(x:np.ndarray, mu:np.ndarray, sigma:np.ndarray):
    '''
    x: 2D points to evaluate on, shape=(N,2)
    mu: mean parameters, shape=(2,)
    sigma: covariance matrix, shape=(2,2)
    '''
    norm_cst = sqrt2pi * sqrt(np.linalg.det(sigma))
    diff = x-mu # (N,2)
    # Do this: (1,2) @ (2,2) @ (2,1) -> (1,1) for every sample from i=1 to N
    return np.exp(-0.5 * np.einsum('nj,jk,nk->n', diff, np.linalg.inv(sigma), diff)) / norm_cst

    #Equivalent
    temp = diff @ np.linalg.inv(sigma) # (N,2)
    return np.sum(temp * diff, axis=1) # np.sum((N,2) * (N,2), axis=1) -> (N,)


def InverseSampling1D(f, a:float, b:float, N:int, n:int):
    '''Generates N samples from a given density f whose support is [a,b].'''
    # Evaluate f between a, b on n points
    x = np.linspace(a, b, n)
    y = f(x)
    # cdf evaluated on the points x
    cdf_values = cumulative_trapezoid(y, x)
    #Normalize
    cdf_values /= cdf_values[-1]

    U = np.random.rand(N)
    # approximate inverse cdf
    generated_samples = x[np.searchsorted(cdf_values, U)]

    return generated_samples

def get_density_points(x:np.ndarray, width=0.02):
    '''From samples x, computes the frequency for every bin of a given width in [x.min(), x.max()].
    Used for a scatter plot that ressembles the density of the generated samples
    '''
    xmin, xmax = x.min(), x.max()

    bin_edges = np.arange(xmin, xmax+width, step=width).reshape(1,-1)
    n_bins = ceil((xmax-xmin)/width)
    y_bins = np.zeros(shape=(n_bins,))

    indexes = (bin_edges > x.reshape(-1,1)).argmax(axis=-1) - 1

    y_bins += np.bincount(indexes) * width
    x_bins = (bin_edges + (width/2.))[0,:-1]
    # Integral over support set
    I = np.trapz(y_bins, dx=width)
    # Normalize
    y_bins /= I

    return x_bins, y_bins





if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from functools import partial

    # Choosing a pdf

    # Beta distribution:
    # Beta_with_params = partial(Beta, alpha=5.0, beta=1.0)
    # a, b = 1e-16, 1.0 - 1e-16

    # Gaussian distribution
    # pdf = partial(Gauss, mu=1.0, std=2.0)
    # a, b = -10.0, 10.0

    # Gaussian mixture: 
    weights = np.array([0.2, 0.8])
    params = [(-4.0, 1.0), (2.0, 2.0)]
    pdf = partial(GaussMixture, weights=weights, params=params)
    a, b = -10.0, 10.0 # edges of the support set of the pdf

    # Generate samples from pdf
    samples = InverseSampling1D(pdf, a=a, b=b, N=100000, n=10000)
    x_bins, y_bins = get_density_points(samples, width=0.01)

    # True density evaluated on 1000 points for plotting
    x = np.linspace(a, b, 1000)
    y = pdf(x)
    #Noramlize just in case
    y /= np.trapz(y, x)

    plt.plot(x, y, label="True density")
    plt.scatter(x_bins, y_bins, marker="+", color="grey", label="Generated samples")
    plt.legend()
    plt.show()
