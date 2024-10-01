import numpy as np
from math import sqrt, log, pi, cos, sin, acos, floor, ceil
from scipy.integrate import quad, trapezoid, cumulative_trapezoid
from code import interact
import warnings
warnings.filterwarnings("ignore")

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

def Gauss2D(x1:np.ndarray, x2:np.ndarray|None, mu:np.ndarray, sigma:np.ndarray):
    '''
    x1: If x2 is provided, First coordinate of the points to evaluate on, shape=(N,)
        If x2 is None, 2D-array of the points to evaluate on, shape=(N,2)
    x2: Second coordinate of the points to evaluate on, shape=(N,)
    mu: mean parameters, shape=(2,)
    sigma: covariance matrix, shape=(2,2)
    '''
    norm_cst = 2*pi * sqrt(np.linalg.det(sigma))
    x = np.column_stack((x1, x2))
    diff = x-mu # (N,2)
    # Do this: (1,2) @ (2,2) @ (2,1) -> (1,1) for every sample from i=1 to N
    return np.exp(-0.5 * np.einsum('nj,jk,nk->n', diff, np.linalg.inv(sigma), diff)) / norm_cst

    #Equivalent
    temp = diff @ np.linalg.inv(sigma) # (N,2)
    return np.sum(temp * diff, axis=1) # np.sum((N,2) * (N,2), axis=1) -> (N,)


def InverseSampling1D(pdf, a:float, b:float, N:int, n_bins:int=1000, **pdf_kwargs):
    '''
    Generates N samples from a given density f whose support is [a,b].
    
    -----------
    Parameters
    - pdf: the callable 1D pdf function
    - a, b: Support edges. The support set is the segment [a,b]
    The pdf can be defined outside of this region as long as it is negligeable (i.e. near 0).
    - N: the number of generated samples.
    - n_bins: the number of points the pdf is evaluated on. The higher the more precise the cdf approximation will be, 
    but the more costly it will be.
    '''
    # Evaluate f between [a,b] on n points
    x = np.linspace(a, b, n_bins)
    dx = np.diff(x, 1).mean()
    y = pdf(x, **pdf_kwargs)

    # cdf evaluated on the points x. We use the cumulative_trapezoid function who computes a cumulative integral
    # from the evaluated points y with a incremental step of dx
    cdf_values = cumulative_trapezoid(y, dx=dx)
    #Normalize just in case
    cdf_values /= cdf_values[-1]

    # Generation step:
    # First step: generate a random uniform [0,1] array
    U = np.random.rand(N)
    # Second step: approximate inverse cdf, but we don't have the actual inverse cdf nor the cdf callable function.
    # Instead we look for the x[i] value such that F(x[i]) = U[i] (inversion problem)
    
    # The paper suggests using a bisection method which requires having the cdf function callable on any x_i
    # Here we only have the cdf evaluated at many points, so we use the searchsorted('array') function which 
    # for a given U[i] searches for the index j to insert U[i] into 'array' such that the order is maintained (sorted)
    # In other words, it gives the index of the closest F(x[i]) to U[i] 
    generated_samples = x[np.searchsorted(cdf_values, U)]

    return generated_samples

def get_density_points1D(x:np.ndarray, width=0.02):
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

def inverseSampling2D(pdf, edges:tuple, N:int, n_bins:int, **pdf_kwargs):
    # Support edges, [a,b] for the first coordinate, [c,d] for the second. The support set is the segment [a,b] x [c,d]
    # The pdf can be defined outside of this region as long as it is negligeable (i.e. near 0).
    a, b, c, d = edges

    # the incremental step size on which we evaluate the pdf, for example we evaluate the pdf on 0.01, 0.01+dx, 0.01+2dx, ...
    x1s = np.linspace(a, b, n_bins)
    x2s = np.linspace(c, d, n_bins)
    dx1, dx2 = np.diff(x1s, 1).mean(), np.diff(x2s, 1).mean()
    X1s, X2s = np.meshgrid(x1s, x2s, indexing="ij") # cartesian product
    X1s, X2s = X1s.ravel(), X2s.ravel() #flatten

    #pdf_values contains the pdf evaluated on many points (roughly the cartesian product [a,b] x [c,d])
    pdf_values = pdf(x1=X1s, x2=X2s, **pdf_kwargs).reshape(n_bins, n_bins)
    # integrate along the second coordinate with the trapezoid method. This results in the marginal density f(x1) evaluated on the x1s
    f_x1s = np.trapz(pdf_values, dx=dx2, axis=1)

    # cdf evaluated on the points x1s. We use the cumulative_trapezoid function that computes a cumulative integral
    # from the evaluated points f_x1s with a incremental step of dx
    cdf_values_x1 = cumulative_trapezoid(f_x1s, dx=dx1)
    # normalize jsut in case
    cdf_values_x1 /= cdf_values_x1[-1]

    # Generate the first coordinate: First generate a random uniform [0,1] array
    U = np.random.rand(N)
    # Then we look for the x[i] values such that F(x[i]) = U[i] (inversion problem)
    # The paper suggests using a bisection method which requires having the cdf function callable on any x_i
    # Here we only have the cdf evaluated at many points, so we use the searchsorted(array) function which 
    # for a given U[i] searches for the index j to insert U[i] into 'array' such that the order is maintained (sorted)
    # Basically, it gives the index of the closest F(x[i]) (that we already computed) to U[i]
    indexes = np.searchsorted(cdf_values_x1, U)
    samples = np.zeros(shape=(N, 2))
    samples[:,0] = x1s[indexes]

    # Second coordinate generation:
    for i in range(N):
        def conditional_pdf(x2):
            return pdf(x1=np.ones(n_bins)*samples[i,0], x2=x2, **pdf_kwargs) / f_x1s[indexes[i]]
        samples[i,1] = InverseSampling1D(conditional_pdf, c, d, N=1, n_bins=n_bins).item()

    return samples



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from functools import partial

    _1D = False
    _2D = True

    if _1D:
        '''1D Example'''
        # Choosing a pdf

        # Beta distribution:
        # pdf = partial(Beta, alpha=5.0, beta=1.0)
        # a, b = 1e-16, 1.0 - 1e-16

        # Gaussian distribution
        # pdf = partial(Gauss, mu=1.0, std=2.0)
        # a, b = -10.0, 10.0

        # Gaussian mixture: 
        weights = np.array([0.2, 0.8])
        params = [(-4.0, 1.0), (2.0, 2.0)]
        pdf = partial(GaussMixture, weights=weights, params=params)
        # Alternatively:
        # kwargs = {"weights":weights, "params":params}
        # pdf = GaussMixture
        a, b = -10.0, 10.0 # edges of the support set of the pdf

        # Generate samples from pdf
        samples = InverseSampling1D(pdf, a=a, b=b, N=100000, n_bins=10000)

        # For plotting
        x_bins, y_bins = get_density_points1D(samples, width=0.01)
        plt.scatter(x_bins, y_bins, marker="+", color="grey", label="Generated samples")

        # True density evaluated on 1000 points for plotting
        x = np.linspace(a, b, 1000)
        y = pdf(x)
        y /= np.trapz(y, x) #Noramlize just in case
        plt.plot(x, y, label="True density")
        plt.legend()
        plt.show()

    elif _2D:
        '''2D example'''
        mu = np.array(
            [+3.0, 
             -4.0]
        )
        sigma = np.array(
            [[5.0, -3.0], 
             [-3.0, 3.0]]
        )
        N = 1000 # number of samples to be generated, keep it low
        n_bins = 1000 # higher -> higher precision, but slower in computation
        pdf = Gauss2D # our pdf
        kwargs = dict(mu=mu, sigma=sigma) # pdf kwargs
        #support edges
        a,b,c,d = (mu[0]-3*sigma[0,0], mu[0]+3*sigma[0,0], mu[1]-3*sigma[1,1], mu[1]+3*sigma[1,1])
        samples = inverseSampling2D(pdf, edges=(a,b,c,d), N=N, n_bins=n_bins, **kwargs)

        x = np.linspace(a,b,n_bins)
        y = np.linspace(c,d,n_bins)
        X,Y= np.meshgrid(x,y)
        values = pdf(x1=X.ravel(), x2=Y.ravel(), **kwargs)
        Z = values.reshape(n_bins, n_bins) # (N,N)

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='Greens')
        plt.colorbar(label='pdf value')
        plt.scatter(samples[:,0], samples[:,1], label="generated", marker="+", linewidths=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D Gaussian pdf')
        plt.legend()
        plt.show()