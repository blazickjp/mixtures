"""
Module Doc Strings
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from numpy.random import multinomial, normal
from scipy.stats import invgamma, norm, multivariate_normal, dirichlet, multivariate_t
from distcan import InverseGamma
from matplotlib.lines import Line2D


class FiniteGMM:
    """
    Creates a FiniteGMM class for fitting Univariate and Multivariate Gaussian
    Mixture Models.

    Args:
        k (int): The number of Gaussians in the mixture
        mu (np.array): The mean of the Gaussians
        sigma (np.array): The covariance of the Gaussians
        phi (np.array): The mixing coefficients
    Returns:
        FiniteGMM: A FiniteGMM object
    """
    def __init__(self, k = None, mu = None, sigma = None, phi = None):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.multivariate = np.array(self.sigma).ndim >= 2
        self.data = np.NaN
        self.results = None
        self.fitted_params = None

    def data_gen(self, n):
        """
        Generates samples from Mixture of K Gaussian Distributions. Method will generate multivariate
        or univariate data depending on the shape of the sigma parameter. Data is assigned to class attribute "data"

        Args:
            N (int): The number of samples you want to generate
        Returns:
            None, but assigns data to class attribute "data"
        """
        try:
            y = np.empty((n, np.array(self.mu).shape[1]))
        except IndexError:
            y = np.empty(n)
        z = []
        for i in range(n):
            ind = multinomial(1, self.phi)
            for j, val in enumerate(ind):
                if val == 1:
                    if self.multivariate:
                        z.append(j)
                        y[i,:] = np.random.multivariate_normal(self.mu[j,:], self.sigma[j,:,:])
                    else:
                        y[i] = norm(self.mu[j], self.sigma[j]).rvs()
        self.data = np.array(y)
        return
    
    def update_pi(self, alpha_vec, z_vec):
        """
        Sample from Posterior Conditional for pi
        """
        return dirichlet(z_vec + alpha_vec).rvs()

    def update_mu(self, z_mat, sigma_vec):
        """
        Sample from Posterior Conditional for mu.
        Args:
            y:
            z_mat:
            sigma_vec:
        Returns:
            arra
        """
        mu_vec = []
        n_j =  np.sum(z_mat, axis=0)
        for j, sig in enumerate(sigma_vec):
            sigma_vec[j] = sig / (n_j[j] + 1)
            mu_vec.append(np.sum(self.data * z_mat[:,j]) / (n_j[j] + 1))
        
        cov = np.diag(sigma_vec)
        return multivariate_normal(mu_vec, cov).rvs()
    
    def update_sigma(self, z_mat, mu):
        """
        Sample from Posterior Conditional for sigma
        """
        n_j = np.sum(z_mat, axis=0)
        alpha = (0.5 * n_j) + 1
        beta = []
        for j in range(len(mu)):
            y = self.data * z_mat[:,j]
            y = y[y != 0]
            beta.append((0.5 * np.square(y - mu[j]).sum()) + 1)
        return InverseGamma(alpha, beta).rvs()

    def update_z(self, mu, sigma, pi):
        """
        Sample from latent variable Z according to likelihoods for class assignment
        """
        a = np.empty((len(self.data), len(mu)))
        out = np.empty((len(self.data), len(mu)))
        for j in range(len(mu)):
            a[:,j] = norm(mu[j], np.sqrt(sigma[j])).pdf(self.data) * pi[0,j]
        
        pi_i = a / np.sum(a, axis=1)[:,None]
        for i in range(len(self.data)):
            out[i,] = multinomial(1, pi_i[i,:])
        return out

    def gibbs(self, iters=10, burnin=5):
        """
        Run Gibb's Sampling for Mixture of Gaussians. Initial States are sample from Priors.
        Will run Collapsed Gibbs in the case where self.data is multivariate data.
        Args:
            iters (int): Number of iterations you would like to run
            burnin (int): Number of iterations to discard when calculating parameters
        Returns:
            Params (dict): Parameters of fitted distributions
        """

        if self.multivariate:
            self.collapsed_gibbs(iters = iters)
            self.params = self.__get_params()
        else:
            # Set initial guesses based on priors
            alpha = [1 for i in range(self.k)]
            mu = normal(0, 1, size=self.k)
            pi = dirichlet(alpha).rvs()
            sigma = InverseGamma(1,1).rvs(size=self.k)
            out = np.empty((iters, self.k*3))

            for i in range(iters):
                # Update Parameters according to conditional posterior distributions
                z_mat = self.update_z(mu, sigma, pi)
                pi = self.update_pi(alpha, np.sum(z_mat, axis=0))
                mu = self.update_mu(z_mat, sigma)
                sigma = self.update_sigma(z_mat, mu)

                # Store Values to monitor trace
                out[i, 0:self.k] = mu
                out[i, self.k:2*self.k] = np.sqrt(sigma)
                out[i, 2*self.k:3*self.k] = pi[0,:]
            self.results = out[burnin:,:]
            return self.__get_params()

    def collapsed_gibbs(self, a = 1, v = 10, iters = 10):
        """
        Runs collapsed gibbs sampling on self.data
        """
        N = self.data.shape[0]
        D = self.data.shape[1]
        assert v > D-1, "v must be greater than D-1"
        alpha = np.repeat(a, self.k)
        z = np.random.choice(self.k, size = N, replace = True, p =dirichlet(alpha / self.k).rvs().squeeze())
        for _ in range(iters):
            for i in range(N):
                # Remove x_i from data and Z
                d2 = np.delete(self.data, i, axis=0)
                z2 = np.delete(z, i, axis=0)
                p_z = []
                for k in range(self.k):
                    mu_k = np.mean(d2[z2 == k], axis=0)
                    # cov_k = np.cov(d2[z2 == k], rowvar=False)
                    n_k = np.sum(z2 == k)
                    p_z_k = (n_k + a/self.k) / (N + a - 1)
                    S_k = np.dot(np.transpose(d2[z2 == k] - mu_k), d2[z2 == k] - mu_k) + np.eye(D)
                    p_x_i = multivariate_t(mu_k, ((n_k+1) / (n_k *(n_k + v - D + 1)))*S_k, n_k+v - D + 1).pdf(self.data[i,])
                    p_z_k = p_z_k * p_x_i
                    p_z.append(p_z_k)
                # Standardize prob vector p(z_i = k)
                p_z = p_z / np.sum(p_z)
                z[i] = np.random.choice(self.k, 1, replace=True, p = p_z)
        self.results = z
        return
    
    def __get_params(self, digits = 2):
        if self.results is None:
            Exception("Can't get parameters before fitting model")

        if self.multivariate:
            mu = []; sigma = []; phi = []
            for i in range(self.k):
                data = self.data[self.results == i, :]
                mu.append(data.mean(axis=0))
                sigma.append(np.cov(data, rowvar=False))
                phi.append(self.results == i / self.data.shape[0])

            self.fitted_params = {
                "mu": mu,
                "sigma": sigma,
                "phi": phi
            }
            return self.fitted_params

        else:
            res = self.results.mean(axis=0)
            params_dict = {
                "mu": np.round(res[0:self.k], digits),
                "sigma": np.round(res[self.k:self.k*2], digits),
                "pi": np.round(res[self.k*2:self.k*3], digits)
            }
            self.fitted_params = params_dict
            return params_dict

    def plot_data(self, **kwargs):
        """
        Plots the original data as histogram for univariate and scatter plot for multivariate data
        
        Args:
            kwargs: Keyword arguments to be passed to matplotlib.pyplot.hist() for univariate data
            or matplotlib.pyplot.scatter() for multivariate data
        """
        if type(self.data) != np.ndarray:
            raise ValueError("You must generate data first!")
        
        if self.multivariate:
            x, y = np.mgrid[min(self.data[:,0])-1:max(self.data[:,0])+1:.1, min(self.data[:,1])-1:max(self.data[:,1])+1:.1]
            pos = np.dstack((x,y))
            fig, ax = plt.subplots()
            ax.scatter(self.data[:,0], self.data[:,1], **kwargs)
            if self.mu is not None and self.sigma is not None and self.k is not None:
                for i in range(len(self.mu)):
                    ax.contour(x,y, 
                        multivariate_normal(self.mu[i,:], 
                        self.sigma[i,:,:]).pdf(pos), 
                        extend='both')
                    fig.suptitle(f"K={self.k} Bivariate Gaussian Distributions Data")
            else:
                warnings.warn("No True Parameters Given... Just plotting the data")
                plt.title("Finite GMM Data")
            ax.grid()
            fig.show()

        else:
            x_data = np.linspace(min(self.data), max(self.data))
            plt.hist(self.data, density=True, **kwargs)
            if self.mu is not None and self.sigma is not None:
                for i in range(len(self.mu)):
                    plt.plot(x_data, norm(self.mu[i], self.sigma[i]).pdf(x_data), color=list(mcolors.TABLEAU_COLORS.values())[i])
                    plt.title(f"Mixture of {len(self.mu)} Gaussians Data")
            else:
                warnings.warn("No True Parameters Given... Just plotting the data")
                plt.title("Finite GMM Data")
            plt.grid()
            plt.show()

    def plot_results(self, **kwargs):
        if self.multivariate:
            fig, ax = plt.subplots()
            x, y = np.mgrid[min(self.data[:,0])-1:max(self.data[:,0])+1:.1, min(self.data[:,1])-1:max(self.data[:,1])+1:.1]
            pos = np.dstack((x,y))
            cols1 = [list(mcolors.TABLEAU_COLORS.values())[i] for i, v in enumerate(np.unique(self.results))]
            for i, v in enumerate(cols1):
                d2 = self.data[np.array(self.results) == i,:]
                ax.scatter(d2[:,0], d2[:,1], color=v, label = v, **kwargs)
                if i in range(3):
                    ax.contour(x,y, multivariate_normal(
                        self.fitted_params["mu"][i], 
                        self.fitted_params["sigma"][i]).pdf(pos), 
                        extend='both')

            fig.suptitle(f"Mixture of {self.k} Multivariate Gaussians")
            ax.grid()
            fig.legend()
            fig.show()
        else:
            x = np.linspace(min(self.data), max(self.data))
            plt.hist(self.data, density=True, **kwargs)
            for j in range(self.k):
                plt.plot(x, norm(self.mu[j], self.sigma[j]).pdf(x), color="red")
                plt.plot(x, norm(self.fitted_params["mu"][j], self.fitted_params["sigma"][j]).pdf(x), color="blue")
            plt.title(f"Mixture of {self.k} Gaussians")

            legend_elements = [
                Line2D([0], [0], color='blue', lw=4, label='Fitted'),
                Line2D([0], [0], color='red', lw=4, label='Actual')
            ]
            plt.legend(handles=legend_elements, loc="upper right")
            plt.grid()
            plt.show()

if __name__ == '__main__':
    pass
