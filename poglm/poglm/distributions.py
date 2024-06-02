import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod

from poglm import utils


def poisson_log_likelihood(x: torch.FloatTensor, lam: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    # lam = mean
    ll = x * (lam+eps).log() - lam - torch.lgamma(x+1)
    return ll


def categorical_log_likelihood(x: torch.FloatTensor, ln_p: torch.FloatTensor) -> torch.FloatTensor:
    n_categories = ln_p.shape[-1]
    suppressed_x = x.clone().to(torch.int64)
    suppressed_x[suppressed_x >= n_categories] = n_categories - 1
    ll = torch.sum(ln_p * F.one_hot(suppressed_x, n_categories), dim=(-1,))
    return ll
    

def exponential_log_likelihood(x: torch.FloatTensor, scale: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    # scale = 1 / \lambda = mean
    ll = -(scale+eps).log() - x / (scale+eps)
    return ll
    

def gumbel_softmax_log_likelihood(x: torch.FloatTensor, ln_p: torch.FloatTensor, tau: float, eps=0) -> torch.FloatTensor:
    n_categories = ln_p.shape[-1]
    ln_x = (x + eps).log()
    ll = torch.lgamma(torch.tensor(n_categories)) + (n_categories - 1) * torch.tensor(tau).log() - n_categories * torch.logsumexp(ln_p - tau * ln_x, dim=-1) + (ln_p - (tau+1) * ln_x).sum(dim=-1)
    return ll


def rayleigh_log_likelihood(x: torch.FloatTensor, sigma: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    # mean = sigma * \sqrt(\pi / 2)
    ll = (x+eps).log() - 2*(sigma+eps).log() - x**2 / (2 * (sigma+eps)**2)
    return ll


def half_normal_log_likelihood(x: torch.FloatTensor, sigma: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
    # mean = sigma * \sqrt(2 / \pi)
    ll = torch.log(torch.tensor(2.0))/2 - (sigma+eps).log() - torch.log(torch.tensor(np.pi))/2 - x**2 / (2 * (sigma+eps)**2)
    return ll


class Distribution(object):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def log_likelihood(self, x: torch.FloatTensor, params: torch.FloatTensor) -> torch.FloatTensor:
        pass
    
    @abstractmethod
    def sample(self, params: torch.FloatTensor, rng=None) -> torch.FloatTensor:
        pass

    @abstractmethod
    def mean_to_params(self, mean: torch.FloatTensor) -> torch.FloatTensor:
        pass


class Poisson(Distribution):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Poisson'
    
    def log_likelihood(self, x: torch.FloatTensor, lam: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        return poisson_log_likelihood(x, lam, eps=eps)
        
    def sample(self, lam: torch.FloatTensor, rng=None, eps: float = 1e-8) -> torch.FloatTensor:
        return torch.poisson(lam + eps, generator=rng)
    
    def mean_to_params(self, mean: torch.FloatTensor) -> torch.FloatTensor:
        return mean
        

class Categorical(Distribution):
    def __init__(self, n_categories: int) -> None:
        super().__init__()
        self.name = 'categorical'
        self.n_categories = n_categories
        
    def log_likelihood(self, x: torch.FloatTensor, ln_p: torch.FloatTensor) -> torch.FloatTensor:
        return categorical_log_likelihood(x, ln_p)
        
    def sample(self, ln_p: torch.FloatTensor, rng=None) -> torch.FloatTensor:
        return torch.multinomial(ln_p.exp().reshape(-1, self.n_categories), num_samples=1, generator=rng).reshape(ln_p.shape[:-1])
    
    def mean_to_params(self, mean: torch.FloatTensor) -> torch.FloatTensor:
        return utils.poisson_to_categorical(mean, self.n_categories)


class Exponential(Distribution):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'exponential'
        
    def log_likelihood(self, x: torch.FloatTensor, scale: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        return exponential_log_likelihood(x, scale, eps=eps)
        
    def sample(self, scale: torch.FloatTensor, rng=None) -> torch.FloatTensor:
        u = torch.rand(scale.shape, generator=rng)
        return -scale * (1 - u).log()
    
    def mean_to_params(self, mean: torch.FloatTensor) -> torch.FloatTensor:
        return mean


class GumbelSoftmax(Categorical):
    def __init__(self, n_categories: int, tau: float) -> None:
        super().__init__(n_categories)
        self.name = 'Gumbel-Softmax'
        self.tau = tau # tau >= 0.5 is important, tau is better to be between 0.5 and 1.
        
    def log_likelihood(self, x: torch.FloatTensor, ln_p: torch.FloatTensor, eps=0) -> torch.FloatTensor: # eps = 0 is important, tau >= 0.5 is enough to promise x is elligible.
        return gumbel_softmax_log_likelihood(x, ln_p, self.tau, eps=eps)
    
    def sample(self, ln_p: torch.FloatTensor, rng=None) -> torch.FloatTensor:
        return F.gumbel_softmax(ln_p, tau=self.tau)
        
        
class Rayleigh(Distribution):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Rayleigh'
    
    def log_likelihood(self, x: torch.FloatTensor, sigma: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        return rayleigh_log_likelihood(x, sigma, eps=eps)
    
    def sample(self, sigma: torch.FloatTensor, rng=None) -> torch.FloatTensor:
        u = torch.rand(sigma.shape, generator=rng)
        return sigma * (-(1 - u).log() * 2).sqrt()
    
    def mean_to_params(self, mean: torch.FloatTensor) -> torch.FloatTensor:
        return mean * torch.sqrt(torch.tensor(2.) / torch.tensor(np.pi))

class HalfNormal(Distribution):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'half-normal'
    
    def log_likelihood(self, x: torch.FloatTensor, sigma: torch.FloatTensor, eps: float = 1e-8) -> torch.FloatTensor:
        return half_normal_log_likelihood(x, sigma, eps=eps)
        
    def sample(self, sigma: torch.FloatTensor, rng=None) -> torch.FloatTensor:
        u = torch.randn(sigma.shape, generator=rng)
        return sigma * torch.abs(u)
    
    def mean_to_params(self, mean: torch.FloatTensor) -> torch.FloatTensor:
        return mean * torch.sqrt(torch.tensor(np.pi) / torch.tensor(2.))