import numpy as np
from scipy.integrate import quad as integrate
import matplotlib.pyplot as plt
import tqdm
import random
import torch as t

def relu(x):
    return (x + np.abs(x)) / 2

def alpha(k, d):
    return k**2 * (d + (k-2)*k) / d / (d - 1)

def variance(k, d, D, p):
    return (p/3) * (D - 1) * (4 * alpha(k, d) / k**2 -4 * k / d + 1) \
           - p**2/4 * (D-1) * ((2*k-d)/d)**2

def mean(k, d, D, p):
    return (p/2) * (D - 1) * (2 * k - d) / d

def G(x):
    cond1 = x > -1
    cond2 = x > 0

    return cond1 * (1-cond2) * (2 + 3 * x - x**3) / 6 + cond2 * (2 + 3 * x) / 6

def H(x):
    cond1 = x > -1
    cond2 = x > 0

    return cond1 * (1-cond2) * (1 + x)**3 / 3 + cond2 * (1 + 3 * x + 3 * x**2) / 3

def optimal_A(mu, sigma, p):
    numerator = integrate(lambda x: G(x) * np.exp(-(x-mu)**2 / (2 * sigma**2)), -1, np.inf)[0]
    denom_1 = integrate(lambda x: H(x) * np.exp(-(x-mu)**2 / (2 * sigma**2)), -1, np.inf)[0]
    denom_2 = integrate(lambda x: relu(x)**2 * np.exp(-(x-mu)**2 / (2 * sigma**2)), 0, np.inf)[0] #none are normalized, but we're taking a ratio

    return p*numerator / (p * denom_1 + (1-p) * denom_2), numerator / np.sqrt(2 * np.pi) / sigma

def variance(k, d, D, p):
    return (D/d - 1) * (p/3 - p**2/4)

def optimal_params(d, D, p):
    opt_k = 0
    opt_loss = np.inf
    opt_A = 0.0
    k = d // 2 + 1
    
    A, num = optimal_A(mean(k, d, D, p), variance(k, d, D, p)**.5, p) 
    l = p / 3 - A * p * num

    # if l < opt_loss:
    #     opt_k = k 
    #     opt_loss = l
    #     opt_A = A

    while opt_loss > l:
        opt_k = k 
        opt_loss = l
        opt_A = A

        k -= 1
        A, num = optimal_A(mean(k, d, D, p), variance(k, d, D, p)**.5, p) 
        l = p / 3 - A * p * num
        
    
    return opt_k, opt_A, opt_loss


def gen_optimal_matrices(D, d, p):
    k, A, _ = optimal_params(d, D, p)
    groups = [random.sample(range(d), k) for _ in range(D)]

    Win = np.zeros((d, D))
    Wout = -np.ones((D, d))

    for i, group in enumerate(groups):
        for g in group:
            Win[g, i] = 1/k
            Wout[i, g] = 1.0
    
    return Win, Wout * A

def gen_matrices(D, d, k, return_torch_tensor=False):
    groups = [random.sample(range(d), k) for _ in range(D)]

    Win = np.zeros((d, D))
    Wout = -np.ones((D, d))

    for i, group in enumerate(groups):
        for g in group:
            Win[g, i] = 1/k
            Wout[i, g] = 1.0
            
    if return_torch_tensor:
        return t.tensor(Win).float(), t.tensor(Wout).float()
    else:
        return Win, Wout

