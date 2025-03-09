# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:10:36 2025

@author: KPFS
"""
import scipy
import scipy.optimize as opt
import scipy.stats as stats
import numpy as np
import sympy as sp

a = -6
b = 3

samples = stats.uniform(a, b-a).rvs(100)
x = np.arange(0.01,1,0.01)
#q = a + x*(b-a) + stats.norm(b = 0.1).rvs(x.size)
q = np.quantile(samples, x)

dx = np.diff(x, prepend = 0, append = 1)


def F(k, a, b):
    return (k - a) / (b-a)

def df(a,b):
    tmp = np.insert(q, [0,len(q)], [a, b])
    return np.diff(F(tmp, a, b))
 
def loss(theta):

    a = theta[0]
    b = theta[1]

    
    DF = df(a,b)

    l = (DF @ np.log(DF) - DF @ np.log(dx))


    return l

x0 = [-10,10]

res = opt.minimize(loss, x0, bounds = [(None, q.min() - 1e-8), (q.max() + 1e-8,None)])
print(res.x)