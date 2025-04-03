# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:53:24 2025

@author: KPFS
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

plt.close('all')

dist = stats.norm()
N = 5

x = np.linspace(-6,6, 10000)
X = np.zeros(N + 2)
#X = np.linspace(-2,2,7)
X[1:-1] = np.sort(stats.norm(scale = 2).rvs(N))
X[0] = x[0]
X[-1] = x[-1]
Q = dist.cdf(X)
P = dist.pdf(X)

P_est = np.zeros(len(X))
for i in range(1,len(X)):
    P_est[i] = 2*(Q[i]-Q[i-1]) / (X[i]-X[i-1]) - P_est[i-1]


blok = ((P_est[1:] - P_est[:-1]) / 2 + P_est[:-1]) * (X[1:] - X[:-1])


plt.figure()
plt.subplot(1,2,1)
plt.plot(x,dist.pdf(x), color = 'k')
plt.plot(X, P, marker = "*", markersize = 8, color = 'b')
plt.plot(X, P_est, marker = "*", markersize = 8, color = 'r' )
plt.subplot(1,2,2)
plt.plot(x,dist.cdf(x), color = 'k')
plt.plot(X, Q, marker = "*", markersize = 8, color = 'b')
plt.plot(X, np.pad(np.cumsum(blok), (1,0)), marker = "*", markersize = 8, color = 'r')


#%%

plt.close('all')

dist = stats.norm()
N = 7

x = np.linspace(-4,4, 10000)
X = np.sort(stats.norm(scale = 1).rvs(N))
Q = dist.cdf(X)

X = np.pad(X, 1, constant_values = x[[0,-1]])
Q = np.pad(Q, 1, constant_values = (0,1))

dq = np.diff(Q)
dx = np.diff(X)

def a_gen(b):
    return (dq - b)

B = np.zeros(len(dq))
A = np.zeros(len(dq))
B[0] = 1.5*dq[0]
for i in range(0,len(B)-1):
    A[i] = dq[i] - B[i]
    B[i+1] = 2*A[i] + B[i]

a1 = A
b1 = B
#b1 = B
#b1 = k*dq
#a1 = a_gen(b1)

XS = (x[:,np.newaxis] - X[:-1]) / dx
clist = (0 <= XS) & (XS < 1)

vlist = a1 * XS**2 + b1 * XS + Q[:-1]
dlist = 2*a1 * XS + b1

plt.subplot(1,2,1)
plt.plot(x[:-1], vlist[clist])
plt.scatter(X,Q)
plt.subplot(1,2,2)
plt.plot(x[:-1], dlist[clist])
plt.show()



