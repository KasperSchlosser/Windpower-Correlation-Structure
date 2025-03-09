# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 19:09:46 2025

@author: KPFS
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dist = stats.norm(loc = 1, scale = 100)
dist2 = stats.norm(loc = 1, scale = 1)

A = dist.rvs((10,10000))
B = dist2.rvs((10,10000))

#%%
plt.close('all')
for trans in [lambda x: x**0.5, lambda x: x, lambda x: x**2]:
    A_right = trans(np.abs(A - 1))
    B_right = trans(np.abs(B - 1))
    a_wrong = trans(np.abs(A - 0))
    b_wrong = trans(np.abs(B - 0))
    
    score_1 = (A_right + B_right)
    score_2 = (A_right + b_wrong)
    score_3 = (a_wrong + B_right)
    score_4 = (a_wrong + b_wrong)
    
    df = pd.DataFrame({"rr":score_1, "rw":score_2, "wr":score_3, "ww":score_4})
    print(df)
    
    print(np.sum(score_1 < score_2))
    """
    df = pd.DataFrame({"right":score_right, "wrong":score_wrong, "pic_wrong": pick_wrong})
    
    plt.figure(layout = "tight", figsize = (14,8))
    plt.scatter(score_right,score_wrong)
    plt.axline((0,0), slope = 1)


    
    print(df.describe(include = "all"))
    """
    