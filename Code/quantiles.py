import numpy as np
import scipy.stats as stats



#%% quantile models
# models estimating cdfs from the nabqr quantiles
# models fits a function to quantiles model(q) -> F(x)
# foward takes and observation x and gives resulting quantile:
#   model.forward(y): F(y) = u
# Backwards takes from cdf-space back to original space
# model.backward(u): F^-1(u) = x

class quantile_model():
    def __init__(self, quantiles = (0.5,0.3,0.5,0.7,0.95), dist = stats.norm()):
        self.quantiles = np.array(quantiles)
        self.dist = dist
        
    def fit(self, est_quantiles):
        return
    def forward(self, y):
        return
    def backward(self, u):
        return
    
    def transform(self, est_quantiles, actuals):
       
        pseudo_resids = np.zeros(len(est_quantiles))
        
        for i in range(len(est_quantiles)):
            self.fit(est_quantiles[i,:])
            pseudo_resids[i] = self.forward(actuals[i])
        
        resids = stats.norm().ppf(pseudo_resids)
        
        return pseudo_resids, resids
    
    def back_transform(self, est_quantiles, resids):
        
        pseudo_resids = stats.norm().cdf(resids)
        
        orig = np.zeros(len(pseudo_resids))
        
        for i in range(len(est_quantiles)):
            self.fit(est_quantiles[i,:])
            orig[i] = self.backward(pseudo_resids[i])
        
        return pseudo_resids, orig

class constant_model(quantile_model):
    
    def fit(self, est_quantiles):
        self.q = est_quantiles[:]
        
    def forward(self, y):
        tmp = y > self.q
        if tmp.any(): 
            return self.quantiles[tmp][-1]
        else:
            return self.quantiles[0]
    def backward(self, u):

        tmp = u == self.q
        if tmp.any(): return self.q[tmp]
        else: return np.nan
     
class piecewise_linear_model(quantile_model):
    
    def __init__(self, quantiles, min_val, max_val):
        
        super().__init__(quantiles)
        
        tmp = np.zeros(len(self.quantiles)+2)
        tmp[1:-1] = self.quantiles
        tmp[0] = 0
        tmp[-1] = 1
        
        self.quantiles = tmp
        
        self.max_val = max_val
        self.min_val = min_val
    
    
    def fit(self, est_quantiles):
        
        self.q_vals = np.zeros(len(est_quantiles)+2)
        self.q_vals[1:-1] = est_quantiles
        
        self.q_vals[-1] = self.max_val
        self.q_vals[0] = self.min_val

        self.diffs = self.q_vals[1:] - self.q_vals[:-1]
        self.coefs = self.quantiles[1:] - self.quantiles[:-1]
    
    def forward(self, y):
        
        conds = (y >= self.q_vals[:-1]) & (y < self.q_vals[1:])
        vals = self.coefs * (y - self.q_vals[:-1]) / self.diffs + self.quantiles[:-1]

        return np.piecewise(y, conds, vals)
    
    def backward(self, u):
        
        conds = (u >= self.quantiles[:-1]) & (u < self.quantiles[1:])
        vals = self.diffs * (u-self.quantiles[:-1]) / self.coefs + self.q_vals[:-1]

        return np.piecewise(u, conds, vals)


