
import numpy as np
import scipy.stats as stats
import scipy.interpolate as interpolate
import pandas as pd


# quantile models
# models estimating cdfs from the nabqr quantiles
# models fits a function to quantiles model(q) -> F(x)
# foward takes and observation x and gives resulting quantile:
#   model.cdf(y): F(y) = u
# Backwards takes from cdf-space back to original space
# model.quantile(u): F^-1(u) = x

class quantile_model():
    def __init__(self, quantiles, min_val = None, max_val = None, dist = stats.norm(), *args, **kwargs):
        
        quant = np.zeros(len(quantiles)+2)
        quant[1:-1] = quantiles
        quant[0] = 0
        quant[-1] = 1
        
        self.original_quantiles = quant
        
        self.max_val = max_val
        self.min_val = min_val
        
        self.dist = dist
        
        return
        
    def fit(self, est_quantiles):
        
        self.q_vals = np.zeros(len(est_quantiles)+2)
        self.q_vals[1:-1] = est_quantiles
        
        self.q_vals[-1] = self.max_val
        self.q_vals[0] = self.min_val
        
        # we sometimes get identical x values.
        # in this case we keep the one corresponding to the lower quantile
        # might be inaccurate, but should only happen when the distribution is very close
        _, idx = np.unique(self.q_vals, return_index = True)
        
        self.q_vals = self.q_vals[idx]
        self.quantiles = self.original_quantiles[idx]
        self.quantiles[-1] = 1 #ensure we still have the 100% quantile
        
        self.dx = self.q_vals[1:] - self.q_vals[:-1]
        self.dq = self.quantiles[1:] - self.quantiles[:-1]
        
        return
    
    def cdf(self, y, *args, **kwargs):
        if not np.isscalar(y) or not np.isfinite(y): return np.nan
        
        if y < self.q_vals[0]: return 0
        if y > self.q_vals[-1]: return 1
        
        return self._cdf(y, *args, **kwargs)
    
    def quantile(self, u, *args, **kwargs):
        if not np.isscalar(u) or not np.isfinite(u): return np.nan
        
        if u < self.quantiles[0]: return np.nan
        if u > self.quantiles[-1]: return np.nan
        
        return self._quantile(u, *args, **kwargs)
    
    def pdf(self, y, *args, **kwargs):
        if not np.isscalar(y) or not np.isfinite(y): return np.nan
        if y < self.q_vals[0] or y > self.q_vals[-1]: return 0
        
        return self._pdf(y, *args, **kwargs)
    
    def _cdf(self, y, *args, **kwargs):
        return
    def _quantile(self, u, *args, **kwargs):
        return
    def _pdf(self, y, *args, **kwargs):
        return
    
    def transform(self, est_quantiles, actuals: np.array):
        
        #est_quantiles: N x K matrix
        #   N observations
        #   K quantiles
        #Actuals: N * M matrix, 
        #   N observation, corresponding to the observed quantiles
        #   M values to transform for each observation
        
        if actuals.ndim == 1:
            actuals = actuals[:,np.newaxis]
        if est_quantiles.ndim == 1:
            est_quantiles = est_quantiles[np.newaxis,:]
       
        pseudo_resids = np.zeros(actuals.shape)
        
        for i in range(len(est_quantiles)):
            
            self.fit(est_quantiles[i,:])

            pseudo_resids[i, :] = np.array([self.cdf(obs) for obs in actuals[i,:]]).squeeze()
        
        resids = stats.norm().ppf(pseudo_resids)
        
        return resids, pseudo_resids
    
    def back_transform(self, est_quantiles, resids):
        
        #est_quantiles: N x K matrix
        #   N observations
        #   K quantiles
        #resids: N * M matrix, 
        #   N observations, corresponding to the observed quantiles
        #   M values to transform for each observation
        
        if resids.ndim == 1:
            resids = resids[:, np.newaxis]
        if est_quantiles.ndim == 1:
            est_quantiles = est_quantiles[np.newaxis, :]
            
        pseudo_resids = stats.norm().cdf(resids)
        orig = np.zeros(pseudo_resids.shape)
        
        for i in range(len(est_quantiles)):
            self.fit(est_quantiles[i,:])
            
            orig[i, :] = np.array([self.quantile(obs) for obs in pseudo_resids[i,:]] ).squeeze()
        
        return orig, pseudo_resids


class constant_model(quantile_model):
    def _cdf(self, y):
        mask = (y >= self.q_vals[:-1]) & (y < self.q_vals[1:])
        
        res = self.quantiles[:-1][mask] 
        return res[0]
        
    def _quantile(self, u):
        mask = (u >= self.quantiles[:-1]) & (u < self.quantiles[1:])
        
        res = self.q_vals[:-1][mask]
        
        return res[0]
    
    # not applicable for this
    # basically a discrete distribution
    def _pdf(self, y):
        return np.nan 
        

class piecewise_linear_model(quantile_model):
    
    def __init__(self, tail_correction = None, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        assert tail_correction in (None, "Flat")
        self.tail_correction = tail_correction
        
        return
    
    def _get_poly_coef(self, ix, dq, dx):
        
        match (ix, self.tail_correction):
            case (0, "Flat"):
                a = dq / dx**2
                b = 0

            case (-1, "Flat"):
                a = - dq / dx**2
                b = 2 * dq / dx

            case _: 
                a = 0
                b = dq / dx

        return a,b

    def _cdf(self, y):
        conds = (y >= self.q_vals[:-1]) & (y < self.q_vals[1:])
        
        ix = np.argmax(conds)
        if ix == len(conds)-1: ix = -1
        
        dq = self.dq[ix]
        dx = self.dx[ix]
        
        x = y - self.q_vals[:-1][ix]
        base_val = self.quantiles[:-1][ix]
    
        a,b = self._get_poly_coef(ix, dq, dx)
        
        return a * x**2 + b * x + base_val
    
    def _quantile(self, u):

        conds = (u >= self.quantiles[:-1]) & (u < self.quantiles[1:])
        
        ix = np.argmax(conds)
        
        dq = self.dq[ix]
        dx = self.dx[ix]
        
        x = u - self.quantiles[:-1][ix]
        base_val = self.q_vals[:-1][ix]
        
        a,b = self._get_poly_coef(ix, dq,dx)
        
        if a > 0:
            res = (-b + np.sqrt(b**2 + 4*a*x)) / (2*a)
        elif a < 0:
            res = (-b - np.sqrt(b**2 + 4*a*x)) / (2*a)
        elif a == 0:
            res = x / b
        else:
            res = np.nan
        return res + base_val
    
    def _pdf(self,y):
        conds = (y >= self.q_vals[:-1]) & (y < self.q_vals[1:])
        
        ix = np.argmax(conds)
        if ix == len(conds)-1: ix = -1
        
        dq = self.dq[ix]
        dx = self.dx[ix]
        
        x = y - self.q_vals[:-1][ix]
    
        a,b = self._get_poly_coef(ix, dq, dx)
        
        return 2*a*x + b

        
class spline_model(quantile_model):
    def fit(self, *args, **kwargs):
        
        super().fit(*args, **kwargs)
        
        
        self.model = interpolate.PchipInterpolator(self.q_vals, self.quantiles, extrapolate = False)
        self.deriv = self.model.derivative(1)
        
        return
    
    def _cdf(self, y):
        return self.model(y)
    def _quantile(self, u):
        return self.model.solve(u)[0]
    def _pdf(self, y):
        return self.deriv(y)
        


