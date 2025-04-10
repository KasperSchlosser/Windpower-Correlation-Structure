import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

class correlation_model():
    
    def __init__(self, n_sim = 10, horizon = 24, alpha = 0.02, sided = 2, burnin = 240):
        
        self.n_sim = n_sim
        self.horizon = horizon
        self.alpha = alpha
        self.sided = 2 # not really used right now
        self.burnin = burnin
        
        return
    
    def fit(self, *args, **kwargs):
        return
    
    def predict(self):
        return
    def simulate(self):
        return
    def transform(self, data):
        #assume univariate
        
        N = len(data)
        forecast = pd.DataFrame(columns = ["Estimate", "Lower interval", "Upper interval"], index = data.index, dtype = np.float64)
        simulation = pd.DataFrame(columns = ['Simulation' + str(x+1) for x in range(self.n_sim)], index = data.index, dtype = np.float64)
        
        for start in range(self.burnin, N, self.horizon):
            
            #size of winow
            #horizon for most cases but smalle at end
            n_window = self.horizon
            if start + self.horizon > N: n_window = N - start
            
            train_idx = data.index[:start]
            new_idx = data.index[start:start+n_window]
            
            self.fit(data[train_idx].values)
            
            forecast.loc[new_idx, :] = self.predict()[:len(new_idx),:]
            simulation.loc[new_idx, :] = self.simulate()[:len(new_idx),:]
            
        return forecast, simulation
        
class correlation_sarma(correlation_model):
    
    def __init__(self, order = (1,0,0), seasonal_order = (0,0,0,0), trend = None, **kwargs):
        
        super().__init__(**kwargs)
        
        self.model = sm.tsa.SARIMAX(np.zeros(2+seasonal_order[-1]), order = order, seasonal_order = seasonal_order, trend = trend)
        self.modelres = self.model.fit(disp = False)
        
        return
    
    def fit(self, data):
        
        self.modelres = self.modelres.apply(data, refit = True, copy_initialization = True)
        
        return
    
    def simulate(self):
        
        sims = self.modelres.simulate(nsimulations = self.horizon, repetitions = self.n_sim,  anchor = "end")
        sims = sims.squeeze()
        
        return sims
    
    def predict(self):
        
        forecast = self.modelres.get_forecast(steps = self.horizon)
        
        mu = forecast.predicted_mean
        interval = forecast.conf_int(self.alpha)
        
        return np.concatenate((mu[:,np.newaxis], interval), axis = 1)
    
class correlation_nabqr(correlation_model):
    
    def __init__(self, dist = None, **kwargs):
        
        super().__init__(**kwargs)
        
        if dist is None:
            self.dist = stats.norm()
        else:
            self.dist = dist
            
        return
    
    def simulate(self):
        
        sims = self.dist.rvs((self.horizon, self.n_sim))
        
        return sims
    
    def predict(self):
        
        intervals = np.ones((self.horizon, 3)) * self.dist.ppf([0.5, self.alpha / self.sided, 1 - self.alpha / self.sided])
        
        return intervals
