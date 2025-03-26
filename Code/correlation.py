import numpy as np
import pandas as pd
import statsmodels.api as sm

class correlation_model():
    
    def fit(self, data):
        return
    def predict(self, horizon = 24, alpha = 0.02, sided = 2):
        return
    def simulate(self, n_sim = 10, horizon = 24):
        return
    def transform(self, data, n_sim = 10, horizon = 24, alpha = 0.02, sided = 2, burnin = 240):
        #assume univariate
        
        N = len(data)
        forecast = pd.DataFrame(columns = ["estimate", "lower Interval", "upper interval"], index = data.index, dtype = np.float64)
        simulation = pd.DataFrame(columns = ['sim' + str(x+1) for x in range(n_sim)], index = data.index, dtype = np.float64)
        
        for start in range(burnin, N, horizon):
            
            #size of winow
            #horizon for most cases but smalle at end
            n_window = horizon
            if start + horizon > N: n_window = N - start
            
            train_idx = data.index[:start]
            new_idx = data.index[start:start+n_window]
            self.fit(data.loc[train_idx,:])
            
            forecast.loc[new_idx, :] = self.predict(n_window, alpha=alpha, sided = sided)
            simulation.loc[new_idx, :] = self.simulate(n_sim = n_sim, horizon = n_window)
            
        return forecast, simulation
        
class correlation_sarma(correlation_model):
    
    def __init__(self, order = (1,0,0), seasonal_order = (0,0,0,0)):
        
        self.model = sm.tsa.SARIMAX(np.zeros(2+seasonal_order[-1]), order = order, seasonal_order = seasonal_order)
        self.modelres = self.model.fit()
        
        return
    
    def fit(self, data):
        self.modelres = self.modelres.apply(data, refit = True)
        return
    
    def simulate(self, n_sim = 10, horizon = 24):
        sims = self.modelres.simulate(nsimulations = horizon, repetitions = n_sim,  anchor = "end")
        sims = sims.squeeze()
        return sims
    
    def predict(self, horizon = 24, alpha = 0.02, sided = 2):
        
        forecast = self.modelres.get_forecast(steps = horizon)
        
        mu = forecast.predicted_mean
        interval = forecast.conf_int(alpha)
        
        return np.concatenate((mu.values[:,np.newaxis], interval.values), axis = 1)
        
        
        
#%%

y = sm.tsa.arma_generate_sample([1, 0.7], [1,0.3], 1000)
y = pd.DataFrame(y)
model = correlation_sarma(order = (1,0,1))

model.fit(y.values)
res = model.transform(y)
