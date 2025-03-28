import pandas as pd
import numpy as np

from pandas import IndexSlice as idx

class pipeline():
    
    def __init__(self, correlation_model, quantile_model = None):
        
        self.quantile_model = quantile_model
        self.correlation_model = correlation_model
        
    def run(self, estimated_quantiles, observations, n_sim = 10, **kwargs):
        
        
        res_cols = pd.MultiIndex.from_product((["Orignal", "CDF", "Normal"], ["Observation", "Estimate", "Upper Interval", "Lower Interval"]), names = ["Space", "Result Type"])
        sim_cols = pd.MultiIndex.from_product((["Orignal", "CDF", "Normal"], ["Simulation" + str(x+1) for x in range(n_sim)]), names = ["Space", "Simulation"])
        
        res = pd.DataFrame(index = observations.index, columns = res_cols, dtype = np.float64)
        sim = pd.DataFrame(index = observations.index, columns = sim_cols, dtype = np.float64)
        
        res["Original", "Observation"] = observations
        
        tmp = self.quantile_model.transform(estimated_quantiles, observations)
        res.loc[:, idx["Normal", "Observation"]] =  tmp[0]
        res.loc[:, idx["CDF", "Observation"]] =  tmp[1] 
        
        tmp = self.correlation_model.transform(res["Normal","Observation"], n_sim = n_sim, **kwargs)
        res.loc[:, idx["Normal", ["Estimate", "Upper Interval", "Lower Interval"] ]] =  tmp[0]
        sim.loc[:, idx["Normal", : ]] = tmp[1]
        
        tmp = self.quantile_model.back_transform(estimated_quantiles, res["Normal", ["Estimate", "Upper Interval", "Lower Interval"]])
        res.loc[:, idx["Original", ["Estimate", "Upper Interval", "Lower Interval"] ]] = tmp[0]
        res.loc[:, idx["CDF", ["Estimate", "Upper Interval", "Lower Interval"] ]] = tmp[1]
        
        tmp = self.quantile_model.back_transform(estimated_quantiles, sim["Normal", :])
        sim.loc[:, idx["Original", : ]] = tmp[0]
        sim.loc[:, idx["CDF", : ]] = tmp[1]
        
        return res, sim
        
        