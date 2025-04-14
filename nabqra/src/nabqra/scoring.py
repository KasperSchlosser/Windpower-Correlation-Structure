import numpy as np
import properscoring as ps

def variogram_weight(simulations,
                    p=0.5, window = 24, offset = 24):
    
    n,m = simulations.shape
    n_windows = (n - offset) // window 
    
    weights = np.zeros((n_windows, window, window))
    
    for k,start in enumerate(range(offset, n, window)):
        
        if start + window > n: break
    
        for i in range(0,window-1):
            for j in range(i+1, window):
                
                Ediff = np.abs(simulations[start + i, :] - simulations[start + j, :]) ** p
                
                weights[k,i,j] = 1 / np.var(Ediff, ddof = 1)
    return weights
    
    
def variogram_score(simulations, actuals,
                    p=0.5, window = 24, offset = 24,
                    weights = None):
    
    #simualtions are n observations times m simulations
    n,m = simulations.shape
    
    score = 0
    variogram = np.zeros((window,window))
    n_windows = (n - offset) // window
    
    if weights is None:
        weights = np.array([[abs(i-j) for i in range(0,window)] for j in range(0,window)])
        np.fill_diagonal(weights, 1) # just to not get warning, is not used
        weights = 1/weights
        weights = np.repeat(weights[np.newaxis,:,:,], n_windows, axis = 0)
    
    for k,start in enumerate(range(offset, n, window)):
        
        if start + window > n: break
        
        for i in range(0,window-1):
            for j in range(i+1, window):
                Ediff =  np.abs(simulations[start + i, :] - simulations[start + j, :]) ** p
                Adiff = np.abs(actuals[start + i] - actuals[start + j]) ** p
                
                
                s = weights[k,i,j] * (Adiff - np.mean(Ediff)) ** 2
                variogram[i,j] += s
                variogram[j,i] += s
                score += s


    return score / n_windows, variogram / n_windows

def continous_ranked_probability_score(simulations, actuals):
    #n observation times k simulations
    return np.mean(ps.crps_ensemble(actuals, simulations))

def mean_average_error(actuals, predicted):
    return np.mean(np.abs(actuals - predicted))

def mean_squared_error(actuals, predicted):
    return np.mean((actuals - predicted)**2)

def calc_scores(actuals, predicted, simulations, VARS_kwargs = None):
    
    #maybe return dict?
    
    if VARS_kwargs is None:
        VARS_kwargs = dict()
    
    MAE = mean_average_error(actuals, predicted)
    MSE = mean_squared_error(actuals, predicted)
    VARS = variogram_score(simulations, actuals, **VARS_kwargs)[0]
    CRPS = continous_ranked_probability_score(simulations, actuals)
    
    return MAE, MSE, VARS, CRPS
