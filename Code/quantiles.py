import unittest
import numpy as np
import scipy.stats as stats

from collections import namedtuple



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
            pseudo_resids[i, :] = np.array([self.forward(obs) for obs in actuals[i,:]]).squeeze()
        
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
            orig[i, :] = np.array([self.backward(obs) for obs in pseudo_resids[i,:]] ).squeeze()
        
        return orig, pseudo_resids

class constant_model(quantile_model):
    
    def fit(self, est_quantiles):
        self.q = est_quantiles[:]
        
    def forward(self, y):
        tmp = y >= self.q
        if tmp.any(): 
            return self.quantiles[tmp][-1]
        else:
            return self.quantiles[0]
    def backward(self, u):
        tmp = u >= self.quantiles
        if tmp.any(): return self.q[tmp][-1]
        else: return self.q[tmp][0]
     
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
        return vals[conds][0]
    
    def backward(self, u):
        
        conds = (u >= self.quantiles[:-1]) & (u < self.quantiles[1:])
        vals = self.diffs * (u-self.quantiles[:-1]) / self.coefs + self.q_vals[:-1]
        return vals[conds][0]

#%% test everything works
class TestQuantileModels(unittest.TestCase):

    def setUp(self):
        self.quantiles = (0.1, 0.5, 0.9)
        self.est_quantiles = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.actuals = np.array([[1.5, 2.5], [4.5, 5.5], [7.5, 8.5]])
        self.resids = np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]])
        self.min_val = 0
        self.max_val = 10

    def test_constant_model_forward(self):
        model = constant_model(self.quantiles)
        model.fit(self.est_quantiles[0])
        self.assertEqual(model.forward(2), 0.5)
        self.assertEqual(model.forward(3), 0.9)
        self.assertEqual(model.forward(0), 0.1)

    def test_constant_model_backward(self):
        model = constant_model(self.quantiles)
        model.fit(self.est_quantiles[0])
        self.assertEqual(model.backward(0.5), 2)
        self.assertEqual(model.backward(0.9), 3)
        self.assertEqual(model.backward(0.1), 1)

    def test_piecewise_linear_model_forward(self):
        model = piecewise_linear_model(self.quantiles, self.min_val, self.max_val)
        model.fit(self.est_quantiles[0])
        np.testing.assert_almost_equal(model.forward(2), 0.5)
        np.testing.assert_almost_equal(model.forward(3), 0.9)
        np.testing.assert_almost_equal(model.forward(1), 0.1)

    def test_piecewise_linear_model_backward(self):
        model = piecewise_linear_model(self.quantiles, self.min_val, self.max_val)
        model.fit(self.est_quantiles[0])
        np.testing.assert_almost_equal(model.backward(0.5), 2)
        np.testing.assert_almost_equal(model.backward(0.9), 3)
        np.testing.assert_almost_equal(model.backward(0.1), 1)

    def test_transform_normal_function(self):
        model = constant_model(self.quantiles)
        pseudo_resids, resids = model.transform(self.est_quantiles, self.actuals)
        
        # Check the shape of the results
        self.assertEqual(pseudo_resids.shape, self.actuals.shape)
        self.assertEqual(resids.shape, self.actuals.shape)

    def test_back_transform_normal_function(self):
        model = constant_model(self.quantiles)
        pseudo_resids, orig = model.back_transform(self.est_quantiles, self.resids)
        
        # Check the shape of the results
        self.assertEqual(pseudo_resids.shape, self.resids.shape)
        self.assertEqual(orig.shape, self.resids.shape)

    def test_transform_correctness_constant_model(self):
        model = constant_model(self.quantiles)
        
        # Expected pseudo residuals and residuals for the given est_quantiles and actuals
        expected_pseudo_resids = np.array([[0.1, 0.5], [0.1, 0.5], [0.1, 0.5]])
        
        # Transform the actuals using the est_quantiles
        pseudo_resids, resids = model.transform(self.est_quantiles, self.actuals)
        
        # Check if the transformed values match the expected values
        np.testing.assert_almost_equal(pseudo_resids[0], expected_pseudo_resids[0])

    def test_transform_correctness_piecewise_linear_model(self):
        model = piecewise_linear_model(self.quantiles, self.min_val, self.max_val)
        
        # Expected pseudo residuals and residuals for the given est_quantiles and actuals
        expected_pseudo_resids = np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])
        
        # Transform the actuals using the est_quantiles
        pseudo_resids, resids = model.transform(self.est_quantiles, self.actuals)
        
        # Check if the transformed values match the expected values
        np.testing.assert_almost_equal(pseudo_resids[0], expected_pseudo_resids[0])

    def test_back_transform_correctness_constant_model(self):
        model = constant_model(self.quantiles)
        
        # Expected original values for the given est_quantiles and residuals
        expected_orig = np.array([[2, 1], [5, 4], [8, 7]])
        
        # Back transform the residuals using the est_quantiles
        pseudo_resids, orig = model.back_transform(self.est_quantiles, self.resids)
        
        # Check if the back-transformed values match the expected values
        for i in range(len(expected_orig)):
            np.testing.assert_almost_equal(orig[i], expected_orig[i])

    def test_back_transform_correctness_piecewise_linear_model(self):
        model = piecewise_linear_model(self.quantiles, self.min_val, self.max_val)
        
        # Expected original values for the given est_quantiles and residuals
        expected_orig = np.array([[ 2.09956959, 1.90043041], [5.19814927, 4.80185073], [8.29477856, 7.70522144]])
        
        # Back transform the residuals using the est_quantiles
        pseudo_resids, orig = model.back_transform(self.est_quantiles, self.resids)
        
        # Check if the back-transformed values match the expected values
        for i in range(len(expected_orig)):
            np.testing.assert_almost_equal(orig[i], expected_orig[i])
        
if __name__ == '__main__':
    unittest.main()
