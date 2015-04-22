from sklearn.base import BaseEstimator
import pandas as pd

class NAFiller:
    def __init__(self, cols, method='mean'):
        self.cols = cols
        self.method = method
    
    def fit(self, X):
        if self.method = 'mean':
            self.filling = X[self.cols].mean(skipna=True)
        elif self.method = 'median':
            self.filling = X[self.cols].median(skipna=True)
        else:
            raise ValueError('method only accpets mean and median')
        
    def transform(self, X):
        return X.fillna(self.filling)

class BayesClassifier(BaseEstimator):
	def __init__(self):
		pass

	def fit(self, X, y):
		
