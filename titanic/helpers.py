from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, Imputer
import numpy as np
import pandas as pd

class NAFiller:
    def __init__(self, cols, method='mean'):
        self.cols = cols
        self.method = method
    
    def fit(self, X):
        if self.method == 'mean':
            self.filling = X[self.cols].mean(skipna=True)
        elif self.method == 'median':
            self.filling = X[self.cols].median(skipna=True)
        else:
            raise ValueError('method only accpets mean and median')
        
    def transform(self, X):
        return X.fillna(self.filling)


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
	def __init__(self, cols=None):
		if not cols:
			raise ValueError('columns are not specified')

		self.cols = cols
		self.le = dict()
		for col in cols:
			self.le[col] = LabelEncoder()

	def fit(self, X, y=None):
		for col, le in self.le.iteritems():
			le.fit(X[:, col])
			
	def transform(self, X):
		X = X.copy()
		for col, le in self.le.iteritems():
			X[:, col] = le.transform(X[:, col])

		return np.array(X, dtype='float32')

	def fit_transform(self, X, y=None):
		X = X.copy()
		for col, le in self.le.iteritems():
			X[:, col] = le.fit_transform(X[:, col])

		return np.array(X, dtype='float32')


class OneHotEncoderMinusOne(BaseEstimator, TransformerMixin):
	def __init__(self, cols=None):
		if not cols:
			raise ValueError('columns are not specified')

		self.cols = cols
		self.oh = dict()
		for col in cols:
			self.oh[col] = OneHotEncoder(sparse=False) 

	def fit(self, X, y=None):
		for col, oh in self.oh.iteritems():
			oh.fit(X[:, col:col+1])

	def transform(self, X):
		tmp = np.hsplit(X, X.shape[1])
		for col, oh in self.oh.iteritems():
			tmp[col] = oh.transform(X[:, col:col+1])[:, 1:]

		return np.hstack(tmp)

	def fit_transform(self, X, y=None):
		tmp = np.hsplit(X, X.shape[1])
		for col, oh in self.oh.iteritems():
			tmp[col] = oh.fit_transform(X[:, col:col+1])[:, 1:]

		return np.hstack(tmp)


class GenericPolyFeatures(BaseEstimator, TransformerMixin):
	def __init__(self, non_dummy_features, degree=2):
		self.conts = non_dummy_features
		
		self.poly_cross = \
			PolynomialFeatures(degree=degree, \
								interaction_only=True, \
								include_bias=False)
		self.poly_cont = \
			PolynomialFeatures(degree=degree, include_bias=False)

	def fit(self, X, y=None):
		# standardize continuous feature index
		self.conts = np.arange(X.shape[1])[self.conts]
		# intialize for cross terms
		self.poly_cross.fit(X)
		self.poly_cont.fit(X[:, :1])
		return self

	def transform(self, X):
		# transform for interction only and exclude duplicate zeros
		cross = self.poly_cross.transform(X)
		cross = cross[:, -np.all(cross==0, axis=0)]

		# transform for continuous features
		cont = np.hstack([self.poly_cont.transform(X[:, idx:idx+1])[:, 1:] \
					for idx in self.conts])

		# combine interaction, dummies and continuous features
		return np.hstack([np.ones((X.shape[0], 1)), cross, cont])

	def fit_transform(self, X, y=None):
		# standardize continuous feature index
		self.conts = np.arange(X.shape[1])[self.conts]

		# transform for interction only and exclude duplicate zeros
		cross = self.poly_cross.fit_transform(X)
		cross = cross[:, -np.all(cross==0, axis=0)]

		# transform for continuous features
		cont = np.hstack( \
				[self.poly_cont.fit_transform(X[:, idx:idx+1])[:, 1:] \
					for idx in self.conts])

		# combine interaction, dummies and continuous features
		return np.hstack([np.ones((X.shape[0], 1)), cross, cont])

class GenericImputer(BaseEstimator, TransformerMixin):
	def __init__(self, cols, methods):
		if len(cols) != len(methods):
			raise ValueError('columns and methods not match')
		
		self.im = dict()
		for col, method in zip(cols, methods):
			self.im(col) = Imputer(strategy=method)

	def __fit__(self, X, y=None):
		for	col, im in self.im.iteritems():
			im.fit(X[, col:col+1]	


	def __transform__(self, X):
		X = X.copy()
		for col, im in self.im.iteritems():
			X[, col] = im.transform(X[, col:col+1])	

		return X

	def __fit_transform__(self, X, y=None):
		X = X.copy()
		for col, im in self.im.iteritems():
			X[, col] = im.fit_transform(X[, col:col+1])	

		return X
