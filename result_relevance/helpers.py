import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class ColumnExtractor(BaseEstimator, TransformerMixin):
	
	def __init__(self, cols, categorical=True):
		"""force: force columns to be categorical """
		if isinstance(cols, str):
			self.cols = [cols]
		else:
			self.cols = cols

		self.categorical = categorical

	def fit(self, X, y=None):
		if self.categorical:
			self.categories = {col: X[col].astype('category').cat.categories
								for col in self.cols}
		else:
			self.categories = None

		return self

	def transform(self, X):
		out = []
		if self.categorical:
			out = [X[col].astype('category', categories=self.categories[col])\
					.cat.codes.replace(-1, np.nan).values
					for col in self.cols]
			return np.column_stack(out)
		else:
			return X[self.cols].values

class DateTimeExtractor(BaseEstimator, TransformerMixin):

	def __init__(self, mode='dayofweek', format=None):
		self.mode = mode
		self.format = format

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		out = []
		for v in X.T:
			if self.mode == 'dayofweek' or self.mode == 'weekday':
				out.append(pd.to_datetime(v, format=self.format).weekday)
			elif self.mode == 'dayofmonth' or self.mode == 'day':
				out.append(pd.to_datetime(v, format=self.format).day)
			elif self.mode == 'dayofyear':
				out.append(pd.to_datetime(v, format=self.format).dayofyear)
			elif self.mode == 'week':
				out.append(pd.to_datetime(v, format=self.format).week)
			elif self.mode == 'month':
				out.append(pd.to_datetime(v, format=self.format).month)
			elif self.mode == 'quarter':
				out.append(pd.to_datetime(v, format=self.format).quarter)
			elif self.mode == 'year':
				out.append(pd.to_datetime(v, format=self.format).year)
			elif self.mode == 'hour':
				out.append(pd.to_datetime(v, format=self.format).hour)
			elif self.mode == 'minute':
				out.append(pd.to_datetime(v, format=self.format).minute)
			else:
				raise ValueError('mode does not exist')

		return np.column_stack(out)
	
class TFIDFVectorizer2(BaseEstimator, TransformerMixin):

	def __init__(self, b=0.7):
		self.b = b
		self.cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1), \
                     		     stop_words='english', lowercase=True, strip_accents='unicode', min_df=3)

	def fit(self, X, y=None):
		X_matrix = self.cv.fit_transform(X)
		n_docs, n_words = X_matrix.shape

		# calculate IDF
		df = np.bincount(X_matrix.indices, minlength=n_words)
		idf = np.log((n_docs+1) / df)
		self.idf_diag = sp.spdiags(idf, diags=0, m=n_words, n=n_words)

		# calculate doc length normalizaton
		dl = X_matrix.sum(axis=1)
		dl_norm = 1 - self.b + self.b * dl / dl.mean()
		self.dl_diag = sp.spdiags(1/np.ravel(dl_norm), diags=0, m=n_docs, n=n_docs)

		return self

	def transform(self, X):
		#calculate TF
		X_matrix = self.cv.transform(X)
		tf = X_matrix.copy()
		tf.data = np.log(1 + np.log(1 + tf.data))

		return self.dl_diag * tf * self.idf_diag
