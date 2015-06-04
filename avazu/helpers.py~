import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
