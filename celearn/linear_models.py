#!/usr/bin/env python
# coding: utf-8

"""Various linear models used to predict classes or continuous values"""

import numpy as np

class LinearRegression:
	"""Classic linear regression model"""

	def __init__(self, learning_rate = 0.0001, num_epochs = 100000):
		"""Initalizes the learning rate and number of epochs the model will take
		
		Parameters
		----------
		learning_rate: float
		num_epochs: int
		"""
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs

	def fit(self, X_train, y_train):
		"""Main function used to fine tune weights and biases
		
		Parameters
		----------
		X_train: np.array (n, m)
		y_train: np.array (n, 1)
		"""
		if type(X_train) is not np.array:
			X_train = np.array(X_train)
		if type(y_train) is not np.array:
			y_train = np.array(y_train)
			
		X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
		self.theta = np.zeros((X_train.shape[1], 1))
		rows = X_train.shape[0]
		
		for _ in range(self.num_epochs):
			h_x = self._f(X_train)
			cost = (1/rows)*(X_train.T@(h_x - y_train))
			self.theta = self.theta - (self.learning_rate)*cost

	def predict(self, X_test):
		"""Uses the trained weights and biases to cast a prediction
		
		Parameters
		----------
		X_test: np.array(n, m)
		
		Returns
		-------
		preds: 2D list
		"""
		if type(X_test) is not np.array:
			X_test = np.array(X_test)
		X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))
		preds = []
		for i in range(X_test.shape[0]):
			points.append([self._f(X_test[i, :])[0]])
		return preds
	
	def _f(self, x):
		"""Matrix multiplication across training set and theta value
		
		Parameters
		----------
		x: np.array (n, m)
		
		Returns
		-------
		np.matmul(x, self.theta): np.array(n, n)
		"""
		return np.matmul(x, self.theta)


class LogisticRegression:
	"""Like linear regression, but uses sigmoid function to cast a class prediction"""

	def __init__(self, learning_rate = 0.001, num_epochs = 100):
		"""Initalizes the learning rate and number of epochs the model will take
		
		Parameters
		----------
		learning_rate: float
		num_epochs: int
		"""
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs

	def fit(self, X_train, y_train):
		"""Main function used to fine tune weights and biases
		
		Parameters
		----------
		X_train: np.array (n, m)
		y_train: np.array (n, 1)
		"""
		if type(X_train) is not np.array:
			X_train = np.array(X_train)
		if type(y_train) is not np.array:
			y_train = np.array(y_train)

		weights = np.random.rand(X_train.shape[1])
		weights = weights[:, np.newaxis]
		for _ in range(self.num_epochs):
			pred = self._sigmoid(np.dot(X_train, weights))
			weights -= self.learning_rate * np.dot(X_train.T, pred - y_train) / len(X_train)
		self.weights = weights 

	def predict(self, X_test):
		"""Uses the trained weights and biases to cast a prediction
		
		Parameters
		----------
		X_test: np.array(n, m)
		
		Returns
		-------
		[[1] if i >= 0.5 else [0] for i in self._sigmoid(_pred)]: 2D list
		"""
		if type(X_test) is not np.array:
			X_test = np.array(X_test)

		_pred = np.dot(X_test, self.weights)
		return [[1] if i >= 0.5 else [0] for i in self._sigmoid(_pred)]

	def _sigmoid(self, num):
		"""Sigmoid function
		
		Parameters
		----------
		num: int
		
		Returns
		-------
		1 / (1 + np.e**(-num)): float
		"""
		return 1 / (1 + np.e**(-num))


class SGDClassifier:
	"""Like logistic regression, but uses one sample point to fine tune weights and biases"""

	def __init__(self, learning_rate = 0.0001, num_epochs = 100000):
		"""Initalizes the learning rate and number of epochs the model will take
		
		Parameters
		----------
		learning_rate: float
		num_epochs: int
		"""
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs

	def fit(self, X_train, y_train):
		"""Main function used to fine tune weights and biases
		
		Parameters
		----------
		X_train: np.array (n, m)
		y_train: np.array (n, 1)
		"""
		if type(X_train) is not np.array:
			X_train = np.array(X_train)
		if type(y_train) is not np.array:
			y_train = np.array(y_train)

		coef = [0.0 for i in range(X_train.shape[1])]
		for _ in range(self.num_epochs):
			sum_error = 0
			for i in range(len(X_train)):
				y_hat = self._predict(X_train[i], coef)
				error = y_train[i][0] - y_hat
				sum_error += error **2
				coef[0] = coef[0] + self.learning_rate * error * y_hat * (1 - y_hat)
				for j in range(len(X_train[i])):
					coef[j] = coef[j] + self.learning_rate * error * y_hat * (1 - y_hat) * X_train[i, j]
		self.coef = coef

	def _predict(self, row, coef):
		"""Used to predict a given row of data with given coefficients
		
		Parameters
		----------
		row: np.array(1, m)
		
		Returns
		-------
		self._sigmoid: float
		"""
		y_hat = coef[0]
		for i in range(len(row) - 1):
			y_hat += coef[i + 1] * row[i]
		return self._sigmoid(y_hat)
	
	def _sigmoid(self, num):
		"""Sigmoid function
		
		Parameters
		----------
		num: int
		
		Returns
		-------
		1 / (1 + np.e**(-num)): float
		"""
		return 1 / (1 + np.e**(-num))

	def predict(self, X_test):
		"""Uses the trained weights and biases to cast a prediction
		
		Parameters
		----------
		X_test: np.array(n, m)
		
		Returns
		-------
		preds: 2D list
		"""
		preds = []
		for i in range(len(X_test)):
			preds.append([round(self._predict(X_train[i], self.coef))])
		return preds

	
class Lasso:
	"""Like linear regression, but penalizes features that are correlated"""

	def __init__(self, learning_rate = 0.001, num_epochs = 1000, l1_penalty = 800):
		"""Initalizes the learning rate, number of epochs the model will take, as well as the L1 penalty
		
		Parameters
		----------
		learning_rate: float
		num_epochs: int
		l1_penalty: int
		"""
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.l1_penalty = l1_penalty


	def fit(self, X_train, y_train):
		"""Main function used to fine tune weights and biases
		
		Parameters
		----------
		X_train: np.array (n, m)
		y_train: np.array (n, 1)
		"""
		if type(X_train) is not np.array:
			X_train = np.array(X_train)
		if type(y_train) is not np.array:
			y_train = np.array(y_train)

		self.num_features = X_train.shape[1]
		self.num_rows = X_train.shape[0]

		self.W = np.zeros(self.num_features)
		self.b = 0

		for _ in range(self.num_epochs):
			self._update_weights(X_train, y_train)

	def _update_weights(self, X_train, y_train):
		"""Helper function used to fine tune weights
		
		Parameters
		----------
		X_train: np.array (n, m)
		y_train: np.array (n, 1)
		"""
		y_hat = self.predict(X_train)
		dw = np.zeros(self.num_features)
		for j in range(self.num_features):
			if self.W[j] > 0:
				dw[j] = np.mean((-(2 * (X_train[:, j]).dot(y_train - y_hat)) + self.l1_penalty) / self.num_rows)
			else:
				dw[j] = np.mean((-(2 * (X_train[:, j]).dot(y_train - y_hat)) - self.l1_penalty) / self.num_rows)

		db = -2 * np.sum(y_train - y_hat) / self.num_rows

		self.W = self.W - self.learning_rate * dw
		self.b = self.b - self.learning_rate * db 

	def predict(self, X_test):
		"""Uses the trained weights and biases to cast a prediction
		
		Parameters
		----------
		X_test: np.array(n, m)
		
		Returns
		-------
		[[i] for i in X_test.dot(self.W) + self.b]: 2D list
		"""
		if type(X_test) is not np.array:
			X_test = np.array(X_test)
		return [[i] for i in X_test.dot(self.W) + self.b]
