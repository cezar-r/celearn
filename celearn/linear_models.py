import numpy as np
import matplotlib.pyplot as plt
from metrics import rmse

class LinearRegression:

	def __init__(self, learning_rate = 0.0001, num_epochs = 100000):
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs

	def fit(self, X_train, y_train):
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

	def _f(self, x):
		return np.matmul(x, self.theta)

	def predict(self, X_test):
		if type(X_test) is not np.array:
			X_test = np.array(X_test)
		X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))
		points = []
		for i in range(X_test.shape[0]):
			points.append(self._f(X_test[i, :])[0])
		return points


X_train = [[1, 2],
           [1, 1], 
           [0, 1],
           [6, 8],
           [6, 9],
           [5, 10]]

y_train = [[1],
           [1],
           [1],
           [0],
           [0],
           [0]]

X_test = [[1, 0],
          [0, 0],
          [7, 8],
          [6, 7]]

y_test = [[1],
          [1],
          [0],
          [0]]

clf = LinearRegression()
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print(rmse(y_hat, y_test))
