# knn with no numpy
from math_ops import mode

class KNearestNeighbors:
	"""Simple KNearestNeighbors algorithm that predicts classes based on the neighbors"""
	
	def __init__(self, neighbors = 3):
		"""Initalizes how many nieghbors should be used when casting a prediction
		
		Parameters:
		neighbors: int
		"""
		self.neighbors = neighbors

	def fit(self, X_train, y_train):
		"""Main function that fits the data to each label
		
		Parameters
		----------
		X_train: 2D list
		y_tran: 2D list
		"""
		self.classes_dict = {}
		for i in range(len(X_train)):
			if y_train[i][0] in self.classes_dict:
				self.classes_dict[y_train[i][0]].append(X_train[i])
			else:
				self.classes_dict[y_train[i][0]] = [X_train[i]]
		assert len(self.classes_dict) != len(y_train), "Continuous values detected, please use discrete classes"

	def _euc_dist(self, row1, row2):
		"""Calculates the euclidean distance between two points, in any dimensionality
		
		Parameters
		----------
		row1: list
		row2: list
		
		Returns
		-------
		under_sqrt ** .5: int
		"""
		under_sqrt = 0
		for i in range(len(row1)):
			under_sqrt += ((row2[i] - row1[i])**2)
		return under_sqrt ** .5

	def predict(self, X_test):
		"""Predicts the labels of given data
		
		Parameters
		----------
		X_test: 2D list
		
		Returns
		-------
		preds: 2D list
		"""
		preds = []
		for i in range(len(X_test)):
			distances = []
			classes = []
			for k in self.classes_dict:
				for row in self.classes_dict[k]:
					distances.append(self._euc_dist(X_test[i], row))
					classes.append(k)
			dist_classes_dict = dict(sorted(zip(distances, classes), key = lambda x : x[0]))
			preds.append([mode(list(dist_classes_dict.values())[:self.neighbors])])
		return preds
