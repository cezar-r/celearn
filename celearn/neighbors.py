from math_ops import mode
from metrics import accuracy_score

class KNearestNeighbors:
	
	def __init__(self, neighbors = 3):
		self.neighbors = neighbors

	def fit(self, X_train, y_train):
		self.classes_dict = {}
		for i in range(len(X_train)):
			if y_train[i][0] in self.classes_dict:
				self.classes_dict[y_train[i][0]].append(X_train[i])
			else:
				self.classes_dict[y_train[i][0]] = [X_train[i]]
		assert len(self.classes_dict) != len(y_train), "Continuous values detected, please use discrete classes"


	def _euc_dist(self, row1, row2):
		under_sqrt = 0
		for i in range(len(row1)):
			under_sqrt += ((row2[i] - row1[i])**2)
		return under_sqrt ** .5


	def predict(self, X_test, k_neighbors = 3):
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
