from math_ops import mode
from metrics import accuracy_score

class KNearestNeighbors:

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
			preds.append([mode(list(dist_classes_dict.values())[:k_neighbors])])
		return preds



if __name__ == '__main__':
	from ensemble import RandomForest



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

	clf = KNearestNeighbors()
	clf.fit(X_train, y_train)
	y_hat = clf.predict(X_test)
	print(accuracy_score(y_hat, y_test))