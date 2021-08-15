import numpy as np
import pandas as pd

class GaussianNB:

	def fit(self, X_train, y_train):
		if type(X_train) is not np.array:
			X_train = np.array(X_train)
		if type(y_train) is not np.array:
			y_train = np.array(y_train)

		self.classes = np.unique([i[0] for i in y_train])
		self.num_classes = len(self.classes)
		self.num_features = X_train.shape[1]
		self.num_rows = X_train.shape[0]

		self._calc_stats(X_train, y_train)
		self._calc_prior(X_train, y_train)


	def predict(self, X_test):
		if type(X_test) is not np.array:
			X_test = np.array(X_test)

		preds = [self._calc_posterior(f) for f in X_test]
		return preds


	def _calc_stats(self, X_train, y_train):
		# convert to pandas first
		X_train = pd.DataFrame(X_train)

		self.mean = X_train.groupby(y_train.flatten()).apply(np.mean).to_numpy()
		self.var = X_train.groupby(y_train.flatten()).apply(np.var).to_numpy()


	def _calc_prior(self, X_train, y_train):
		X_train = pd.DataFrame(X_train)
		
		self.prior = (X_train.groupby(y_train.flatten()).apply(lambda x : len(x)) / self.num_rows).to_numpy()

	def _gaussian(self, cls_idx, row):
		mean = self.mean[cls_idx]
		var = self.var[cls_idx]

		numerator = np.exp((-1/2) * ((row - mean) ** 2) / (2 * var))
		denominator = np.sqrt(2 * np.pi * var)
		prob = numerator / denominator
		return prob


	def _calc_posterior(self, row):
		posteriors = []
		for i in range(self.num_classes):
			prior = np.log(self.prior[i])
			cond = np.sum(np.log(self._gaussian(i, row)))
			posteriors.append(prior + cond)
		return self.classes[np.argmax(posteriors)]


if __name__ == '__main__':

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

	clf = GaussianNB()
	clf.fit(X_train, y_train)
	y_hat = clf.predict(X_test)
	print(y_hat)


