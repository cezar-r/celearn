import random

def train_test_split(X, y, test_size = .25):
	xy = list(zip(X, y))
	random.shuffle(xy)
	X, y = zip(*xy)

	length = int(round(len(X) * test_size, 0))
	X_train = X[length:]
	X_test = X[:length]
	y_train = y[length:]
	y_test = y[:length]
	return X_train, X_test, y_train, y_test






if __name__ == '__main__':
	X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	y = [[1], [0], [1]]
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	print(X_train, y_train)
	print(X_test, y_test)
