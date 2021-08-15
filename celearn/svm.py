import numpy as np 

class LinearSVC:

	def __init__(self, pos_label = 1):
		self.pos_label = pos_label


	def fit(self, X_train, y_train):
		# sequential minimal optimization - batch-like system

		y_train = self._relabel_y(y_train)
		print(y_train)

		opt_dict= {}
		transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

		self.max_feat = max(sum(X_train, []))
		self.min_feat = min(sum(X_train, []))
		print(self.max_feat)
		print(self.min_feat)


		step_sizes = [self.max_feat * .1,
					self.max_feat * .01,
					self.max_feat * .001]

		b_range_step_sizes = 2
		b_multiple = 5
		latest_optimum = self.max_feat * 10

		for step in step_sizes:
			w = np.array([latest_optimum, latest_optimum])
			optimized = False
			while not optimized:
				for b in np.arange(-1 * (self.max_feat * b_range_step_sizes), self.max_feat * b_range_step_sizes, step*b_multiple):
					for transformation in transforms:
						w_t = w * transformation
						found = True
						for i in range(len(X_train)):
							if not y_train[i][0] * (np.dot(w_t, X_train[i])+b) >= 1: # broken -> convert to num
								found = False
						if found:
							opt_dict[np.linalg.norm(w_t)] = [w_t, b]

				if w[0] < 0:
					optimized = True
					print('optimized a step')
				else:
					w = w - step 

			norms = sorted([n for n in opt_dict])
			opt_choice = opt_dict[norms[0]]

			self.w = opt_choice[0]
			self.b = opt_choice[1]
			latest_optimum = opt_choice[0][0] + step * 2



	def predict(self, X_test):
		cls = np.sign(np.dot(np.array(X_test), self.w) + self.b)
		return cls

	def _relabel_y(self, y):
		return [i if i == [self.pos_label] else [-1] for i in y]
