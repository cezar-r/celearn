def accuracy_score(pred_arr, true_arr):
	pred_arr, true_arr = _convert(pred_arr, true_arr)

	tp = 0
	for i, j in list(zip(pred_arr, true_arr)):
		if i == j:
			tp += 1
	return tp / len(pred_arr)


def precision_score(pred_arr, true_arr, pos_label = 1):
	pred_arr, true_arr = _convert(pred_arr, true_arr)

	tp = 0
	fp = 0
	for i, j in list(zip(pred_arr, true_arr)):
		if i == j:
			tp += 1
		elif i[0] == pos_label and j[0] != pos_label:
			fp += 1
	return tp / (tp + fp)


def recall_score(pred_arr, true_arr, pos_label = 1):
	pred_arr, true_arr = _convert(pred_arr, true_arr)

	tp = 0
	fn = 0
	for i, j in list(zip(pred_arr, true_arr)):
		if i == j:
			tp += 1
		elif j[0] == pos_label and i[0] != pos_label:
			fn += 1
	return tp / (tp + fn)


def f1_score(pred_arr, true_arr, pos_label = 1):
	pred_arr, true_arr = _convert(pred_arr, true_arr)

	recall = recall_score(pred_arr, true_arr, pos_label)
	precision = precision_score(pred_arr, true_arr, pos_label)
	return 2 * ((precision * recall) / (precision + recall))


def classification_report(pred_arr, true_arr):
	pred_arr, true_arr = _convert(pred_arr, true_arr)

	labels = [i[0] for i in true_arr]
	labels_set = list(set(labels))
	precision_scores = []
	recall_scores = []
	f1_scores = []
	support = []

	for label in labels_set:
		precision_scores.append(round(precision_score(pred_arr, true_arr, label), 2))
		recall_scores.append(round(recall_score(pred_arr, true_arr, label), 2))
		f1_scores.append(round(f1_score(pred_arr, true_arr, label), 2))
		support.append(labels.count(label))

	report = ClassificationReport(labels_set, precision_scores, recall_scores, f1_scores, support)
	return report


def rmse(pred_arr, true_arr):
	pred_arr, true_arr = _convert(pred_arr, true_arr)	

	error = 0
	for i, j in list(zip(pred_arr, true_arr)):
		error += (i[0] - j[0]) ** 2
	return error / len(pred_arr)


def _convert(pred_arr, true_arr):
	if type(pred_arr) is not list:
		pred_arr = pred_arr.tolist()
	if type(true_arr) is not list:
		true_arr = true_arr.tolist()

	return pred_arr, true_arr




class ClassificationReport:

	def __init__(self, labels, precision_scores, recall_scores, f1_scores, support):
		self.labels = labels
		self.precision_scores = precision_scores
		self.recall_scores = recall_scores
		self.f1_scores = f1_scores
		self.support = support
		self._stringify()
		self._output_dict()

	def __repr__(self):

		length = len(max(self._labels + self._precision_scores + self._recall_scores + self._f1_scores + self._support, key = len)) + 4
		if length < 9:
			length = 12
		f = '{0:>%d} {1:>%d} {2:>%d} {3:>%d} {4:>%d}\n' % (length, length, length, length, length)
		retval = f.format("labels", "precision", "recall", "f1-score", "support")
		retval += '\n'
		for i in range(len(self.labels)):
			retval += f.format(self.labels[i], self.precision_scores[i], self.recall_scores[i], self.f1_scores[i], self.support[i])
		return retval

	def _stringify(self):
		self._labels = [str(i) for i in self.labels]
		self._precision_scores = [str(i) for i in self.precision_scores]
		self._recall_scores = [str(i) for i in self.recall_scores]
		self._f1_scores = [str(i) for i in self.f1_scores]
		self._support = [str(i) for i in self.support]

	def _output_dict(self):
		self.as_dict = {}
		for i, label in enumerate(self.labels):
			self.as_dict[label] = {'precision' : self.precision_scores[i], 'recall' : self.recall_scores[i], 'f1-score' : self.f1_scores[i], 'support' : self.support[i]}
		return self.as_dict




if __name__ == '__main__':

	y_test = [[1], [2], [3], [4], [1], [2], [3]]
	y_pred = [[1], [2], [3], [3], [1], [2], [3]]

	report = classification_report(y_pred, y_test)
	print(report)
	print(report.as_dict)

