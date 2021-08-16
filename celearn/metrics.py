#!/usr/bin/env python
# coding: utf-8

"""Contains various metrics to measure reliability of models"""

def accuracy_score(pred_arr, true_arr):
	"""Accuracy formula: True Positve / Length of Data
	
	Parameters
	----------
	pred_arr: list
	true_arr: list
	
	Returns
	-------
	tp / len(pred_arr): float
	"""
	pred_arr, true_arr = _to_list(pred_arr, true_arr)

	tp = 0
	for i, j in list(zip(pred_arr, true_arr)):
		if i == j:
			tp += 1
	return tp / len(pred_arr)


def precision_score(pred_arr, true_arr, pos_label = 1):
	"""Precision formula: True Positive / (True Positive + False Positive)
	
	Parameters
	----------
	pred_arr: list
	true_arr: list
	pos_label: any type
	
	Returns
	-------
	tp / (tp + fp): float
	"""
	pred_arr, true_arr = _to_list(pred_arr, true_arr)

	tp = 0
	fp = 0
	for i, j in list(zip(pred_arr, true_arr)):
		if i == j:
			tp += 1
		elif i[0] == pos_label and j[0] != pos_label:
			fp += 1
	return tp / (tp + fp)


def recall_score(pred_arr, true_arr, pos_label = 1):
	"""Recall formula: True Positive / (True Positive + False Negative)
	
	Parameters
	----------
	pred_arr: list
	true_arr: list
	pos_label: any type
	
	Returns
	-------
	tp / (tp + fn): float
	"""
	pred_arr, true_arr = _to_list(pred_arr, true_arr)

	tp = 0
	fn = 0
	for i, j in list(zip(pred_arr, true_arr)):
		if i == j:
			tp += 1
		elif j[0] == pos_label and i[0] != pos_label:
			fn += 1
	return tp / (tp + fn)


def f1_score(pred_arr, true_arr, pos_label = 1):
	"""F1 formula: 2 ((Precision * Recall) / (Precision + Recall))
	
	Parameters
	----------
	pred_arr: list
	true_arr: list
	pos_label: any type
	
	Returns
	-------
	2 * ((precision * recall) / (precision + recall)): float
	"""
	pred_arr, true_arr = _to_list(pred_arr, true_arr)

	recall = recall_score(pred_arr, true_arr, pos_label)
	precision = precision_score(pred_arr, true_arr, pos_label)
	return 2 * ((precision * recall) / (precision + recall))


def classification_report(pred_arr, true_arr):
	"""Returns a classification report for each label
	Report contains precison, recall, f1 and number of classes in total

	Parameters
	----------
	pred_arr: list
	true_arr: list
	
	Returns
	-------
	report: ClassificationReport()
	"""
	pred_arr, true_arr = _to_list(pred_arr, true_arr)

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


def mean_squared_error(pred_arr, true_arr):
	"""Mean Squared Error Formula: sum((pred_arr_i - true_arr_i)**2)
	
	Parameters
	----------
	pred_arr: list
	true_arr: list
	
	Returns
	-------
	error / len(pred_arr): float
	"""
	pred_arr, true_arr = _to_list(pred_arr, true_arr)	

	error = 0
	for i, j in list(zip(pred_arr, true_arr)):
		error += (i[0] - j[0]) ** 2
	return error / len(pred_arr)


def _to_list(pred_arr, true_arr):
	"""Converts non-list types (such as np arrays and pandas Series and DataFrames) to lists
	
	Parameters
	----------
	pred_arr: list
	true_arr: list
	
	Returns
	-------
	pred_arr: list
	true_arr: list
	"""
	if type(pred_arr) is not list:
		pred_arr = pred_arr.tolist()
	if type(true_arr) is not list:
		true_arr = true_arr.tolist()

	return pred_arr, true_arr


class ClassificationReport:
	"""Object in which returns a formatted string, which can be accessed as a dictionary as well"""

	def __init__(self, labels, precision_scores, recall_scores, f1_scores, support):
		"""Initalizes the scores for each metric

		Parameters
		----------
		labels: list
		precision_scores: list
		recall_scores: list
		f1_scores: list
		support: list
		"""
		self.labels = labels
		self.precision_scores = precision_scores
		self.recall_scores = recall_scores
		self.f1_scores = f1_scores
		self.support = support
		self._stringify()
		self._output_dict()

	def __repr__(self):
		"""Creates formatted string
		
		Returns
		-------
		retval: str
		"""
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
		"""Converts each element in each list to string"""
		self._labels = [str(i) for i in self.labels]
		self._precision_scores = [str(i) for i in self.precision_scores]
		self._recall_scores = [str(i) for i in self.recall_scores]
		self._f1_scores = [str(i) for i in self.f1_scores]
		self._support = [str(i) for i in self.support]

	def _output_dict(self):
		"""Creates dictionary that contains all of the data
		
		Returns
		-------
		self.as_dict: dict
		"""
		self.as_dict = {}
		for i, label in enumerate(self.labels):
			self.as_dict[label] = {'precision' : self.precision_scores[i], 'recall' : self.recall_scores[i], 'f1-score' : self.f1_scores[i], 'support' : self.support[i]}
		return self.as_dict
	
	
