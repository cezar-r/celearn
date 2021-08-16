#!/usr/bin/env python
# coding: utf-8

"""
Contains functions that can be used to manipulate data so that it can be used for fitting models
"""

import random

def train_test_split(X, y, test_size = .25):
	"""Splits X and y into training and testing data
	
	Parameters
	----------
	X: 2D list
	y: 2D list
	test_size: float
	
	Returns
	-------
	X_train: 2D list
	X_test: 2D list
	y_train: 2D list
	y_test: 2D list
	"""
	xy = list(zip(X, y))
	random.shuffle(xy)
	X, y = zip(*xy)

	length = int(round(len(X) * test_size, 0))
	X_train = X[length:]
	X_test = X[:length]
	y_train = y[length:]
	y_test = y[:length]
	return X_train, X_test, y_train, y_test


