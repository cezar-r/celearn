#!/usr/bin/env python
# coding: utf-8

"""Math operations that are commonly used"""

def mode(arr):
	"""Finds the most common element in a list
	
	Parameters
	----------
	arr: list
	
	Retunrs
	-------
	sorted(list(counts.items()), key = lambda x : x[1])[::-1][0][0]: Any type
	"""
	counts = {}
	for i in arr:
		if i in counts:
			counts[i] += 1
		else:
			counts[i] = 1
	return sorted(list(counts.items()), key = lambda x : x[1])[::-1][0][0]
