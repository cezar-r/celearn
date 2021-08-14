def mode(arr):
	counts = {}
	for i in arr:
		if i in counts:
			counts[i] += 1
		else:
			counts[i] = 1
	return sorted(list(counts.items()), key = lambda x : x[1])[::-1][0][0]