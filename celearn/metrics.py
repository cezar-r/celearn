def accuracy_score(arr1, arr2):
	if type(arr1) is not list:
		arr1 = arr1.tolist()
	if type(arr2) is not list:
		arr2 = arr2.tolist()

	correct = 0
	for i, j in list(zip(arr1, arr2)):
		if i == j:
			correct += 1
	return correct / len(arr1)


def rmse(arr1, arr2):
	if type(arr1) is not list:
		arr1 = arr1.tolist()
	if type(arr2) is not list:
		arr2 = arr2.tolist()	

	error = 0
	for i, j in list(zip(arr1, arr2)):
		error += (i - j) ** 2
	return (error / len(arr1))[0]