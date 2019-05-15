import numpy as np


def dist(xTrain, xTest):
	return ((xTrain-xTest)**2).sum(1).astype(np.float32)
def smote(outliers,k):
	D = np.empty((outliers.shape[0],outliers.shape[0]))
	for i in range(outliers.shape[0]):
		D[i,:] = dist(outliers, outliers[i,:])
	prednum = np.argsort(D, axis = 1)[:,1:(k+1)]
	outX = outliers.copy()
	for j in range(len(prednum)):
		oldpoint = outliers[j,:]
		r = outliers[prednum[j],:]
		for l in range(len(r)):
			newpoint = oldpoint + np.random.uniform(low = 0,high = 1, size = 1)*(r[l,] - oldpoint)
			outX = np.row_stack((outX,newpoint))
	return outX