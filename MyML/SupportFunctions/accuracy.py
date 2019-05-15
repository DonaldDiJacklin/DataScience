import numpy as np
def accuracy(preds,Y):
	return np.mean(preds == Y)