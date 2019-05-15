import numpy as np
def recall(preds,Y):
	return (np.sum(np.multiply(preds,Y), axis = 0)/np.sum(Y,axis = 0)).tolist()[0]
