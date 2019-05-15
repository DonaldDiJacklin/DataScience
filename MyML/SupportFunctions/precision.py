import numpy as np
def precision(preds, Y):
	return (np.sum(np.multiply(preds,Y), axis = 0)/np.sum(preds,axis = 0)).tolist()[0]