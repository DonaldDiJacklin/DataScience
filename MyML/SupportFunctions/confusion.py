import numpy as np

def confusion(preds,Y):
	truePos = np.sum(np.multiply(preds,Y))
	falsePos = np.sum(np.multiply(preds, 1-Y))
	trueNeg = np.sum(np.multiply(1-preds,1-Y))
	falseNeg = np.sum(np.multiply(1-preds,Y))
	conf = np.row_stack(([trueNeg,falsePos],[falseNeg,truePos]))
	return conf