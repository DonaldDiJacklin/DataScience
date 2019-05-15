import numpy as np
def colorBins(Y):
	cols = ['blue', 'red', 'black', 'gray']
	if type(Y) == type([]):
		if type(Y[0]) == type(3):
			cols = [cols[x] for x in Y]
			return cols
		if type(Y[0]) == type(3.2):
			r = list(map(int,Y))
			cols = [cols[x] for x in r]
			return cols
	if type(Y) == type(np.matrix(np.ones((2,3)))):
		if Y.shape[0] == 1:
			r = Y.astype(int).tolist()[0]
			col = [cols[i] for i in r]
			return col
		if Y.shape[1] == 1:
			r = Y.T.astype(int).tolist()[0]
			col = [cols[i] for i in r]
			return col

