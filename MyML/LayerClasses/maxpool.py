import numpy as np
import scipy.signal as sig

class maxpool:
	def __init__(self, inputs, outputs):
		self.weights = []
		self.biases = []
		for i in range(0,32):
			self.weights.append(np.random.randn(5,inputs))
			self.biases.append(np.random.randn(1))
	def forward(self,A, p = 1):
		o0=int(A.shape[0]/2)
		o1=int(A.shape[1])

		Z = np.empty((o0,o1))
		self.locations = np.empty((o0,o1))
		cX=0
		for i in range(0,A.shape[0],2):
			Z[cX,:] = np.max(A[i:i+2,:], axis=0)
			self.locations[cX,:]=np.argmax(A[i:i+2,:], axis=0)
			cX+=1
		return Z

	def backward(self, Dy,lr,l1,l2, moment):
		o0=int(Dy.shape[0]*2)
		o1=int(Dy.shape[1])

		D = np.zeros((o0,o1))
		cX=0
		for i in range(0,o0,2):
			for j in range(o1):
				D[i:i+2,j][self.locations[cX,j].astype(int)] = Dy[cX,j]
			cX+=1
		return D
	def backwardAdam(self, Dy, lr, l1,l2,moment, gamma, t):
		o0=int(Dy.shape[0]*2)
		o1=int(Dy.shape[1])

		D = np.zeros((o0,o1))
		cX=0
		for i in range(0,o0,2):
			for j in range(o1):
				D[i:i+2,j][self.locations[cX,j].astype(int)] = Dy[cX,j]
			cX+=1
		return D