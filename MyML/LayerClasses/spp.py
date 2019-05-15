import numpy as np
import scipy.signal as sig

class spp:
	def __init__(self, inputs, outputs):
		self.weights = []
		self.biases = []
		for i in range(0,32):
			self.weights.append(np.random.randn(5,inputs))
			self.biases.append(np.random.randn(1))
	def forward(self, A, p = 1):
		self.dims=A.shape
		o1=int(A.shape[1])
		out4 = np.empty((4,o1))
		locations4 = np.empty((4,o1))
		cX=0
		s=int(A.shape[0]/4)
		for i in range(0,A.shape[0],s):
			out4[cX,:] = np.max(A[i:i+s,:], axis=0)
			locations4[cX,:]=np.argmax(A[i:i+s,:], axis=0)
			cX+=1

		out2 = np.empty((2,o1))
		locations2 = np.empty((2,o1))
		cX=0
		s=int(A.shape[0]/2)
		for i in range(0,A.shape[0],s):
			out2[cX,:] = np.max(A[i:i+s,:], axis=0)
			locations2[cX,:]=np.argmax(A[i:i+s,:], axis=0)
			cX+=1

		out1 = np.max(A, axis=0)
		location1=np.argmax(A, axis=0)

		self.locations=np.vstack((locations4.astype(int),locations2.astype(int),location1.astype(int)))
		return np.vstack((out4,out2,out1))
	def backward(self, Dy,lr,l1,l2, moment):
		D = np.zeros(self.dims)
		s=int(self.dims[0]/4)
		cX=0
		for i in range(0,self.dims[0],s):
			for j in range(self.dims[1]):
				D[i:i+s,j][self.locations[cX,j]] = Dy[cX,j]
			cX+=1
		return D
	def backwardAdam(self, Dy, lr, l1,l2,moment, gamma, t):
		D = np.zeros(self.dims)
		s=int(self.dims[0]/4)
		cX=0
		for i in range(0,self.dims[0],s):
			for j in range(self.dims[1]):
				D[i:i+s,j][self.locations[cX,j]] = Dy[cX,j]
			cX+=1
			self.D = D
		return D