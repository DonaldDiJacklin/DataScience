import numpy as np
import scipy.signal as sig

class conv:
	def __init__(self, inputs, outputs):
		self.indims=inputs
		self.filters=4
		self.weights=np.random.rand(5,self.indims,self.filters)
		self.biases=np.random.rand(self.filters)
	def forward(self, A, p = 1):
		self.A=A
		padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
		padA=np.vstack((np.zeros((padrows,self.A.shape[1])),self.A,np.zeros((padrows,self.A.shape[1]))))
		self.H=np.empty((self.A.shape[0],self.filters))
		for k in range(self.filters):
			self.H[:,k]=sig.correlate(padA,self.weights[:,:,k], mode="valid").reshape(-1)
		self.H+=self.biases
		self.Z=ReLU(self.H)
		return self.Z
	def backward(self, D, lr, l1,l2,moment):
		derivterm=D*reluder(self.H)
		D=np.empty((D.shape[0],self.A.shape[1]))
		padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
		padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
		for k in range(self.weights[0,:,0].shape[0]):
			D[:,k]=sig.correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
			for l in range(self.weights[0,0,:].shape[0]):
				self.weights[:,k,l]-=lr*sig.correlate(self.A[:,k],derivterm[:,l], mode="valid")
		self.biases-=lr*(np.sum(derivterm, axis=0))
		return D
	def backwardAdam(self, D, lr, l1,l2,moment, gamma, t):
		derivterm=D*reluder(self.H)
		D=np.empty((D.shape[0],self.A.shape[1]))
		padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
		padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
		for k in range(self.weights[0,:,0].shape[0]):
			D[:,k]=sig.correlate(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
			for l in range(self.weights[0,0,:].shape[0]):
				self.weights[:,k,l]-=lr*sig.correlate(self.A[:,k],derivterm[:,l], mode="valid")
		self.biases-=lr*(np.sum(derivterm, axis=0))
		return D