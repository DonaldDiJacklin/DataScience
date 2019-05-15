import numpy as np
from ..SupportFunctions import *


class Sigmoid:
	def __init__(self, inputs = 5, outputs = 5):
		self.inputs = inputs
		self.outputs = outputs
		self.weights = np.random.randn(self.inputs, self.outputs)*np.sqrt(2/(self.inputs+self.outputs))
		self.biases = np.random.randn(1,self.outputs)*np.sqrt(2/(self.inputs+self.outputs))
		self.mw = np.zeros((self.inputs,self.outputs))
		self.mb = np.zeros((1,self.outputs))
		self.Gw = np.ones((self.inputs,self.outputs))
		self.Gb = np.ones((1,self.outputs))
		self.mw = np.zeros((self.inputs,self.outputs))
		self.mb = np.zeros((1,self.outputs))
	def forward(self, A, p = 1):
		self.A = A*(np.random.rand(A.shape[0], A.shape[1])<p)/p
		self.G = self.A@self.weights
		self.H = self.G+self.biases
		self.Z = sigmoid(self.H)
		return self.Z
	def forwardNest(self, A,p = 1, moment = .8):
		self.A = A*(np.random.rand(A.shape[0], A.shape[1])<p)/p
		self.weights = self.weights+moment*self.mw
		self.biases = self.biases + moment*self.mb
		self.G = self.A@self.weights
		self.H = self.G+self.biases
		self.Z = sigmoid(self.H)
		return self.Z
	def backward(self, D, lr, l1,l2,moment):
		self.D = D
		self.weights = self.weights - lr*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)
		self.biases = self.biases - lr*(np.sum(self.D*sigder(self.Z), axis = 0)  + l1*np.sign(self.biases) + l2*self.biases)
		return (self.D*sigder(self.Z))@self.weights.T
	def backwardada(self, D, lr, l1,l2,moment):
		self.D = D
		self.Gw = self.Gw + (self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)**2
		self.etaw = lr/np.sqrt(self.Gw + .000000001)
		self.mw = moment*self.mw - self.etaw*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights + self.mw
		self.Gb = self.Gb+(np.sum(self.D*sigder(self.Z), axis = 0) + l1*np.sign(self.biases) + l2*self.biases)**2
		self.etab = lr/np.sqrt(self.Gb + .000000001)
		self.mb = moment*self.mb - self.etab*(np.sum(self.D*sigder(self.Z)) + l1*np.sign(self.biases) +l2*self.biases)
		self.biases = self.biases +self.mb
		return (self.D*sigder(self.Z))@self.weights.T
	def backwardRMS(self, D, lr,l1,l2,gamma, moment):
		self.D = D
		self.Gw = gamma*self.Gw + (1-gamma)*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)**2
		self.etaw = lr/np.sqrt(self.Gw + .000000001)
		self.mw = moment*self.mw - self.etaw*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights + self.mw
		self.Gb = gamma*self.Gb+(1-gamma)*(np.sum(self.D*sigder(self.Z)) + l1*np.sign(self.biases) + l2*self.biases)**2
		self.etab = lr/np.sqrt(self.Gb + .000000001)
		self.mb = moment*self.mb - self.etab*(np.sum(self.D*sigder(self.Z)) + l1*np.sign(self.biases) +l2*self.biases)
		self.biases = self.biases +self.mb
		return (self.D*sigder(self.Z))@self.weights.T
	def backwardAdam(self, D, lr, l1, l2, moment, gamma, t):
		self.D = D
		self.mw = moment*self.mw + (1-moment)*((self.A.T@(self.D*sigder(self.Z)))+ l1*np.sign(self.weights)+l2*self.weights)
		self.Gw = gamma*self.Gw+(1-gamma)*((self.A.T@(self.D*sigder(self.Z))) + l1*np.sign(self.weights) + l2*self.weights)**2
		mhatw = self.mw/(1+moment**t)
		ghat = self.Gw/(1+gamma**t)
		self.etaw = lr/np.sqrt(ghat+.000000001)
		self.weights = self.weights - self.etaw*mhatw

		self.mb = moment*self.mb + (1-moment)*(((np.sum(self.D*sigder(self.Z)))+ l1*np.sign(self.biases)+l2*self.biases))
		self.Gb = gamma*self.Gb + (1-gamma)*(np.sum(self.D*sigder(self.Z)) + l1*np.sign(self.biases) + l2*self.biases)**2
		mhatb = self.mb/(1+moment**t)
		ghatb = self.Gb/(1+gamma**t)
		self.etab = lr/np.sqrt(ghatb+.000000001)
		self.biases = self.biases - self.etab*mhatb

		return (self.D*sigder(self.Z))@self.weights.T
	def backwardNestAda(self, D, lr,l1,l2,moment, gamma):
		self.D = D
		mw = self.mw
		mb = self.mb
		self.Gw = self.Gw + (self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)**2
		self.etaw = lr/np.sqrt(self.Gw + .000000001)
		self.mw = moment*self.mw - self.etaw*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights - moment*mw
		self.weights += self.mw
		self.Gb = self.Gb+(np.sum(self.D*sigder(self.Z)) + l1*np.sign(self.biases) + l2*self.biases)**2
		self.etab = lr/np.sqrt(self.Gb + .000000001)
		self.mb = moment*self.mb - self.etab*(np.sum(self.D*sigder(self.Z), axis = 0)  + l1*np.sign(self.biases) + l2*self.biases)
		self.biases -= moment*mb
		self.biases = self.biases + self.mb
		return (self.D*sigder(self.Z))@self.weights.T
	def backwardNestRMS(self, D, lr,l1,l2,moment, gamma):
		self.D = D
		mw = self.mw
		mb = self.mb
		self.Gw = gamma*self.Gw + (1-gamma)*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)**2
		self.etaw = lr/np.sqrt(self.Gw + .000000001)
		self.mw = moment*self.mw - self.etaw*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights - moment*mw
		self.weights += self.mw
		self.Gb = gamma*self.Gb+(1-gamma)*(np.sum(self.D*sigder(self.Z)) + l1*np.sign(self.biases) + l2*self.biases)**2
		self.etab = lr/np.sqrt(self.Gb + .000000001)
		self.mb = moment*self.mb - self.etab*(np.sum(self.D*sigder(self.Z), axis = 0)  + l1*np.sign(self.biases) + l2*self.biases)
		self.biases -= moment*mb
		self.biases = self.biases + self.mb
		return (self.D*sigder(self.Z))@self.weights.T
	def backwardNest(self, D, lr,l1,l2,moment, gamma):
		self.D = D
		mw = self.mw
		mb = self.mb
		self.mw = moment*self.mw - lr*(self.A.T@(self.D*sigder(self.Z)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights - moment*mw
		self.weights += self.mw
		self.mb = moment*self.mb - lr*(np.sum(self.D*sigder(self.Z), axis = 0)  + l1*np.sign(self.biases) + l2*self.biases)
		self.biases -= moment*mb
		self.biases = self.biases + self.mb
		return (self.D*sigder(self.Z))@self.weights.T