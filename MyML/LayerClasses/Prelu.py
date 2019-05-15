import numpy as np
from ..SupportFunctions import *


class Prelu:
	def __init__(self, inputs, outputs):
		self.inputs = inputs
		self.outputs = outputs
		self.weights = np.random.randn(self.inputs, self.outputs)*np.sqrt(2/(self.inputs+self.outputs))
		self.biases = np.random.randn(1,self.outputs)*np.sqrt(2/(self.inputs+self.outputs))
		self.p = np.ones((1,self.outputs))
		self.Gw = np.ones((self.inputs,self.outputs))
		self.Gb = np.ones((1,self.outputs))
		self.Gp = np.ones((1, self.outputs))
		self.mw = np.zeros((self.inputs,self.outputs))
		self.mb = np.zeros((1,self.outputs))
		self.mp = np.zeros((1,self.outputs))
	def forward(self, A, p = 1):
		self.A = A*(np.random.rand(A.shape[0], A.shape[1])<p)/p
		self.G = self.A@self.weights
		self.H = self.G+self.biases
		self.Z = prelu(self.H, self.p)
		return self.Z
	def backward(self, D, lr, l1,l2,moment):
		self.D = D
		self.Gw = self.Gw + (self.A.T@(self.D*deprelua(self.H,self.p)) + l1*np.sign(self.weights)+l2*self.weights)**2
		self.etaw = lr/np.sqrt(self.Gw + .000000001)
		self.mw = moment*self.mw - lr*(self.A.T@(self.D*deprelua(self.H,self.p)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights + self.mw
		self.Gb = self.Gb+(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) + l2*self.biases)**2
		self.etab = lr/np.sqrt(self.Gb + .000000001)
		self.mb = moment*self.mb - lr*(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) +l2*self.biases)
		self.biases = self.biases +self.mb
		self.Gp = self.Gp+(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)**2
		self.etap = lr/np.sqrt(self.Gp+.0000000001)
		self.mp = moment*self.mp - lr*(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)
		self.p = self.p + self.mp
		return (self.D*deprelua(self.H, self.p))@self.weights.T
	def backwardRMS(self, D, lr,l1,l2,gamma, moment):
		self.D = D
		self.Gw = gamma*self.Gw + (1-gamma)*(self.A.T@(self.D*deprelua(self.H,self.p)) + l1*np.sign(self.weights)+l2*self.weights)**2
		self.etaw = lr/np.sqrt(self.Gw + .000000001)
		self.mw = moment*self.mw - self.etaw*(self.A.T@(self.D*deprelua(self.H,self.p)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights + self.mw
		self.Gb = gamma*self.Gb+(1-gamma)*(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) + l2*self.biases)**2
		self.etab = lr/np.sqrt(self.Gb + .000000001)
		self.mb = moment*self.mb - self.etab*(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) +l2*self.biases)
		self.biases = self.biases +self.mb
		self.Gp = gamma*self.Gp+(1-gamma)*(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)**2
		self.etap = lr/np.sqrt(self.Gp+.0000000001)
		self.mp = moment*self.mp - self.etap*(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)
		self.p = self.p + self.mp
		return (self.D*deprelua(self.H, self.p))@self.weights.T
	def backwardada(self, D, lr, l1,l2,moment):
		self.D = D
		self.Gw = self.Gw + (self.A.T@(self.D*deprelua(self.H,self.p)) + l1*np.sign(self.weights)+l2*self.weights)**2
		self.etaw = lr/np.sqrt(self.Gw + .000000001)
		self.mw = moment*self.mw - self.etaw*(self.A.T@(self.D*deprelua(self.H,self.p)) + l1*np.sign(self.weights) + l2*self.weights)
		self.weights = self.weights + self.mw
		self.Gb = self.Gb+(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) + l2*self.biases)**2
		self.etab = lr/np.sqrt(self.Gb + .000000001)
		self.mb = moment*self.mb - self.etab*(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) +l2*self.biases)
		self.biases = self.biases +self.mb
		self.Gp = self.Gp+(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)**2
		self.etap = lr/np.sqrt(self.Gp+.0000000001)
		self.mp = moment*self.mp - self.etap*(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)
		self.p = self.p + self.mp
		return (self.D*deprelua(self.Z, self.p))@self.weights.T
	def backwardAdam(self, D, lr, l1, l2, moment, gamma, t):
		self.D = D
		self.mw = moment*self.mw + (1-moment)*(self.A.T@(self.D*deprelua(self.H,self.p)) + l1*np.sign(self.weights)+l2*self.weights)
		self.Gw = gamma*self.Gw+(1-gamma)*((self.A.T@(self.D*deprelua(self.H,self.p))) + l1*np.sign(self.weights)+l2*self.weights)**2
		mhatw = self.mw/(1+moment**t)
		ghat = self.Gw/(1+gamma**t)
		self.etaw = lr/np.sqrt(ghat+.000000001)
		self.weights = self.weights - self.etaw*mhatw

		self.mb = moment*self.mb + (1-moment)*(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) + l2*self.biases)
		self.Gb = gamma*self.Gb + (1-gamma)*(np.sum(self.D*deprelua(self.H, self.p)) + l1*np.sign(self.biases) + l2*self.biases)**2
		mhatb = self.mb/(1+moment**t)
		ghatb = self.Gb/(1+gamma**t)
		self.etab = lr/np.sqrt(ghatb+.000000001)
		self.biases = self.biases - self.etab*mhatb

		self.mp = moment*self.mp + (1-moment)*(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)
		self.Gp = gamma*self.Gp + (1-gamma)*(np.sum(self.D*deprelup(self.H)) + l1*np.sign(self.p) + l2*self.p)**2
		mhatp = self.mp/(1+moment**t)
		ghatp = self.Gp/(1+gamma**t)
		self.etap = lr/np.sqrt(ghatp+.000000001)
		self.p = self.p - self.etap*mhatp
		return (self.D*deprelua(self.H, self.p))@self.weights.T