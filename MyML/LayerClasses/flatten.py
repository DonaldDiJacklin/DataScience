import numpy as np
import scipy.signal as sig

class flatten:
	def __init__(self, inputs, outputs):
		self.weights = []
		self.biases = []
	def forward(self, A, p = 1):
		self.flattenedthing = A.flatten()
		self.shape = A.shape
		return self.flattenedthing.reshape(1,-1)
	def backward(self, Dy, lr,l1,l2, moment):
		return Dy.reshape(*self.shape)
	def backwardAdam(self, Dy, lr, l1,l2,moment, gamma, t):
		self.newD = Dy.reshape(*self.shape)
		return self.newD