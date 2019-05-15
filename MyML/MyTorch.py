import numpy as np
import matplotlib.pyplot as plt
from .SupportFunctions import *
import scipy.signal as sig
from .LayerClasses import *


class NeuralNetwork:
	def __init__(self, indims, nodes = [5,5,5,3], activations = ['tanh','tanh','tanh'], taskType = 'c'):
		self.layers = []
		self.indims = indims
		self.nodes = [self.indims]+nodes
		self.activations = activations
		if taskType == 'regression' or taskType == 'r':
			self.task = 'r'
			self.activations = self.activations+['ident']
			if self.nodes[-1] == 1:
				self.costfunc = SSE
			else:
				self.costfunc = FroNorm
		elif taskType == 'classification' or taskType == 'c':
			self.task = 'c'
			if self.nodes[-1] == 1:
				self.activations = self.activations+['sigmoidout']
				self.costfunc = BCEC
			else:
				self.activations = self.activations+['softmax']
				self.costfunc = GCEC
		for i in range(0,len(nodes)):
			if self.activations[i] == 'tanh':
				self.layers.append(Tanh(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] == 'sigmoid':
				self.layers.append(Sigmoid(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] == 'softmax':
				self.layers.append(Softmax(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] == 'relu':
				self.layers.append(Relu(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] == 'sigmoidout':
				self.layers.append(SigmoidOut(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] == 'prelu':
				self.layers.append(Prelu(inputs = self.nodes[i],outputs = self.nodes[i+1]))
			if self.activations[i] == 'ident':
				self.layers.append(Ident(inputs = self.nodes[i],outputs = self.nodes[i+1]))
			if self.activations[i] == 'conv':
				self.layers.append(conv(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] == 'maxpool':
				self.layers.append(maxpool(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] == 'spp':
				self.layers.append(spp(inputs = self.nodes[i], outputs = self.nodes[i+1]))
			if self.activations[i] =='flatten':
				self.layers.append(flatten(inputs = self.nodes[i], outputs = self.nodes[i+1]))

	def predict(self,X, p = 1):
		temp = X.copy()
		for i in range(0,len(self.layers)):
			temp = self.layers[i].forward(temp, p = 1)
		self.probabilities = temp
		if self.task == 'r':
			self.prediction = self.probabilities
		else:
			if self.nodes[-1] == 1:
				self.prediction = np.round(self.probabilities)
			else:
				self.prediction = np.eye(
					self.probabilities.shape[1])[np.argmax(
					self.probabilities,axis = 1)]
		return self.probabilities
	def drop_predict(self,X, p):
		temp = X.copy()
		if self.optim == 'nest' or self.optim == 'adanest' or self.optim == 'rmsnest':
			for i in range(0,len(self.layers)):
				temp = self.layers[i].forwardNest(temp, p, self.moment)
		else:
			for i in range(0,len(self.layers)):
				temp = self.layers[i].forward(temp, p)
		self.probabilities = temp
		if self.task == 'r':
			self.prediction = self.probabilities
		else:
			if self.nodes[-1] == 1:
				self.prediction = np.round(self.probabilities)
			else:
				self.prediction = np.eye(
					self.probabilities.shape[1])[np.argmax(
					self.probabilities,axis = 1)]
		return self.probabilities
	def backProp(self, Y, lr, t):
		temp = Y.copy()
		if self.optim == 'rms':
			for i in range(len(self.layers)-1,-1,-1):
				temp = self.layers[i].backwardRMS(temp,lr = lr, gamma = self.gamma,l1 = self.l1, l2 = self.l2, moment = self.moment)
		elif self.optim == 'ada':
			for i in range(len(self.layers)-1,-1,-1):
				temp = self.layers[i].backwardada(temp,lr, l1 = self.l1,l2 = self.l2,moment = self.moment)
		elif self.optim == 'adam':
			for i in range(len(self.layers)-1,-1,-1):
				temp = self.layers[i].backwardAdam(temp,lr, gamma = self.gamma, moment = self.moment, l1 = self.l1, l2 = self.l2, t =self.t)
		elif self.optim == 'nest':
			for i in range(len(self.layers)-1,-1,-1):
				temp = self.layers[i].backwardNest(temp,lr, moment = self.moment, l1 = self.l1, l2 = self.l2, gamma = self.gamma)
		elif self.optim == 'rmsnest':
			for i in range(len(self.layers)-1,-1,-1):
				temp = self.layers[i].backwardNestRMS(temp,lr, gamma = self.gamma, moment = self.moment, l1 = self.l1, l2 = self.l2)
		elif self.optim == 'adanest':
			for i in range(len(self.layers)-1,-1,-1):
				temp = self.layers[i].backwardNestAda(temp,lr, moment = self.moment, l1 = self.l1, l2 = self.l2, gamma = self.gamma)
		else:
			for i in range(len(self.layers)-1,-1,-1):
				temp = self.layers[i].backward(temp,lr = lr,l1 = self.l1, l2 = self.l2, moment = self.moment)
	def train(self,X,Y,Xval = [],Yval = [],epochs = 10,lr = .0001, gamma = 0, l1 = 0,l2 = 0, moment = 0, p = 1, batchsize = 'full',decay = 0, k = .9,T = 20, optim = 0, calcerr = True):
		if Xval == []:
			Xval = X
			Yval = Y
		self.gamma = gamma
		self.l1 = l1
		self.l2 = l2
		self.moment = moment
		self.errs = []
		self.p = p
		self.optim = optim
		self.decay = decay
		if batchsize != 'full':
			batch = int(np.round(len(X)/batchsize))
			for i in range(0,epochs):
				for j in range(0,batch):
					self.t = i+1
					phat = self.drop_predict(X[(j*batch):(j+1)*batch,],self.p)
					self.backProp(Y[(j*batch):(j+1)*batch,], lr,self.t)
					if calcerr == True:
						self.errs.append(self.costfunc(Yval,self.predict(Xval, p=1)))


		else:
			lrnew = lr
			for i in range(0,epochs):
				self.t = i+1
				phat = self.drop_predict(X, self.p)
				self.backProp(Y, lr, self.t)
				if calcerr == True:
					self.errs.append(self.costfunc(Yval,self.predict(Xval, p=1)))
				if self.decay == 'scheduled':
					lrnew = lr*(k)**(i/T)
				elif self.decay == 'inverse':
					lrnew = lr/(k*i + 1)
				elif self.decay == 'exponential':
					lrnew = lr*np.exp(-k*i)
			# acc = accuracy(self.prediction,Y)
			# if acc == 1.0:
			# 	print(i)
			# 	break
		if calcerr == True:
			plt.plot(self.errs)
			plt.xlabel('Epochs')
			plt.ylabel('Error')
			plt.title('Cost')