import numpy as np
import matplotlib.pyplot as plt
def softmax(matrix):
    return np.exp(matrix)/np.sum(np.exp(matrix), axis = 1).reshape(len(matrix[:,0]),1)
def tander(Z):
    return 1-np.multiply(Z,Z)
def relu(Z):
    return np.maximum(Z,0)
def reluder(Z):
    X = Z
    X[X>0] = 1
    X[X<=0] = 0
    return X
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def sigder(Z):
    return np.multiply(sigmoid(Z),1-sigmoid(Z))
def ident(Z):
    return Z
def GCEC(Y,Yhat):
    return -np.sum(np.multiply(Y,np.log(Yhat)))/len(Y[:,0])
def SSE(Y,Yhat):
    return sum(np.multiply(Y-Yhat,Y-Yhat))/len(Y[:,0])
def FroNorm(Y,Yhat):
    return np.trace((Y-Yhat).T@(Y-Yhat))/len(Y[:,0])
def BCEC(Y, Yhat):
    return -np.sum(np.multiply(Y,np.log(Yhat))+np.multiply((1-Y),np.log(1-Yhat)))/len(Y[:,0])
    
class TSNN:
    def __init__(self, nodes = [5,3,6,3],indims = 2,
                 activation = "tanh", taskType = 'regression'):
        self.layers = len(nodes)
        self.indims = indims
#         self.outdims = outdims
        self.nodes = nodes
        self.taskType = taskType
        if activation == "tanh":
            self.activation = np.tanh
            self.actder = tander
        elif activation == "relu":
            self.activation = relu
            self.actder = reluder
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.actder = sigder
        else:
            print("The activation provided is not currently supported.\n Please select one of the following:\n tanh, sigmoid, relu.")
        if taskType == 'regression' or taskType == 'r':
            self.outputactivation = ident
            if self.nodes[-1] == 1:
                self.costfunc = SSE
            else:
                self.costfunc = FroNorm
        elif taskType == 'classification' or taskType == 'c':
            if self.nodes[-1] == 1:
                self.outputactivation = sigmoid
                self.costfunc = BCEC
            else:
                self.outputactivation = softmax
                self.costfunc = GCEC
        else:
            print("Please set taskType equal to classification or regression.")
        self.weights = {}
        self.biases = {}
    def weightInitialization(self):
        self.weights['w0']=np.random.randn(self.indims,self.nodes[0])
        self.biases['b0'] = np.random.randn(1,self.nodes[0])
        for i in range(1,self.layers):
            self.weights['w'+str(i)] = np.random.randn(self.nodes[i-1],self.nodes[i])
            self.biases['b'+str(i)] = np.random.randn(1,self.nodes[i])
    def predict(self, X):
        if(len(self.weights) == 0):
            print("Weights not previously initialized. Initializing now.")
            self.weightInitialization()
        self.Z = {}
        self.Z['0'] = X
        for i in range(1,len(self.nodes)):
            self.Z[str(i)] = self.activation(
                self.Z[str(i-1)]@self.weights['w'+str(i-1)]
                +self.biases['b'+str(i-1)])
        self.probabilities = self.outputactivation(
            self.Z[str(len(self.nodes)-1)]@self.weights['w'+str(
                self.layers-1)] + self.biases['b'+str(self.layers- 1)])
        if self.taskType == 'regression' or self.taskType == 'r':
            self.prediction = self.probabilities
        else:
            self.prediction = np.eye(
                self.probabilities.shape[1])[np.argmax(
                self.probabilities,axis = 1)][:,0,:]
        return self.prediction
    def train(self, X,Y,Xval = [],Yval = [], epochs = 100, learningRate = .00001):
        if(len(self.weights) == 0):
            print("Weights not previously initialized. Initializing now.")
            self.weightInitialization()
        errs = []
        if len(Xval) == 0:
            Xval = X
            Yval = Y
        for i in range(0,epochs):
            js = np.linspace(len(self.weights)-1,0,len(self.weights))
            self.predict(X)
            d = self.probabilities - Y
            for j in range(len(self.weights)-1,-1,-1):
                self.weights['w'+str(j)] = self.weights['w'+str(j)]- learningRate*self.Z[str(j)].T@d
                self.biases['b'+str(j)] = self.biases['b'+str(j)]\
                - learningRate*np.sum(d,axis = 0)
                d = np.multiply(d@self.weights['w'+str(j)].T,
                                self.actder(self.Z[str(j)]))
            self.predict(Xval)
            errs.append(self.costfunc(Yval,self.probabilities))
        plt.plot(errs)
class TSLogisticRegression:
    def __init__(self,features, outputs):
        self.features = features
        self.outputs = outputs
        self.weight = []
        if outputs == 1:
            self.costfunc = BCEC
            self.activation = sigmoid
        else:
            self.costfunc = GCEC
            self.activation = softmax
    def weightInitialization(self):
        self.weight = np.random.randn(self.features+1,self.outputs)
    def predict(self,X):
        if not np.array_equal(
            X[:,0].T,
            np.matrix(np.ones(len(X[:,0].tolist())))):
            X = np.column_stack((np.ones(len(X[:,0])),X))
        if len(self.weight) == 0:
            self.weightInitialization()
        self.probabilities = self.activation(X@self.weight)
        if self.outputs == 1:
            self.prediction = np.round(self.probabilities)
        else:
            self.prediction = np.eye(
                self.probabilities.shape[1])[np.argmax(
                self.probabilities,axis = 1)][:,0,:]
        return self.prediction
    def train(self,X,Y,Xval = [],Yval =[],
              lr = .00001, epochs = 100, l1 = 0, l2 = 0):
        err = []
        if Xval == []:
            Xval = X
            Yval = Y
        X = np.column_stack((np.ones(len(X[:,0])),X))
        Xval = np.column_stack((np.ones(len(Xval[:,0])),Xval))
        for i in range(0,epochs):
            self.predict(X)
            self.weight = self.weight - lr*X.T@(self.probabilities-Y) - l1*np.sign(self.weight) - l2*self.weight
            self.predict(Xval)
            err.append(self.costfunc(Yval,self.probabilities))
        plt.plot(err)