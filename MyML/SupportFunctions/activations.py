import numpy as np
def softmax(matrix):
    return np.exp(matrix)/np.sum(np.exp(matrix), axis = 1).reshape(len(matrix[:,0]),1)
def tander(Z):
    return 1-np.multiply(Z,Z)
def ReLU(Z):
    R = Z
    return np.multiply(R,(R>0))
def reluder(Z):
    return Z>0
def sigmoid(Z):
    
    return 1/(1+np.exp(-Z))
def sigder(Z):
    return np.multiply(sigmoid(Z),1-sigmoid(Z))
def ident(Z):
    return Z
def LeakReLU(x):
    R = x
    return np.multiply(R,(R>0))+.2*np.multiply(R,(R<=0))
def lerelder(x):
    return x>0 + .2*(x<=0)
def prelu(x,p):
    R = x
    return np.multiply(R,(R>0))+np.multiply(p,np.multiply(R,(R<=0)))
def deprelua(x,p):
    R = x
    return (R>0).astype(float) + np.multiply(p,(R<=0).astype(float))
def deprelup(x):
    return np.multiply(0,x>0)+np.multiply(x,(x<=0).astype(float))
