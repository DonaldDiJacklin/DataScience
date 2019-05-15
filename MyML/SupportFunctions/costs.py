import numpy as np
def GCEC(Y,Yhat):
    return -np.sum(np.multiply(Y,np.log(Yhat)))/len(Y[:,0])
def SSE(Y,Yhat):
    return np.sum(np.multiply(Y-Yhat,Y-Yhat))/len(Y)
def FroNorm(Y,Yhat):
    return np.trace((Y-Yhat).T@(Y-Yhat))/len(Y[:,0])



def BCEC(Y, Yhat):
    return -np.sum(np.multiply(Y,np.log(Yhat))+np.multiply((1-Y),np.log(1-Yhat)))/len(Y)