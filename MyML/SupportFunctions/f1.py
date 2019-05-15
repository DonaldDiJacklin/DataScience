import numpy as np 
from .precision import *
from .recall import *

def f1(preds, Y):
	return 2*precision(preds, Y)*recall(preds,Y)/(precision(preds,Y)+recall(preds,Y))