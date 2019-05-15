import numpy as np 


def sample(matrix, obs):
	indices = np.random.choice(len(matrix), obs)
	return(matrix[indices,])