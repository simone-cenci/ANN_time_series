import numpy as np
def MakeNoise(X, percentage_std):
        ### Input: X = training set of the time series
        ###        percentage_std = percentage of standard deviation for the noise [0,1]
        X = X + np.random.normal(loc=0.0, scale=percentage_std*np.std(X[:,1]), size= X.shape)
	return(X)
