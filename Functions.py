import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math,sys,os



### Function to create the lagged dataset
def create_dataset(dtset, look_back=1):
	### Input dtset = time series
	### look_back = time lage of predictions
	### Output: two data set one the same and one lagged
        dataX = np.zeros((np.shape(dtset)[0] - look_back - 1, np.shape(dtset)[1]))
        dataY0 = []; dataY1 = []; dataY2 = []; dataY3 = [];
        dataY = np.zeros((np.shape(dtset)[0] - look_back - 1, np.shape(dtset)[1]))
        for i in range(np.shape(dtset)[0] - look_back - 1):
                dataX[i,:] = dtset[i:(i+look_back), :]
                dataY[i,:] = dtset[i+look_back,:]
        return np.array(dataX), np.array(dataY)
def plot_function(train, test, predicted):
        all_data = np.concatenate((train,test), axis = 0)

	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	interval_forecast = range((np.shape(train)[0]+1), np.shape(train)[0]+1+np.shape(test)[0])

	ax1.plot(all_data[:,0], color = 'b')
	ax1.plot(interval_forecast, predicted[:,0], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax1.axvline(x = (np.shape(train)[0]+1), lw = 2, ls = '--')
	ax1.legend()


	ax2.plot(all_data[:,1], color = 'b')
	ax2.plot(interval_forecast, predicted[:,1], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax2.axvline(x = (np.shape(train)[0]+1), lw = 2, ls = '--')


	ax3.plot(all_data[:,2], color = 'b')
	ax3.plot(interval_forecast, predicted[:,2], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax3.axvline(x = (np.shape(train)[0]+1), lw = 2, ls = '--')


	ax4.plot(all_data[:,3], color = 'b')
	ax4.plot(interval_forecast, predicted[:,3], lw = 2, linestyle = '--', color = 'r', label = 'Forecast')
	ax4.axvline(x = (np.shape(train)[0]+1), lw = 2, ls = '--')
	plt.show()

