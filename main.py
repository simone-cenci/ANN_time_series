import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math,sys,os
import random as rnd
from forecast_function import lstm_forecast
from Functions import plot_function

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#data  = np.loadtxt('deterministic_chaos.txt')
data  = np.loadtxt('CR.txt')

num_data_for_training = (300+1)
num_data_to_predict = 30
tstart=int(np.random.uniform(0,100,1))
training_data = data[tstart:(tstart+num_data_for_training),:]
test_data = data[(tstart+num_data_for_training):(tstart+num_data_for_training+num_data_to_predict)]
forecast = lstm_forecast(training_data, num_data_to_predict+1,do_cv = True)

print('Correlation coefficient:', np.mean([pearsonr(test_data[:,i],forecast[:,i])[0] for i in range(np.shape(forecast)[1])])  )
print('RMSE:', np.sqrt(np.mean((test_data - forecast)**2)))
########################################################################################################
plot = True
if plot == True:
	plot_function(training_data, test_data, forecast)
