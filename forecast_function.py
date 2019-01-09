from keras.models import Sequential
from keras.layers import Dense, regularizers
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
import math,sys,os
from Functions import create_dataset
from sklearn.model_selection import ParameterGrid



def training(X,grid):
	error = []
	for hyper in range(len(grid)):
		Xtr = X[0:(np.shape(X)[0]-15),:]
		Xtest = X[(np.shape(X)[0]-15 + 1):(np.shape(X)[0]),:]
		pred = lstm_forecast(Xtr, np.shape(Xtest)[0]+1, grid[hyper]['neurons'],grid[hyper]['epoch'], do_cv = False)
		error.append(np.sqrt(np.mean((Xtest - pred)**2)))
	return(error)

def lstm_forecast(dataset, test_length, neurons = 32, epoche = 300, do_cv = True, tstart=0):
	if do_cv:
		reg_path = {'neurons': map(int, np.linspace(8,64,5)), 'epoch': map(int, np.linspace(50,300,5))}
		reg_path = list(ParameterGrid(reg_path))
		training_path = training(dataset,reg_path)
		neurons = reg_path[np.argmin(training_path)]['neurons']
		epoche = reg_path[np.argmin(training_path)]['epoch']
	validation_length = 10
	train_length = np.shape(dataset)[0] - validation_length
	### Take the training set
	ts_training = dataset
	scaler_ts_training = preprocessing.StandardScaler().fit(ts_training)
	ts_training = preprocessing.scale(ts_training)
	num_species = ts_training.shape[1]
	ts_training_original = ts_training
	#### Reshape into X=t and Y=t+look_back
	look_back = 1
	### Here you create an array Ytrain with the column to predict scale by look_back points (e.g.,, 1)
	ts_training_tr = ts_training[0:train_length,:]
	ts_training_vl = ts_training[train_length:(train_length + validation_length),:]
	trainX, trainY = create_dataset(ts_training_tr, look_back)
	ValX, ValY = create_dataset(ts_training_vl, look_back)
	####################################################################################
	# reshape input to be [samples, time steps, features]
	trainX = trainX.reshape((trainX.shape[0], 1, num_species))
	ValX = ValX.reshape((ValX.shape[0], 1, num_species))
	#### Take last point of the training set and start predictions from there
	ts_training_reshaped = ts_training_original.reshape((ts_training_original.shape[0], 1, num_species))
	last_point_kept = np.array(ts_training_reshaped[(np.shape(ts_training_reshaped)[0] - 1), 0, :])
	#####################################################################################

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(neurons, input_shape=(look_back,  num_species)))
	### Decide whether to use sparsity or not
	model.add(Dense(num_species, activation = 'linear', activity_regularizer=regularizers.l2(10e-5)))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	sys.stdout = open(os.devnull, "w")
	model.fit(trainX, trainY, epochs=epoche, validation_data = (ValX, ValY), verbose=0)
	sys.stdout = sys.__stdout__

	# make predictions point by point starting from the last point of the training set
	length_predictions = test_length
	realizations = 30
	next_point = np.zeros((length_predictions,num_species))
	for prd in range(realizations):
		##### Last point of the training set for predictions
		last_point  = last_point_kept
		for i in range(0,length_predictions):
			last_point = last_point.reshape((1, 1, num_species))
			last_point = model.predict(last_point)
			next_point[i,:] = next_point[i,:] + last_point
	next_point = next_point/realizations
	next_point = np.delete(next_point, (0), 0)
	next_point = scaler_ts_training.inverse_transform(next_point)
	return(next_point)
