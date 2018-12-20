import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math,sys,os
import random as rnd
from keras.models import Sequential
from keras.layers import Dense, regularizers
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Functions import AE, create_dataset
from MakeNoise import MakeNoise
################################################################################################
### run as python -W ignore main.py DataName.txt 0/1 (for no/yes observational noise) tstart ###
### For example: python -W ignore main.py deterministic_chaos.txt 0 0		             ###
################################################################################################
# load the dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NomeFile = sys.argv[1]
f = open(NomeFile, 'r')
dataset = np.asmatrix([map(float,line.split(' ')) for line in f ])
dataset = dataset.astype('float32')
########################################################################################
train_length = 150
validation_length = 2
test_length = 50
tstart =  int(sys.argv[3])
### Take the training set
ts_training = dataset[tstart:(tstart + train_length + validation_length),:]
scaler_ts_training = preprocessing.StandardScaler().fit(ts_training)
ts_training = preprocessing.scale(ts_training)
Noise = int(sys.argv[2])
if Noise == 1:
        ts_training = MakeNoise(ts_training, 0.2)
        #print 'The time series has been contaminated with observational noise'
###
num_species = ts_training.shape[1]
#### Give a different representation of the training set
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
####################################################################################
test_set = dataset[(tstart + train_length + validation_length):(tstart + train_length + validation_length + test_length), :]
test_set = scaler_ts_training.transform(test_set)
####################################################################################
#### Take last point of the training set and start predictions from there
ts_training_reshaped = ts_training_original.reshape((ts_training_original.shape[0], 1, num_species))
last_point_kept = np.array(ts_training_reshaped[(np.shape(ts_training_reshaped)[0] - 1), 0, :])
#####################################################################################


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(32, input_shape=(look_back,  num_species)))
### Decide whether to use sparsity or not
model.add(Dense(num_species, activation = 'linear', activity_regularizer=regularizers.l2(10e-5)))
#model.add(Dense(num_species, activation = 'linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
sys.stdout = open(os.devnull, "w")
model.fit(trainX, trainY, epochs=300, validation_data = (ValX, ValY), verbose=0)
sys.stdout = sys.__stdout__

# make predictions point by point starting from the last point of the training set
length_predictions = test_length
realizations = 30
next_point = np.zeros((length_predictions,num_species))
for prd in range(realizations):
	##### Last point of the training set for predictions
	last_point  = last_point_kept
	last_point = last_point.reshape((1, 1, num_species))
	##
	last_point = model.predict(last_point)
	next_point[0,:] = next_point[0,:] + last_point
	### Now last_point = next_point
	last_point = last_point.reshape((1, 1, num_species))
	##
	for i in range(1,length_predictions):
		last_point = model.predict(last_point)
		next_point[i,:] = next_point[i,:] + last_point
		last_point = last_point.reshape((1, 1, num_species))
next_point = next_point/realizations
next_point = np.delete(next_point, (0), 0)
training_data = model.predict(ts_training_reshaped)
training_data = np.insert(training_data, 0, np.array(np.repeat('nan',num_species)), 0)

### Naive forecast
naive = np.tile(last_point_kept, (length_predictions-1,1))
os_naive = np.sqrt(np.mean((naive - test_set[1:(length_predictions),:])**2))

os_rmse = np.sqrt(np.mean((next_point - test_set[1:(length_predictions),:])**2))
os_correlation = np.mean([pearsonr(next_point[:,i], test_set[1:(length_predictions), i])[0] for i in range(num_species)])
print 'RMSE of naive forecast = ', os_naive
print 'RMSE of LSTM  forecast = ', os_rmse


########################################################################################################
plot = True
if plot == True:
        all_data = np.concatenate((ts_training_original,test_set[0:(length_predictions),:]), axis = 0)
        all_data_reconstructed = np.concatenate((training_data,next_point), axis = 0)
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

	ax1.plot(all_data_reconstructed[:,0], lw = 2, color = 'r', label = 'Forecast')
	ax1.plot(all_data[:,0], color = 'b')
	ax1.axvline(x = (train_length + validation_length), lw = 2, ls = '--')
	ax1.legend()

	ax2.plot(all_data_reconstructed[:,1], lw = 2, color = 'r', label = 'Forecast')
	ax2.plot(all_data[:,1], color = 'b')
	ax2.axvline(x = (train_length + validation_length), lw = 2, ls = '--')

	ax3.plot(all_data_reconstructed[:,2], lw = 2,color = 'r', label = 'Forecast')
	ax3.plot(all_data[:,2], color = 'b')
	ax3.axvline(x = (train_length + validation_length), lw = 2, ls = '--')

	ax4.plot(all_data_reconstructed[:,3], lw = 2, color = 'r', label = 'Forecast')
	ax4.plot(all_data[:,3], color = 'b')
	ax4.axvline(x = (train_length + validation_length), lw = 2, ls = '--')
	plt.show()
