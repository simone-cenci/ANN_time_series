from keras.layers import Input, Dense, regularizers
from keras.models import Model
import numpy as np
from scipy import stats
from sklearn import preprocessing
import sys, os
import math,sys,os
import random as rnd
from keras.models import Sequential
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error



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
def CV(output, trX,trY, VlX, VlY, tst_st, last_known_point, lnt_prd, neurons, epoche, time_lag = 1, act_fun = 'linear'):
	### input:
	### outout = for parallel computing 
	### trX,trY,VlX,VlY,tst_st = trivial
	### last_known_point = last point in the training set
	### lnt_prd = length of the test set to predict
	### neurons,epoche = parameters to cross-validate
	### Example of useage:
	### a,b,c = CV(trainX,trainY,ValX,ValY,test_set,last_point_kept,50, 16,100)
	nm_sp = tst_st.shape[1]
	model = Sequential()
	model.add(LSTM(neurons, input_shape=(time_lag,  nm_sp)))
	model.add(Dense(nm_sp, activation = act_fun, activity_regularizer=regularizers.l2(10e-3)))
	model.compile(loss='mean_squared_error', optimizer='adam')
	sys.stdout = open(os.devnull, "w")
	model.fit(trX, trY, epochs=epoche, batch_size=1, validation_data = (VlX, VlY), verbose=3)
	sys.stdout = sys.__stdout__
	next_point = np.zeros((lnt_prd,nm_sp))

	##### Last point of the training set for predictions
	last_point  = last_known_point
	last_point = last_point.reshape((1, 1, nm_sp))
	##
	last_point = model.predict(last_point)
	next_point[0,:] = last_point
	### Now last_point = next_point
	last_point = last_point.reshape((1, 1, nm_sp))
	##
	for i in range(1,lnt_prd):
		last_point = model.predict(last_point)
		next_point[i,:] = last_point
		last_point = last_point.reshape((1, 1, nm_sp))
	next_point = np.delete(next_point, (0), 0)
	os_rmse = np.sqrt(np.mean((next_point - tst_st[0:(lnt_prd-1),:])**2))
	#return(os_rmse, neurons, epoche)
	output.put(os_rmse)


def MakeNormalNoise(X, percentage_std):
        ### Input: X = time series
        ###        percentage_std = percentage of standard deviation for the noise [0,1]
        X = X + np.random.normal(loc=0.0, scale=percentage_std*np.std(X[:,1]), size= X.shape)
	return(X)

##### AutoEncoder
def AE(ts_, train_length, validation_length, dim_red = 1, act_fun_enc = 'hard_sigmoid', act_fun_dec = 'linear', epoche = 2000, BatchSize = 128):
        ## Input:
        ## ts = training set (This has to be already normalized)
        ## train_length, validation_length = length of the training and validation set in the training set
        ## dim_red = number of dimensions of which you want to compress your input time series inthe encoder
        ## act_fun_* = activation functions for the encoder and decoder
        ## ------------
        ## Output:
        ## A new time series which should be a faithful reproduction of the original
        #########################################################################

        x_train = ts_[0:(train_length),:]
        x_validation = ts_[(train_length):(train_length + validation_length), :]
        #### Some properties of the autoencoder
        encoding_dim = np.shape(ts_)[1] - dim_red
        ## This is the size of the decoder (dimension of the state space)
        decoding_dim = np.shape(ts_)[1]
        ###########################################################################


        input_ts = Input(shape = (decoding_dim,))
        encoded = Dense(encoding_dim, activation= act_fun_enc, activity_regularizer=regularizers.l2(10e-3))(input_ts)
        decoded = Dense(decoding_dim, activation= act_fun_dec, activity_regularizer=regularizers.l2(10e-3))(encoded) 
        autoencoder = Model(input_ts, decoded)

        encoder = Model(input_ts, encoded)

        # create a placeholder for an encoded (d-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        # choose your loss function and otpimizer
        autoencoder.compile(loss='mean_squared_error', optimizer='adam')

        all_x_train = np.concatenate((x_train,x_validation), axis = 0)
        ########################
        #### Train the autoencoder but avoid writing on stdoutput
        sys.stdout = open(os.devnull, "w")
        autoencoder.fit(x_train, x_train,
                        epochs= epoche,
                        batch_size = BatchSize,
                        shuffle = True,
                        validation_data=(x_validation, x_validation))
	#########################
	encoded_ts = encoder.predict(all_x_train) ### h(x)    I think this is the econded representation
	decoded_ts = decoder.predict(encoded_ts) ###  g[h(x)]  and this is the one you are interested in

        sys.stdout = sys.__stdout__
        #### Standardize training and testing separately (the test set is already stdrzed)
        decoded_ts = preprocessing.scale(decoded_ts)
        return(decoded_ts)

def DAE(ts_, train_length, validation_length, dim_red = 1, act_fun_enc = 'hard_sigmoid', act_fun_dec = 'linear', epoche = 2000, BatchSize = 128):
        ## Input:
        ## ts = training set (This has to be already normalized)
        ## train_length, validation_length = length of the training and validation set in the training set
        ## dim_red = number of dimensions of which you want to compress your input time series inthe encoder
        ## act_fun_* = activation functions for the encoder and decoder
        ## ------------
        ## Output:
        ## A new time series which should be a faithful reproduction of the original
        #########################################################################

        x_train = ts_[0:(train_length),:]
        x_validation = ts_[(train_length):(train_length + validation_length), :]
	x_train_noisy = MakeNormalNoise(x_train, 0.2)
	x_valid_noisy = MakeNormalNoise(x_validation,0.2)
        #### Some properties of the autoencoder
        encoding_dim = np.shape(ts_)[1] - dim_red
        ## This is the size of the decoder (dimension of the state space)
        decoding_dim = np.shape(ts_)[1]
        ###########################################################################


        input_ts = Input(shape = (decoding_dim,))
        encoded = Dense(encoding_dim, activation= act_fun_enc, activity_regularizer=regularizers.l2(10e-3))(input_ts)
        decoded = Dense(decoding_dim, activation= act_fun_dec, activity_regularizer=regularizers.l2(10e-3))(encoded) 
        autoencoder = Model(input_ts, decoded)

        encoder = Model(input_ts, encoded)

        # create a placeholder for an encoded (d-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        # choose your loss function and otpimizer
        autoencoder.compile(loss='mean_squared_error', optimizer='adam')

        all_x_train = np.concatenate((x_train,x_validation), axis = 0)
        ########################
        #### Train the autoencoder but avoid writing on stdoutput
        sys.stdout = open(os.devnull, "w")
        autoencoder.fit(x_train_noisy, x_train,
                        epochs= epoche,
                        batch_size = BatchSize,
                        shuffle = True,
                        validation_data=(x_valid_noisy, x_validation))
	#########################
	encoded_ts = encoder.predict(all_x_train) ### h(x)    I think this is the econded representation
	decoded_ts = decoder.predict(encoded_ts) ###  g[h(x)]  and this is the one you are interested in

        sys.stdout = sys.__stdout__
        #### Standardize training and testing separately (the test set is already stdrzed)
        decoded_ts = preprocessing.scale(decoded_ts)
        return(decoded_ts)

def StackDAE(ts_, train_length, validation_length, num_stack = 10, dim_red = 1, act_fun_enc = 'hard_sigmoid', act_fun_dec = 'linear', epoche = 2000, BatchSize = 
128):
	denoised_ts = ts_
	for i in range(num_stack):
		denoised_ts = DAE(denoised_ts,train_length, validation_length, dim_red, act_fun_enc, act_fun_dec, epoche, BatchSize)
	return(denoised_ts)

def Standardizza(X,meanY,sdY):
	#### This is for standardize a dataset with mean and standard deviation of another
	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - meanY[i])/sdY[i]
	return(X)
