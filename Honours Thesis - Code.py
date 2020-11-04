#!/usr/bin/env python
# coding: utf-8

# # Loading packages and libraries

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd
from numpy.random import seed
import numpy as np
import math

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import sklearn

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, LSTM, SimpleRNN, Dropout
from keras.callbacks import EarlyStopping
from tensorflow import keras
import tensorflow

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels
import statistics as st
import time


# # Exploratory Data Analysis

# In[22]:


# loading the data as a pandas dataframe and using the date column as the index
df = pd.read_csv("HT_DATA_Log_Transformed.csv", index_col=0)

#converting the date index column into Pandas Date format
df.index = pd.to_datetime(df.index)


# In[23]:


#checks
print(df.tail())
print(df.describe())

# check for missing values
print(df.isnull().sum())

# checking the data types of each variable in the dataset
print(df.dtypes)


# ## Creating X and Y dataframes 

# In[24]:


# extracting the relevant columns for X (inputs) and Y(output) and converting them to numpy arrays
X = df.drop(columns=['BTC', 'BTC_Yest']).values
Y = df[['BTC']].values


# In[25]:


#checks
print(rf" Shape of X: {X.shape}")
print(rf" Shape of Y: {Y.shape}")


# ## Standardizing the data

# In[26]:


from sklearn import preprocessing
X = preprocessing.scale(X)
# standardization to mean=0, variance=1


# ## Train-Test Split

# In[27]:


# finding n to separate the data into the first 70% and next 30%
n_train = round(2159/10*7)
n_test = round(2159/10*3)
print(rf'The length of the training set is {n_train}')
print(rf'The length of the testing set is {n_test}')


# In[28]:


# separating the data using train and test
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


# In[29]:


#checks
print(rf" Shape of X_train: {X_train.shape}")
print(rf" Shape of Y_train: {Y_train.shape}")
print(rf" Shape of X_test:  {X_test.shape}")
print(rf" Shape of Y_test:  {Y_test.shape}")


# # Feature Selection

# In[92]:


from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest

#fixing the random state to generate reproducible results
seed(6)


# define feature selection
fs = SelectKBest(score_func=mutual_info_regression, k='all')

# apply feature selection
X_selected = fs.fit_transform(X_train, Y_train.ravel())

print(X_selected.shape)

# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %s: %f' % (df.drop(columns=['BTC','BTC_Yest']).columns.values[i], fs.scores_[i]))
#plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# In[93]:


#chosing the featues with MIR>0
chosenfeatures = sum(fs.scores_>0)
print(rf'The number of features having a score>=0 are {chosenfeatures}')

# creating a new model with the chosen features
fs_chosen = SelectKBest(score_func=mutual_info_regression, k=chosenfeatures )

# learn relationship from training data
fs_chosen.fit(X_train, Y_train.ravel())

# transform train input data
X_train_fs = fs_chosen.transform(X_train)
# transform test input data
X_test_fs = fs_chosen.transform(X_test)


#checks
print(rf'The shape of training input data after feature selection {X_train_fs.shape}')
print(rf'The shape of testing input data after feature selection {X_test_fs.shape}')


# # Random Walk / Naive Forecasting

# In[94]:


# Generating predictions using the random walk
Y_predict_RandomWalk = Y_test[:-1]


# In[95]:


# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
# not including the value of Y at t=0 as no predicted value exists at that time
plt.plot(Y_test[1:],"b-", label ="Actual Value of Y")
plt.plot(Y_predict_RandomWalk,"g*", label="Predicted Value of Y") 
plt.title("Predicition using Random Walk/Naive Forecasting")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test[1:], Y_predict_RandomWalk))*100
print(rf"The RMSE is {rmse:2.4f}%")


# # ARIMA

# ### Identifying hyperparameters

# In[96]:


# Plotting the PACF and ACF to identify AR and MA lags (as per Box-Jenkins identification)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(Y_train, lags=100)
plot_pacf(Y_train,lags=100)
plt.show()

# For MA(q), only the first q autocorrelations are nonzero, so the ACF should cut off after lag q.
# For AR(p), the autocorrelations may decline gradually, but the PACF should cut off after lag p


# ### Building and training the model

# In[97]:


#marking start time for model training
start_time = time.time()

#training the ARIMA model
ARIMA_model = ARIMA(endog = Y_train, order=(0,0,0)).fit()

#calculating time taken to train
print("--- %s seconds ---" % (time.time() - start_time))

#printing model summary statistics 
print(ARIMA_model.summary())


# ### Testing the model using validation data

# In[98]:


#predicting using test data
forecast_length = len(Y_test)
ARIMA_forecast = ARIMA_model.forecast(forecast_length)

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test,"b-", label ="Actual Value of Y")
plt.plot(ARIMA_forecast[0],"g-", label="Predicted Value of Y") 
plt.title("Predicition using ARIMA model")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test, ARIMA_forecast[0]))*100
print(rf"The RMSE is {rmse:2.4f}%")


# # ARMAX

# ### Building and Training the model

# In[90]:


#marking start time for model training
start_time = time.time()

# fit model
# seasonal effects and Integrating is set to zero to convert SARIMAX into ARMAX
ARMAX = SARIMAX(Y_train, exog=X_train_fs, order=(0, 0, 0)).fit(disp=False)

#printing time taken to train the model
print("--- %s seconds ---" % (time.time() - start_time))

#print model summary stats
print(ARMAX.summary())

# make prediction
Y_predict_ARMAX= ARMAX.predict(1,len(Y_test),exog=X_test_fs)


# ### Testing the model using validation data

# In[91]:


# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_ARMAX,"g-", label="Predicted Value of Y") 
plt.title("Predicition using ARIMAX model")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict_ARMAX))*100
print(rf"The RMSE is {rmse:2.4f}%")


# # MACHINE LEARNING

# ## Support Vector Regressor

# ### Importing SVR model and setting seed for randomization to generate reproducible results

# In[78]:


from sklearn.svm import SVR
seed(6)


# ### Building the model and tuning hyperparameters

# In[60]:


# loading the model as SVR_Model with kernel = rbf
SVR_Model = SVR(kernel="rbf")

# fixing a range for evaluation of each parameter
c1= np.linspace(-0.0001,1000,5000)
gamma1=np.linspace(-0.0001,1000,5000)
epsilon1=np.linspace(-0.0001,1000,5000)

#creating a parameter grid
param_grid=dict(C=c1, gamma= gamma1, epsilon=epsilon1)

# conducting a random search and fitting the data to the model
randomsearch = RandomizedSearchCV(SVR_Model, param_grid, cv=5, scoring ='neg_mean_squared_error', random_state=6)
randomsearch.fit(X_train_fs, Y_train.ravel())  


# ### Identifying chosen hyperparameters

# In[61]:


print(rf'The chosen hyperparameters are {randomsearch.best_estimator_}')


# ### Building and training the model with chosen hyperparameters

# In[62]:


# marking start time for training 
start_time = time.time()

#loading the model with the chosen hyperparameters
SVR_Model_chosen = SVR(kernel="rbf", epsilon = 485.8971280256051, C = 371.8743120624125, gamma=76.01511064212842)

#training the model
SVR_Model_chosen.fit(X_train_fs, Y_train.ravel())

#calculating the time taken to train
print("--- %s seconds ---" % (time.time() - start_time))

# Training RMSE - calculate and print
Y_train_SVR = SVR_Model_chosen.predict(X_train_fs)
rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_SVR)))*100
print(rf"The RMSE is {rmse:2.4f}%")


# ### Testing the model using Validation Data

# In[102]:


# Use the model to make predicitions about test data
Y_predict_SVR = SVR_Model_chosen.predict(X_test_fs)

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_SVR,"g*", label="Predicted Value of Y") 
plt.title("Predicition using Support Vector Regressor")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = (np.sqrt(mean_squared_error(Y_test, Y_predict_SVR)))*100
print(rf"The RMSE is {rmse:2.4f}%")


# ## Random Forest Regressor

# ### Importing Random Forest model and setting seed for randomization to generate reproducible results

# In[66]:


from sklearn import ensemble
seed(7)


# ### Building the model and tuning hyperparameters

# In[67]:


# loading the model as RFR (acronym for Random Forest Regressor)
RFR = ensemble.RandomForestRegressor()


# fixing a range for evaluation of each parameter
trees_range = np.arange(500,3000,250) 
depth_range = np.arange(0,21,1)
features_range = np.arange(0,6,1)
min_samples_leaf_range = np.arange(0,20,1)
min_samples_split_range = np.arange(0,50,2)


#creating a parameter grid
param_grid= dict(max_depth=depth_range,n_estimators=trees_range, max_features=features_range, min_samples_leaf=min_samples_leaf_range, min_samples_split= min_samples_split_range)


# conducting a random search and fitting the data to the model
randomsearch_RFR = RandomizedSearchCV(RFR, param_grid, cv=5, scoring ='neg_mean_squared_error', random_state=6, return_train_score=True)
randomsearch_RFR.fit(X_train_fs, Y_train.ravel())  


# ### Identifying the chosen hyperparameters

# In[ ]:


# printing the chosen hyperparameters
print(rf'The chosen hyperparameters are {randomsearch_RFR.best_estimator_}')


# ### Builiding and Training the model with chosen hyperparameters

# In[75]:


# marking the start time for model training
start_time = time.time()

#loading the RFR model with the chosen hyperparameters
RFR_chosen = ensemble.RandomForestRegressor(n_estimators=2500, max_depth=6, max_features=2, min_samples_leaf=10, min_samples_split=38, random_state=6)

#training the model
RFR_chosen.fit(X_train_fs, Y_train.ravel())

#calculating and printing training time
print("--- %s seconds ---" % (time.time() - start_time))

# Training RMSE
Y_train_RFR = RFR_chosen.predict(X_train_fs)
rmse = (np.sqrt(mean_squared_error(Y_train, Y_train_RFR)))*100
print(rf"The Training RMSE is {rmse:2.4f}%")


# ### Testing the model using Validation Data

# In[101]:


# Use the model to make predicitions about testing data
Y_predict_RFR = RFR.predict(X_test_fs)

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_RFR,"g-", label="Predicted Value of Y") 
plt.title("Predicition using Random Forest Regressor")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict_RFR))*100
print(rf"The RMSE is {rmse:2.4f}%")


# # Deep Learning

# ### Data Prep for DL

# In[103]:


# changing the data into 3d arrays
from numpy import hstack
from numpy import array
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

DL_trainingdata= hstack((X_train_fs,Y_train))
DL_testingdata= hstack((X_test_fs,Y_test))


# ### Setting seeds to generate reproducible results

# In[104]:


seed(6)
tensorflow.random.set_seed(6)


# ## DMLP

# ### Data Prep for MLP

# In[105]:


# timestep is 1 in this model

X_train_DL, Y_train_DL = split_sequences(DL_trainingdata, 1)
X_test_DL, Y_test_DL = split_sequences(DL_testingdata, 1)

n_input = X_train_DL.shape[1] * X_train_DL.shape[2]

#reshaping into 2D array as per MLP requirements
X_train_MLP = X_train_DL.reshape((X_train_DL.shape[0], n_input))
X_test_MLP = X_test_DL.reshape((X_test_DL.shape[0], n_input))


print(X_train_MLP.shape, Y_train_DL.shape)
print(X_test_MLP.shape, Y_test_DL.shape)


# ### Building the DMLP Model

# In[106]:


def DMLP_builder(activation1='relu', activation2='relu', dropoutrate=0.2,opt='adam',nn=15 ):
    DMLP = Sequential()
    #add model layers
    DMLP.add(Dense(nn, activation= activation1, input_dim=n_input))
    DMLP.add(Dropout(dropoutrate))
    DMLP.add(Dense(nn,activation=activation2))
    DMLP.add(Dropout(dropoutrate))
    DMLP.add(Dense(1))
    #compile model
    DMLP.compile(optimizer=opt, loss='mean_squared_error')
    return DMLP


# ### Using Random Search for hypertuning

# In[380]:


#set early stopping monitor so the model stops training when loss wont reduce more
early_stopping_monitor = EarlyStopping(patience=5,  monitor='loss')

#setting a range for each hyperparameter
activation_range = ['relu','sigmoid','tanh']
dropout_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
opt_range = ['Adagrad', 'Adam', 'RMSProp']
epoch_range = [50,100,150,200,250,300,350,400,450,500]
batch_range = [8,16,32,64,128,256]
nn_range = [5,10,11,15,20,25]

# setting up a grid for hyperparameter values
param_grid1 = dict(activation1 = activation_range, activation2 = activation_range, dropoutrate= dropout_range, opt=opt_range, epochs=epoch_range, batch_size=batch_range, nn=nn_range)

# running a random search and fitting the data to the model
model_DMLP = KerasRegressor(build_fn = DMLP_builder)
randomsearch_DMLP = RandomizedSearchCV(model_DMLP, param_grid1, cv=5, random_state=6, verbose=0)
randomsearch_DMLP.fit(X_train_MLP, Y_train_DL,callbacks=early_stopping_monitor,verbose=0)  


# ### Identifying the best hyperparameters

# In[107]:


print(randomsearch_DMLP.best_params_)


# ### Building and training the model with chosen hyperparameters

# In[108]:


seed(8)
tensorflow.random.set_seed(8)
from keras.callbacks import EarlyStopping

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5)

#setting the start time for model training
start_time = time.time()

#builidng the model with the chosen hyperparameters
DMLP_chosen = DMLP_builder(activation1='tanh', activation2='relu', dropoutrate=0.5, opt='RMSProp', nn=25)
print(DMLP_chosen.summary())

#training model
DMLP_chosen.fit(X_train_MLP, Y_train_DL, validation_data = (X_test_MLP, Y_test_DL), epochs=250, batch_size=16, callbacks=[early_stopping_monitor], verbose=1)

#calculating and printing training time
print("--- %s seconds ---" % (time.time() - start_time))

# Training RMSE
Y_train_DMLP = DMLP_chosen.predict(X_train_MLP)
rmse = (np.sqrt(mean_squared_error(Y_train_DL, Y_train_DMLP)))*100
print(rf"The training RMSE is {rmse:2.4f}%")


# ### Testing the model using Validation Data

# In[109]:


# Use the model to make predicitions about unseen data
Y_predict_DMLP = DMLP_chosen.predict(X_test_MLP)

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test_DL,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_DMLP,"g*", label="Predicted Value of Y") 
plt.title("Predicition using MLP")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test_DL, Y_predict_DMLP))*100
print(rf"The RMSE is {rmse:2.4f}%")


# ## RNN

# ### Data prep for RNN

# In[110]:


#timestep = 30 (chosen bymanual search)

X_train_DL, Y_train_DL = split_sequences(DL_trainingdata, 30)
X_test_DL, Y_test_DL = split_sequences(DL_testingdata, 30)

print(X_train_DL.shape, Y_train_DL.shape)
print(X_test_DL.shape, Y_test_DL.shape)


# ### Building the RNN model

# In[112]:


def RNN_builder(activation1='relu', activation2='relu', dropoutrate=0.2, opt='adam', nn=100 ):
    RNN_model = Sequential()
     #add model layers
    RNN_model.add(SimpleRNN(nn, activation=activation1,
                        input_shape=(X_train_DL.shape[1], X_train_DL.shape[2]),return_sequences=True))
    RNN_model.add(Dropout(dropoutrate))
    RNN_model.add(SimpleRNN(nn,activation=activation2))
    RNN_model.add(Dropout(dropoutrate))
    RNN_model.add(Dense(1)) # should be 1
    RNN_model.compile(loss='mean_squared_error', optimizer=opt)
    return RNN_model


# ### RandomSearch for hyperparameter tuning

# In[526]:


#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5,  monitor='loss')

#setting a range for each hyperparameter
activation_range = ['relu','sigmoid','tanh']
dropout_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
opt_range = ['Adagrad', 'Adam', 'RMSProp']
epoch_range = [50,100,150,200,250,300,350,400,450,500]
batch_range = [8,16,32,64,128,256]
nn_range = [10,20,30,40,50,60,70,80,90,100]


# setting up a grid for all hyperparameter values
param_grid1 = dict(activation1 = activation_range, activation2 = activation_range, dropoutrate= dropout_range, opt=opt_range, epochs=epoch_range, batch_size=batch_range, nn=nn_range)


#conducting random search and fitting the data
model_RNN = KerasRegressor(build_fn = RNN_builder)
randomsearch_RNN = RandomizedSearchCV(model_RNN, param_grid1, cv=5, random_state=6,verbose=0)
randomsearch_RNN.fit(X_train_DL, Y_train_DL,callbacks=early_stopping_monitor, verbose=0)  


# ### Identifying the best hyperparameters

# In[ ]:


print(rf'The chosen hyperparameters are {randomsearch_RNN.best_params_}')


# ### Building and training the model with the chosen hyperparameters

# In[114]:


#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5,  monitor='loss')

# setting 
start_time = time.time()

RNN_chosen = RNN_builder(activation1='sigmoid', activation2='tanh', dropoutrate=0.8, opt='RMSProp', nn=10 )
print(RNN_chosen.summary())

#training model
RNN_chosen.fit(X_train_DL, Y_train_DL,epochs=100, batch_size=128, callbacks=[early_stopping_monitor], verbose=1)

print("--- %s seconds ---" % (time.time() - start_time))

#Training RMSE
Y_predict_train_RNN = RNN_chosen.predict(X_train_DL)
rmse = (np.sqrt(mean_squared_error(Y_train_DL.flatten(), Y_predict_train_RNN.flatten())))*100
print(rf"The training RMSE is {rmse:2.4f}%")


# ### Testing the model with validation data

# In[115]:


# Use the model to make predicitions about unseen data
Y_predict_RNN = RNN_chosen.predict(X_test_DL)

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test_DL.flatten(),"b-", label ="Actual Value of Y")
plt.plot(Y_predict_RNN.flatten(),"g*", label="Predicted Value of Y") 
plt.title("Predicition using RNN")
plt.grid(True)
plt.legend()
plt.show()

#calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test_DL.flatten(), Y_predict_RNN.flatten()))*100
print(rf"The RMSE is {rmse:2.4f}%")


# ### LSTM

# ### Data prep for LSTM

# In[116]:


# Timestep = 50 (manual search)
X_train_DL, Y_train_DL = split_sequences(DL_trainingdata, 50)
X_test_DL, Y_test_DL = split_sequences(DL_testingdata, 50)

print(X_train_DL.shape, Y_train_DL.shape)
print(X_test_DL.shape, Y_test_DL.shape)


# ### Building LSTM model

# In[117]:


def LSTM_builder(activation1='relu', activation2='relu', dropoutrate=0.4, opt='adam' ):
    LSTM_model = Sequential()
     #add model layers
    LSTM_model.add(LSTM(100, activation=activation1,
                        input_shape=(X_train_DL.shape[1], X_train_DL.shape[2]),return_sequences=True))
    LSTM_model.add(Dropout(dropoutrate))
    LSTM_model.add(LSTM(100,activation=activation2))
    LSTM_model.add(Dropout(dropoutrate))
    LSTM_model.add(Dense(1)) # should be 1
    LSTM_model.compile(loss='mean_squared_error', optimizer=opt)
    return LSTM_model


# ### Random Search for Hyperparameter Tuning

# In[580]:


#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5,  monitor='loss')

#setting a range for each hyperparameter
activation_range = ['relu','sigmoid','tanh']
dropout_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
opt_range = ['Adagrad', 'Adam', 'RMSProp']
epoch_range = [50,100,150,200,250,300,350,400,450,500]
batch_range = [8,16,32,64,128,256]

#setting up a grid for hyperparameters
param_grid1 = dict(activation1 = activation_range, activation2 = activation_range, 
                   dropoutrate= dropout_range, opt=opt_range, epochs=epoch_range, batch_size=batch_range)

#running random search and fitting model to data
model_LSTM = KerasRegressor(build_fn = LSTM_builder)
randomsearch_LSTM = RandomizedSearchCV(model_LSTM, param_grid1, cv=5, random_state=6, verbose=0)
randomsearch_LSTM.fit(X_train_DL, Y_train_DL,callbacks=early_stopping_monitor, verbose=0)  


# ### Identifying the best hyperparameters

# In[ ]:


print(rf'The chosen hyperparameters are {randomsearch_LSTM.best_params_}')


# ### Building and Training the model with the chosen hyperparameters

# In[120]:


#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5,  monitor='loss')


# marking the start time for model training
start_time = time.time()

#building the model with chosen hyperparameters
LSTM_chosen = LSTM_builder(activation1='tanh', activation2='relu', dropoutrate=0.7, opt='RMSProp' )
print(LSTM_chosen.summary())

#training model
LSTM_chosen.fit(X_train_DL, Y_train_DL,epochs=450, batch_size=16, callbacks=[early_stopping_monitor], verbose=1)

#calculating and printing the training time
print("--- %s seconds ---" % (time.time() - start_time))

#Training RMSE
Y_predict_train_LSTM = LSTM_chosen.predict(X_train_DL)
rmse = (np.sqrt(mean_squared_error(Y_train_DL.flatten(), Y_predict_train_LSTM.flatten())))*100
print(rf"The training RMSE is {rmse:2.4f}%")


# ### Testing the model with validation data

# In[121]:


# Use the model to make predicitions about unseen data
Y_predict_LSTM = LSTM_chosen.predict(X_test_DL)

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test_DL.flatten(),"b-", label ="Actual Value of Y")
plt.plot(Y_predict_LSTM.flatten(),"g*", label="Predicted Value of Y") 
plt.title("Predicition using LSTM")
plt.grid(True)
plt.legend()
plt.show()

#calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test_DL.flatten(), Y_predict_LSTM.flatten()))*100
print(rf"The RMSE is {rmse:2.4f}%")


# # Combination Forecast

# In[122]:


# Generating an average of the forecast for LSTM and RNN
Y_predict_combination = (Y_predict_LSTM + Y_predict_RNN[20:])/2

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test_DL.flatten(),"b-", label ="Actual Value of Y")
plt.plot(Y_predict_combination.flatten(),"g*", label="Predicted Value of Y") 
plt.title("Predicition using Combination")
plt.grid(True)
plt.legend()
plt.show()

#calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test_DL.flatten(), Y_predict_combination.flatten()))*100
print(rf"The RMSE is {rmse:2.4f}%")

