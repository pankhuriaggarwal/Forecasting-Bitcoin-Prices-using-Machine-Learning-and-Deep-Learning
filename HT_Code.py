#!/usr/bin/env python
# coding: utf-8

# # Loading packages and libraries

# In[209]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math

from sklearn.metrics import mean_squared_error
import sklearn

from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten

from statsmodels.tsa.arima_model import ARIMA
import statsmodels
import statistics as st


# # Exploratory Data Analysis

# In[210]:


# loading the data as a pandas dataframe and using the date column as the index
df = pd.read_csv("HT_DATA.csv", index_col=0)
#converting the date index column into Pandas Date format
df.index = pd.to_datetime(df.index)


# In[211]:


#checks
print(df.head())


# In[212]:


#checks
print(df.describe())


# In[213]:


# check for missing values
print(df.isnull().sum())


# In[214]:


# checking the data types of each variable in the dataset
print(df.dtypes)


# ## Creating X and Y dataframes 

# In[215]:


# printing all column names
print(df.columns.values)


# In[218]:


# extracting the relevant columns for X (inputs) and Y(output) and converting them to numpy arrays
X = df.drop(columns=['BTC']).values
Y = df[['BTC']].values


# In[217]:


#checks
print(rf" Shape of X: {X.shape}")
print(rf" Shape of Y: {Y.shape}")


# ## Standardizing the data

# In[219]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)


# ## Train-Test Split

# In[220]:


#train and test mode. using train size:test size :: 7:3
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=6)


# In[221]:


#checks
print(rf" Shape of X_train: {X_train.shape}")
print(rf" Shape of Y_train: {Y_train.shape}")
print(rf" Shape of X_test:  {X_test.shape}")
print(rf" Shape of Y_test:  {Y_test.shape}")


# # Random Walk / Naive Forecasting

# In[222]:


# initialize an array of zeroes for the predicted value
Y_predict_RandomWalk = np.zeros(len(Y_test))

# Standard Deviation of Y_train calculated and used as Standard Deviation of White Noise
noise_sigma = np.std(Y_train)

#setting seed for randomisation
np.random.seed(6)

# Generating predictions using the random walk
for k in range(1,len(Y_test)):
    # generating white noise with mean=0 and SD=noise_sigma
    noise = np.random.normal(0,noise_sigma)
    # Predicted value at time k = white noise + value at time k-1 
    Y_predict_RandomWalk[k] = Y_test[k-1] + noise


# In[223]:


# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
# not including the value of Y at t=0 as no predicted value exists at that time
plt.plot(Y_test[1:],"b-", label ="Actual Value of Y")
plt.plot(Y_predict_RandomWalk[1:],"g*", label="Predicted Value of Y") 
plt.title("Predicition using Random Walk/Naive Forecasting")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
# not including the value of Y at t=0 as no predicted value exists at that time
rmse = np.sqrt(mean_squared_error(Y_test[1:], Y_predict_RandomWalk[1:]))
print(rf"The RMSE is {rmse:2.2f}")


# # ARIMA

# data = Y.flatten()
# len(data)
# train = len(y_train.flatten())
# test = len(y_test.flatten())

# ARIMA_model = ARIMA(endog = y_train.flatten(), order=(5,0,1)).fit()
# forecast_length = len(y_test.flatten())
# ARIMA_forecast = ARIMA_model.forecast(forecast_length)[1]
# summary(ARIMA_model)
# #RMSE = np.sqrt(st.mean((y_test.flatten() - ARIMA_forecast)^2))

# plt.plot(y_test.flatten(), "b-")
# plt.plot(ARIMA_forecast, "g*")

# # MACHINE LEARNING

# ## Support Vector Regressor

# In[224]:


from sklearn.svm import SVR

# loading the model as SVR_Model (acronym for Support Vector Regressor)
SVR_Model = SVR(kernel="linear")

#training the model
SVR_Model.fit(X_train, Y_train.ravel())

# Use the model to make predicitions about unseen data
Y_predict_SVR = SVR_Model.predict(X_test)


# In[225]:


# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_SVR,"g*", label="Predicted Value of Y") 
plt.title("Predicition using Support Vector Regressor (Kernel = Linear)")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict_SVR))
print(rf"The RMSE is {rmse:2.2f}")


# ## Random Forest Regressor

# In[191]:


# importing for Random Forest Regressor model
from sklearn import ensemble

# loading the model as RFR (acronym for Random Forest Regressor)
# n is the number of estimates that will be generated - set to 1,000 for better results.
RFR = ensemble.RandomForestRegressor(n_estimators=1000)

#training the model
RFR.fit(X_train, Y_train.ravel())

# Use the model to make predicitions about unseen data# Use the model to make predicitions about unseen data
Y_predict_RFR = RFR.predict(X_test)


# In[226]:


# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_RFR,"g*", label="Predicted Value of Y") 
plt.title("Predicition using Random Forest Regresspr (n=1000)")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict_RFR))
print(rf"The RMSE is {rmse:2.2f}")


# # Deep Learning

# ## Data Preparation

# In[227]:


# converting all arrays to dataframes for Deep Learning Models (using Keras)
X_train_DL = pd.DataFrame(X_train)
Y_train_DL = pd.DataFrame(Y_train)
X_test_DL = pd.DataFrame(X_test)
Y_test_DL = pd.DataFrame(Y_test)


# In[228]:


# checks 
print(rf" Shape of X_train_DL: {X_train_DL.shape}")
print(rf" Shape of Y_train_DL: {Y_train_DL.shape}")
print(rf" Shape of X_test_DL:  {X_test_DL.shape}")
print(rf" Shape of Y_test_DL:  {Y_test_DL.shape}")


# ## DMLP

# In[230]:


# importing Keras' model
from keras.models import Sequential
from keras.layers import Dense

# loading the model as DMLP (acronym for Deep Multi Layer Perceptron)
DMLP = Sequential()

#get number of columns in training data
n_cols = X_train_DL.shape[1]

#add model layers
DMLP.add(Dense(200, activation='relu', input_shape=(n_cols,)))
DMLP.add(Dense(200, activation='relu'))
DMLP.add(Dense(200, activation='relu'))
DMLP.add(Dense(200, activation='tanh'))
DMLP.add(Dense(1))

#compile model using mse as a measure of model performance
DMLP.compile(optimizer='adam', loss='mean_squared_error')

print(DMLP.summary())

from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5)

#training model
DMLP.fit(X_train_DL, Y_train_DL, validation_data = (X_test_DL, Y_test_DL), epochs=50, callbacks=[early_stopping_monitor])

# Use the model to make predicitions about unseen data
Y_predict_DMLP = DMLP.predict(X_test_DL)


# In[231]:


print(Y_predict_DMLP)

# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test_DL,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_DMLP,"g*", label="Predicted Value of Y") 
plt.title("Predicition using MLP")
plt.grid(True)
plt.legend()
plt.show()

# calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test_DL, Y_predict_DMLP))
print(rf"The RMSE is {rmse:2.2f}")


# ## RNN

# In[ ]:


X_train_LSTM = X_train.reshape((X_train.shape[0],1, X_train.shape[1]))
X_test_LSTM = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# In[153]:


# checks 
print(rf" Shape of X_train_LSTM: {X_train_LSTM.shape}")
print(rf" Shape of X_test_LSTM:  {X_test_LSTM.shape}")


# In[167]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import GRU


RNN_model = Sequential()
RNN_model.add(GRU(50,input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2]), return_sequences=True))
RNN_model.add(SimpleRNN(50))
RNN_model.add(Dense(100))
RNN_model.compile(loss='mean_squared_error', optimizer='adam')

from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5)

RNN_model.fit(X_train_LSTM, Y_train, validation_data=(X_test_LSTM, Y_test), 
               epochs=5000,batch_size=50, callbacks=[early_stopping_monitor] )


# Use the model to make predicitions about unseen data
Y_predict_RNN = RNN_model.predict(X_test_LSTM)


# In[ ]:


# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test_DL,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_RNN,"g*", label="Predicted Value of Y") 
plt.title("Predicition using RNN")
plt.grid(True)
plt.legend()
plt.show()

#calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict_RNN))
print(rf"The RMSE is {rmse:2.2f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### LSTM

# In[155]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

LSTM_model = Sequential()
LSTM_model.add(LSTM(100,input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
LSTM_model.add(Dense(100))
LSTM_model.compile(loss='mean_squared_error', optimizer='adam')

from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5)

LSTM_model.fit(X_train_LSTM, Y_train, validation_data=(X_test_LSTM, Y_test), 
               epochs=1000,batch_size=100, callbacks=[early_stopping_monitor] )


# Use the model to make predicitions about unseen data
Y_predict_LSTM = LSTM_model.predict(X_test_LSTM)


# In[173]:


# Generating a plot for actual and predicted values
plt.figure(figsize=[10,4])
plt.plot(Y_test_DL,"b-", label ="Actual Value of Y")
plt.plot(Y_predict_LSTM,"g*", label="Predicted Value of Y") 
plt.title("Predicition using LSTM")
plt.grid(True)
plt.legend()
plt.show()

#calculating and printing root mean squared error
rmse = np.sqrt(mean_squared_error(Y_test, Y_predict_LSTM))
print(rf"The RMSE is {rmse:2.2f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print('Train set shape', X_train_series.shape)
print('Test set shape', X_test_series.shape)


# In[28]:


model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(1))
model_cnn.compile(loss='mse', optimizer=adam)
model_cnn.summary()


# In[ ]:


cnn_history = model_cnn.fit(X_train_series, y_test, validation_data=(X_test_series, y_test), epochs=epochs, verbose=2)


# In[ ]:


model = Sequential()
model.add(LSTM(50, input_dim=(X_train.shape[1])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')


# In[ ]:


history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
 
# make a prediction
yhat = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = mean_squared_error(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)


# In[ ]:





# In[ ]:





# # EXTRA STUFF

# ## MLR

# In[ ]:


# Import the machine learning library Scikit-Learn
from sklearn import linear_model
import math

regr = linear_model.LinearRegression()

# Fit the model to the training data
regr.fit(X_train, y_train)

prediction = regr.predict(X_test)
print(prediction[0:5])
print(y_test[0:5])

# Compare the values to get the score  # to find the accuracy of the prediction - best value is 1 and worst is 0. 
# usually for ML - 0.9 and above is amazing and above 0.5 is acceptable
score = regr.score(X_test, y_test)
print("The accuracy of the prediction is", score)

#prediction = np.array(prediction)
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test, prediction))
print(rmse)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_test, prediction)


# ## LASSO

# In[ ]:


from sklearn import linear_model
import math

clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train.ravel())
#print(clf.coef_)
#print(clf.intercept_)

prediction = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(prediction[0:5])
print(y_test[0:5])
print("The accuracy of the prediction is", score)

prediction = np.array(prediction)

rmse = np.sqrt(mean_squared_error(y_test, prediction))
print(rmse)
#from sklearn.model_selection import cross_val_score
#cross_val_score(clf, X, Y.ravel(), cv=5, scoring='recall_macro')

