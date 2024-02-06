#!/usr/bin/env python
# coding: utf-8

# ## **Importing Libraries**

# In[40]:


import numpy as np
import pandas as pd
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical


# ## **LSTM Parameters**

# In[41]:


# parameters
la = "RecommendHiring"
l_sec = 5
nKineme, seqLen, nClass = 16, l_sec-1, 1 #num of kineme, seqlength for LSTM to be consider and num of output classes


# ## **Making LSTM Models**

# In[42]:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
Model = Sequential()

Model.add(LSTM(20,activation="tanh",dropout=0.2,recurrent_dropout=0.0,input_shape=(seqLen, 17)))
Model.add(Dense(units = nClass, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.01)

Model.compile(optimizer = opt, loss = 'mean_absolute_error')
Model.summary()

# In[43]:


file_path = "path for complete data matrix....."  #We have a created an npy array for AU+Kineme Data that need to be passed here as Data_60_Overall.npy
final_features = np.load(file_path)
file_to_work = final_features[:,seqLen:] #We have taken only AU values here. For eg, 60 sec data will have 59 onwards column as AU
file_to_work.shape

labe_path = "Path for labels......."
y_data = np.load(labe_path)
y_data = np.around(y_data,3)  #Rounding of floating upto 3 digit

#final_features.shape


# In[45]:

#Lists to have PCC and MAE
train_mae=[]
test_mae =[]
train_PCC =[]
test_PCC = []
train_acc =[]
test_acc = []

n=1
randnums= np.random.randint(1,file_to_work.shape[0],138)   #To make train and test fold
random_state = 42
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
for train_idx, test_idx in rkf.split(randnums):
    train_features, test_features, train_labels, test_labels = file_to_work[train_idx], file_to_work[test_idx], y_data[train_idx], y_data[test_idx] 
    train_aus = train_features.reshape((train_features.shape[0], seqLen, 17))
    test_aus = test_features.reshape((test_features.shape[0], seqLen, 17))
    
    print(train_aus.shape, test_aus.shape, train_labels.shape, test_labels.shape)

    zero_bias_history = Model.fit(train_aus, train_labels, epochs = 30, batch_size = 32, validation_split=0.1,callbacks=[callback])  #Fitting the model 
    
    #train predictions
    y_pred_train = Model.predict(train_aus)
    y_pred_train = np.around(y_pred_train,3)
    
    y_pred_test = Model.predict(test_aus)
    y_pred_test = np.around(y_pred_test,3)
    
    
    train_mae.append(mean_absolute_error(train_labels, y_pred_train)) ##mean squarred train error
    test_mae.append(mean_absolute_error(test_labels, y_pred_test)) #mean squarred test error
    
    train_acc.append(1-mean_absolute_error(train_labels, y_pred_train))
    test_acc.append(1-mean_absolute_error(test_labels, y_pred_test))

    y_train = train_labels.reshape(-1,1)
    b = np.corrcoef(y_train.T,y_pred_train.T)
    train_PCC.append(b[0][1])
    y = test_labels.reshape(-1,1)
    a = np.corrcoef(y.T,y_pred_test.T)
    test_PCC.append(a[0][1])
    print(n)
    n = n+1

    

     


# In[46]:

#Printing of all matices
print("Chunk-Level")
print("For 20 neurons with dropout 0.2")
print("For time {0} and label {1}".format(l_sec,la))

print("Train-mean square error is:{0}±{1}".format(round(np.array(train_mae).mean(),3),round(np.array(train_mae).std(),2)))
print("Test-mean square error is:{0}±{1}".format(round(np.array(test_mae).mean(),3),round(np.array(test_mae).std(),2)))

print("Train-accuracy is:{0}±{1}".format(round(np.array(train_acc).mean(),3),round(np.array(train_acc).std(),2)))
print("Test-accuracy is:{0}±{1}".format(round(np.array(test_acc).mean(),3),round(np.array(test_acc).std(),2)))
print("PCC for training data is:{0}±{1}".format(round(np.array(train_PCC).mean(),3),round(np.array(train_PCC).std(),2)))
print("PCC for testing data is:{0}±{1}".format(round(np.array(test_PCC).mean(),3),round(np.array(test_PCC).std(),2)))


# In[ ]:




