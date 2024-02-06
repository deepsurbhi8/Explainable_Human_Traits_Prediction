#!/usr/bin/env python
# coding: utf-8

# ## **Importing Libraries**

# In[11]:


import numpy as np
import pandas as pd
import glob
import math
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


# ## **Encoding of Input Data**

# In[12]:


# onehot encoding of kineme sequence
def onehot_encoding(ks, nKineme):
    #print(ks)
    onehot_encoded = list()
    for k in ks:
        #print(k)
        vec = [0 for _ in range(nKineme)]
        vec[k-1] = 1
        onehot_encoded.append(vec)
        #print("Vector")
        #print(vec)
    return onehot_encoded


def ks_encoding(ks, nKineme):
    # ks is a numpy ndarray
    m, n = ks.shape #m=92, n=29
    #print(m, n)
    ks = ks.tolist() #converted to list
    encoded_features = np.asarray(
        [np.asarray(onehot_encoding(ks[i], nKineme)) for i in range(m)]
    )
    return encoded_features


# 
# ## **LSTM Parameters**

# In[13]:



# parameters for LSTM models
nKineme, seqLen, nClass = 16,14, 1   #Num of kineme, seqLen means n-D vector to be passed and nClass: Num of output neuron
BATCH_SIZE = 32
nNeuron = 5

#For this we have created AU matrix for train, test and validation separately and here is the need to pass that data matrix and labels. Kineme matrix is in .npy format

X_train = np.load('')  #path for AU train matrix
y_train = np.load('') #path for train labels
y_train = y_train[:,1].astype(np.float)
y_train = np.around(y_train,3)

X_val = np.load('') #path for AU validation matrix
y_val = np.load('') #path for validation labels
y_val = y_val[:,1].astype(np.float)
y_val = np.around(y_val,3)

X_test = np.load('') #path for AU test matrix
y_test = np.load('') #path for test labels
y_test = y_test[:,1].astype(np.float)
y_test = np.around(y_test,3)



# In[14]:


#Printing of datashape {For Example: X_train will have 5990,238, 5990 are number of samples and 238 = 17*14 (17D vector for every 2 sec data with 50% overlap for 15 sec)}
print(X_train.shape, X_val.shape, X_test.shape)


# In[15]:


#LSTM Model Architecture
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

Model = Sequential()
Model.add(LSTM(32,activation="tanh",dropout=0.2,recurrent_dropout=0.0,input_shape=(seqLen, 17),return_sequences =False ))
Model.add(Dense(units = nClass, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.01)

Model.compile(optimizer = opt, loss = 'mean_absolute_error')
Model.summary()


# In[16]:


#Lists to contain MAE and PCC
train_mse=[]
test_mse =[]
train_PCC =[]
test_PCC = []
n=1

train_features,train_labels = X_train,y_train
test_features, test_labels = X_test, y_test
val_features, val_labels = X_val, y_val


#Converting data from 2D to 3D to be input to LSTM
train_aus = train_features.reshape((train_features.shape[0], seqLen, 17)) 
test_aus = test_features.reshape((test_features.shape[0], seqLen, 17))
val_aus = val_features.reshape((val_features.shape[0], seqLen, 17))

#Train Data will have shape: (5990, 14, 17)
#print(train_aus.shape, val_aus.shape, test_aus.shape)
#print(train_labels.shape, test_labels.shape, val_labels.shape)


# In[17]:


test_labels


# In[18]:



#Fitting of LSTM Model
zero_bias_history = Model.fit(train_aus, train_labels, epochs = 250, batch_size = 100, validation_data = (val_aus, val_labels))


# ## **K-Fold Validation Code for Model Calling**

# In[19]:


#Prediction and MAE, PCC calculation
y_pred_train = Model.predict(train_aus)
y_pred_train = np.around(y_pred_train,3)
y_pred_test = Model.predict(test_aus)
y_pred_test = np.around(y_pred_test,3)
train_mae = mean_absolute_error(train_labels, y_pred_train) ##mean squarred train error
test_mae = mean_absolute_error(test_labels, y_pred_test) #mean squarred test error
y_train = train_labels.reshape(-1,1)
b = np.corrcoef(y_train.T,y_pred_train.T)
train_PCC= b[0][1]
y = test_labels.reshape(-1,1)
a = np.corrcoef(y.T,y_pred_test.T)
test_PCC = a[0][1]
tr_acc = 1 - train_mae
te_acc = 1-test_mae


# In[20]:


#Printing of all values
print("Train_MAE {0}".format(np.round(train_mae,3)))
print("Test_MAE {0}".format(np.round(test_mae,3)))
print("Train_Acc {0}".format(np.round(tr_acc,3)))
print("Test_Acc {0}".format(np.round(te_acc,3)))
print("Train_PCC {0}".format(np.round(train_PCC,3)))
print("Test_PCC {0}".format(np.round(test_PCC,3)))


# In[ ]:




