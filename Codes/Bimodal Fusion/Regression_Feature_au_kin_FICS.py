#!/usr/bin/env python
# coding: utf-8

# **Importing of all Libraries**

# In[2]:


import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
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
from keras.layers import merge
import numpy as np
# from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# from sklearn.linear_model import Ridge


# In[3]:


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


# In[4]:


# parameters for LSTM Model
nKineme, seqLen, nClass = 16, 14, 1 


# In[5]:


#Two input LSTM Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input, Dense, concatenate

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#Left input LSTM
left_branch_input = Input(shape=(seqLen,nKineme), name='Left_input')
left_branch_output = LSTM(32, activation='tanh')(left_branch_input)

#Right input LSTM
right_branch_input = Input(shape=(seqLen,17), name='Right_input')
right_branch_output = LSTM(32, activation='tanh')(right_branch_input)

# Merged Layer
merged = concatenate([left_branch_output, right_branch_output], name='Concatenate')
final_model_output = Dense(1, activation='linear')(merged)
final_model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                    name='Final_output') 
opt = keras.optimizers.Adam(learning_rate=0.01)  #Optimizer
final_model.compile(optimizer = opt, loss = 'mean_absolute_error')
final_model.summary()


# In[6]:


#For this we have prepared Kineme+AU Matrix for train, test and validation data of FICS Dataset
#This is to be in input format for LSTM. First 14 entries are kineme while end entries are AUs
#These matrices are .npy files
train_features = np.load('Training Data matrix path here........') #Training Path
train_labels = np.load('Training Labels Path here...........') #Train Labels
train_labels = train_labels[:,1]
train_labels = pd.to_numeric(train_labels, downcast='integer') #Float to integer conversion
train_labels = np.around(train_labels,3)
test_features = np.load('Test Matrix Path here......') 
test_labels = np.load('Test Label Path here.......')
test_labels = test_labels[:,1]
test_labels = pd.to_numeric(test_labels, downcast='integer') #Float to integer conversion
test_labels = np.around(test_labels,3)
val_features = np.load('validation Data matrix path here........')
val_labels = np.load('Validation Labels Path here...........')
val_labels = val_labels[:,1]
val_labels = pd.to_numeric(val_labels, downcast='integer') #Float to integer conversion
val_labels = np.around(val_labels,3)


# In[7]:


print(train_labels.shape)
print(test_labels.shape)
print(val_labels.shape)


# In[8]:


#Printing of all Data shapes
print(train_features.shape)
print(test_features.shape)
print(val_features.shape)


# In[9]:


#Lists to have MAE and PCC
test_loss=[]
test_acc=[]
trai_loss = []
trai_acc = []
f1_wei=[]
f1_mac=[]
va_loss = []
val_acc = []
train_mse =[]
test_mse =[]
train_PCC =[]
test_PCC =[]
n=1
random_state = 42
print(n)

#KS Encoding
train_kinemes = ks_encoding(train_features[:,0:seqLen], 16)
test_kinemes = ks_encoding(test_features[:,0:seqLen], 16)
val_kinemes = ks_encoding(val_features[:,0:seqLen], 16)

#REshaping of Action Units
train_action = train_features[:, seqLen:]
test_action = test_features[:, seqLen:]
val_action = val_features[:, seqLen:]
train_aus = train_action.reshape((train_action.shape[0], seqLen, 17))
test_aus = test_action.reshape((test_action.shape[0], seqLen, 17))
val_aus = val_action.reshape((val_action.shape[0], seqLen, 17))
print(np.shape(train_kinemes), np.shape(test_kinemes), np.shape(train_aus), np.shape(test_aus))


#Fitting of Model
zero_bias_history = final_model.fit([train_kinemes, train_aus], train_labels, epochs = 50, batch_size = 32, validation_data = ([val_kinemes, val_aus],val_labels))

#Prediction of values
trainpredmerge=final_model.predict([train_kinemes, train_aus])
y_pred_train = final_model.predict([train_kinemes, train_aus])
y_pred_train = np.around(y_pred_train,3)
y_pred_test = final_model.predict([test_kinemes, test_aus])
y_pred_test = np.around(y_pred_test,3)
train_mse.append(1-mean_absolute_error(train_labels, y_pred_train)) ##mean squarred train error
print(train_mse)
test_mse.append(1-mean_absolute_error(test_labels, y_pred_test)) #mean squarred test error
print(test_mse)
y_train = train_labels.reshape(-1,1)
b = np.corrcoef(y_train.T,y_pred_train.T)
train_PCC.append(b[0][1])
y = test_labels.reshape(-1,1)
a = np.corrcoef(y.T,y_pred_test.T)
test_PCC.append(a[0][1])
print(n)
n = n+1
     


# In[11]:


#Printing of all results
print("For 100 neurons 60 sec Agreeableness")
print("Train accuracy {0}±{1}".format(round(np.array(train_mse).mean(),3),round(np.array(train_mse).std(),3)))
print("Test accuracy {0}±{1}".format(round(np.array(test_mse).mean(),3),round(np.array(test_mse).std(),3)))
print("Train PCC {0}±{1}".format(round(np.array(train_PCC).mean(),3),round(np.array(train_PCC).std(),3)))
print("Test PCC {0}±{1}".format(round(np.array(test_PCC).mean(),3),round(np.array(test_PCC).std(),3)))


# In[ ]:




