#!/usr/bin/env python
# coding: utf-8

# **Max Pooling Encoding**

# In[11]:


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
# from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# from sklearn.linear_model import Ridge


# In[12]:


def bin_labels(y1):              #fun to converting continous labels into binary labels
    co = 0
    co1 = 0
    val = np.median(y1)
    for i in range(0,len(y1),1):
        if y1[i]<=val :
            co += 1
            y1[i]=0
        else:
            co1 += 1
            y1[i]=1
    print(co, co1)
    return y1  


# In[13]:


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


# In[35]:


# parameters
nKineme, seqLen, nClass = 16, 59, 2


# In[36]:

#LSTM Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input, Dense, concatenate

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#Left InputLSTM
left_branch_input = Input(shape=(seqLen,nKineme), name='Left_input')
left_branch_output = LSTM(20, activation='tanh')(left_branch_input)

#Right Input LSTM
right_branch_input = Input(shape=(seqLen,17), name='Right_input')
right_branch_output = LSTM(20, activation='tanh')(right_branch_input)

#Merged Layer
merged = concatenate([left_branch_output, right_branch_output], name='Concatenate')
final_model_output = Dense(1, activation='linear')(merged)
final_model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                    name='Final_output')

opt = keras.optimizers.Adam(learning_rate=0.01) #Optimizer

final_model.compile(optimizer = opt, loss = 'mean_absolute_error')
final_model.summary()
# In[37]:


#Data Formation
final_mat = np.load('Path for Data marrix.......... as npy for eg "Data_60_Excited.npy"')
y_data = np.load('Path for labels as ....................Label_60_Excited.npy')



# In[40]:

#Lists to contain MAE and PCC
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
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
for train_idx, test_idx in rkf.split(final_mat):
    train_features, test_features, train_labels, test_labels = final_mat[train_idx], final_mat[test_idx], y_data[train_idx], y_data[test_idx] 
    train_kinemes = ks_encoding(train_features[:,0:seqLen], 16) #One hot encoding for kineme
    test_kinemes = ks_encoding(test_features[:,0:seqLen], 16)
    # print(train_features.shape)
    train_action = train_features[:, seqLen:] #Reshaping of data for AU
    test_action = test_features[:, seqLen:]
    train_aus = train_action.reshape((train_action.shape[0], seqLen, 17))
    test_aus = test_action.reshape((test_action.shape[0], seqLen, 17))
    print(np.shape(train_kinemes), np.shape(test_kinemes), np.shape(train_aus), np.shape(test_aus))

    zero_bias_history = final_model.fit([train_kinemes, train_aus], train_labels, epochs = 30, batch_size = 32, validation_split = 0.1,callbacks=[callback])  
    
    trainpredmerge=final_model.predict([train_kinemes, train_aus])
   
    y_pred_train = final_model.predict([train_kinemes, train_aus])
    y_pred_train = np.around(y_pred_train,2)
    y_pred_test = final_model.predict([test_kinemes, test_aus])
    y_pred_test = np.around(y_pred_test,2)
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

     


# In[41]:
#Printing of all matrices
print("For 20 neurons 60 sec Excited")
print("Train accuracy {0}±{1}".format(round(np.array(train_mse).mean(),3),round(np.array(train_mse).std(),3)))
print("Test accuracy {0}±{1}".format(round(np.array(test_mse).mean(),3),round(np.array(test_mse).std(),3)))
print("Train PCC {0}±{1}".format(round(np.array(train_PCC).mean(),3),round(np.array(train_PCC).std(),3)))
print("Test PCC {0}±{1}".format(round(np.array(test_PCC).mean(),3),round(np.array(test_PCC).std(),3)))

