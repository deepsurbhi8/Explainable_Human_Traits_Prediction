#!/usr/bin/env python
# coding: utf-8

# ## **Importing Libraries**

# In[2]:


#import required modules
#basic
import numpy as np
import pandas as pd
import glob
from math import sqrt
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn for required metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

#tensorflow and other imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

#importing keras layers and models as required
from keras.layers import merge
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input, Dense, concatenate


# File Paths

# In[16]:


#function to convert continous labels into binary labels
def bin_labels(data_rec):             
    count_0 = 0
    count_1 = 0
    median_value = np.median(data_rec)
    for it in range(0,len(data_rec),1):
        if data_rec[it]<=median_value :
            count_0 += 1
            data_rec[it]=0
        else:
            count_1 += 1
            data_rec[it]=1
    print(str(count_0)+": "+ str(count_1))
    return data_rec  


# ## **LSTM Parameters**

# In[15]:


# #finding the class division -- to find the number of rows belonging to each data class and the median for each label

# y_train_path = 'Kinemes/train_O.npy'
# X_train_path = 'Kinemes/train_kineme_5990.npy'
# y_test_path = 'Kinemes/test_O.npy'
# X_test_path = 'Kinemes/test_kineme_1997.npy'
# y_val_path = 'Kinemes/val_O.npy'
# X_val_path = 'Kinemes/val_kineme_1995.npy'
# print("Neuroticism")
# y_train = np.load(y_train_path)
# y_train = y_train[:,1].astype(np.float)
# y_train = bin_labels(y_train)
# train_mat = np.load(X_train_path)
# y_test = np.load(y_test_path)
# y_test = y_test[:,1].astype(np.float)
# y_test = bin_labels(y_test)
# test_mat = np.load(X_test_path)
# y_val = np.load(y_val_path)
# y_val = y_val[:,1].astype(np.float)
# y_val = bin_labels(y_val)
# val_mat = np.load(X_val_path)


# In[17]:


#define all data and label paths in npy format (can use csv format as well; just access the csv file using readcsv)
y_train_path = 'Kinemes/train_O.npy'
X_train_path = 'Kinemes/train_AU_KIN_FICS.npy'
y_test_path = 'Kinemes/test_O.npy'
X_test_path = 'Kinemes/test_AU_KIN_FICS.npy'
y_val_path = 'Kinemes/val_O.npy'
X_val_path = 'Kinemes/val_AU_KIN_FICS.npy'
training_lstm(y_train_path, X_train_path, y_test_path, X_test_path, y_val_path, X_val_path)


# In[16]:


def training_lstm(y_train_path, X_train_path, y_test_path, X_test_path, y_val_path, X_val_path):
    # parameters
    #nKineme contains the number of kinemes clusters created
    #seqLen defines the length/size of eaach chunk
    #nAction is the size of vector for action units; as we extract 17 action units from openface
    nKineme, seqLen, nClass = 16, 14, 2
    nAction = 17
    EPOCHS = 50
    BATCH_SIZE = 32
    nNeuron = 12

    #load and convert each label to categorical data
    y_train = np.load(y_train_path)
    y_train = y_train[:,1].astype(np.float)
    y_train = bin_labels(y_train)
    train_mat = np.load(X_train_path)
    y_test = np.load(y_test_path)
    y_test = y_test[:,1].astype(np.float)
    y_test = bin_labels(y_test)
    test_mat = np.load(X_test_path)
    y_val = np.load(y_val_path)
    y_val = y_val[:,1].astype(np.float)
    y_val = bin_labels(y_val)
    val_mat = np.load(X_val_path)
    


    #The model defined
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    Model = Sequential()

    Model.add(LSTM(nNeuron,activation="tanh",dropout=0.1,input_shape=(seqLen, nAction)))
    #regressor.add(Dropout(0.0))

    Model.add(Dense(units = nClass,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.01)

    Model.compile(optimizer = opt, loss = 'categorical_crossentropy',metrics=['accuracy'])
    Model.summary()



    test_loss=[]
    test_acc=[]
    train_loss = []
    train_acc = []
    fi_weighted=[]
    fi_macro=[]
    val_loss = []
    val_acc = []

    #train test split is not required as we have an already defined(fixed) train, test and validation split
#     random_state = 42
#     rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
#     for train_idx, test_idx in rkf.split(X_data):
#         train_features, test_features, train_labels, test_labels = X_data[train_idx], X_data[test_idx], y_data[train_idx], y_data[test_idx] 
    
    train_features,  train_labels = train_mat, y_train
    test_features,  test_labels = test_mat, y_test
    val_features,  val_labels = val_mat, y_val
    print(train_features.shape, train_labels.shape)
    train_action = train_features[:, seqLen:]
    test_action = test_features[:, seqLen:]
    val_action = val_features[:, seqLen:]
    print(train_action.shape)
    train_action = train_action.reshape((train_action.shape[0], seqLen, nAction))
    test_action = test_action.reshape((test_action.shape[0], seqLen, nAction))
    val_action = val_action.reshape((val_action.shape[0], seqLen, nAction))
    # convert labels into categorical
    train_labels = to_categorical(train_labels)   
    val_labels = to_categorical(val_labels)  
    #train_features, val_data, train_labels, val_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)
    zero_bias_history = Model.fit(train_action, train_labels, epochs = EPOCHS, batch_size = 32, validation_data=(val_action, val_labels), callbacks=[callback]) 
    score = Model.evaluate(test_action, to_categorical(test_labels), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    test_loss.append(score[0])
    test_acc.append(score[1])
    #trai_loss.append(zero_bias_history.history['loss'][9])
    train_acc.append(np.array(zero_bias_history.history['accuracy']).mean())
    val_acc.append(np.array(zero_bias_history.history['val_accuracy']).mean())
    #va_loss.append(zero_bias_history.history['val_loss'][9])
    #val_acc.append(zero_bias_history.history['val_accuracy'][9])
    y_testpred = Model.predict_classes(test_action)
#     y_testclass = np.argmax(test_labels,axis=-1)
    f1_w_epoch = f1_score(test_labels, y_testpred, average='weighted')
    f1_m_epoch = f1_score(test_labels, y_testpred, average='macro')
    fi_weighted.append(f1_w_epoch)
    fi_macro.append(f1_m_epoch)


    print("For: Openness")
    print("Train_accuracy {0}±{1}".format(round(np.array(train_acc).mean(),3),round(np.array(train_acc).std(),3)))
    print("Test_accuracy {0}±{1}".format(round(np.array(test_acc).mean(),3),round(np.array(test_acc).std(),3)))
    print("F1_Weighted {0}±{1}".format(round(np.array(fi_weighted).mean(),3),round(np.array(fi_weighted).std(),3)))
    print("F1_Macro {0}±{1}".format(round(np.array(fi_macro).mean(),3),round(np.array(fi_macro).std(),3)))

