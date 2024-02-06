#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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
    print(count_0, count_1)
    return data_rec  

# onehot encoding of kineme sequence
def onehot_encoding(kineme_seq, nKineme):
    #print(kineme_seq)
    onehot_encoded = list()
    for each_kineme in kineme_seq:
        #print(each_kineme)
        vector = [0 for _ in range(nKineme)]
        vector[each_kineme-1] = 1
        onehot_encoded.append(vector)
        #print("Vector")
        #print(vec)
    return onehot_encoded


def ks_encoding(kineme_seq, nKineme):
    # ks is a numpy ndarray
    m, n = kineme_seq.shape #m=92, n=29
    #print(m, n)
    kineme_seq = kineme_seq.tolist() #converted to list
    encoded_features = np.asarray(
        [np.asarray(onehot_encoding(kineme_seq[i], nKineme)) for i in range(m)]
    )
    return encoded_features


# In[3]:


#pass on the label data path, train data path, label, sequence length and weight used in fusion
def training_lstm(y_data_path, X_data_path, Label_class, sl, w):
    # parameters
    seqLen = sl
    nKineme, nClass = 16, 1
    nAction = 17
    EPOCHS = 30
    BATCH_SIZE = 32
    nNeuron = 12
    
    #load y_data, convert to float and then convert the values to categorical labels as 0 or 1
    y_data = np.load(y_data_path)
    y_data = y_data.astype(np.float)
    y_data = bin_labels(y_data)
    X_data = np.load(X_data_path)


    #model defined for kineme
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    Model_kineme = Sequential()

    Model_kineme.add(LSTM(nNeuron,activation="tanh",dropout=0.15,input_shape=(seqLen, nKineme)))

    Model_kineme.add(Dense(units = nClass,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.01)

    Model_kineme.compile(optimizer = opt, loss = 'binary_crossentropy',metrics=['accuracy'])
    Model_kineme.summary()
    
    
    #AU model architecture
    Model_AU = Sequential()
    Model_AU.add(LSTM(nNeuron,activation="tanh",dropout=0.1,input_shape=(seqLen, nAction)))
    Model_AU.add(Dense(units = nClass,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    Model_AU.compile(optimizer = opt, loss = 'binary_crossentropy',metrics=['accuracy'])
    Model_AU.summary()

    #define the loss and accuracy lists
    test_loss=[]
    test_acc=[]
    train_loss = []
    train_acc = []
    fi_weighted=[]
    fi_macro=[]
    val_loss = []
    val_acc = []

    random_state = 42
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
    #split X_data into train and test data for 10 fold 5 times
    for train_idx, test_idx in rkf.split(X_data):
        train_features, test_features, train_labels, test_labels = X_data[train_idx], X_data[test_idx], y_data[train_idx], y_data[test_idx] 
        #the first seqLen values of each list are the kineme values and we encode these kineme values both for train and test data
        train_kinemes = ks_encoding(train_features[:,0:seqLen], nKineme)
        test_kinemes = ks_encoding(test_features[:,0:seqLen], nKineme)
        #rest of the values in the list are AU values, reshape the values to 500*4*17(depending on the data shape and sequence length value)
        train_action = train_features[:, seqLen:]
        test_action = test_features[:, seqLen:]
        train_action = train_action.reshape((train_action.shape[0], seqLen, nAction))
        test_action = test_action.reshape((test_action.shape[0], seqLen, nAction))

        #fit the kineme and AU model
        kineme_history = Model_kineme.fit(train_kinemes, train_labels, epochs = EPOCHS, batch_size = 32, validation_split=0.1,callbacks=[callback])  #Fitting the model 
        print("Kineme Model Training is Done")
        AU_history = Model_AU.fit(train_action, train_labels, epochs = EPOCHS, batch_size = 32, validation_split=0.1,callbacks=[callback])
        print("AU Model Training is Done")
        #predict values for the training kinemes and AUs and find their weighted prediction and finally predict the label as 0 or 1 according to the final predicted value
        train_pred_kineme=Model_kineme.predict(train_kinemes)
        train_pred_au=Model_AU.predict(train_action) 
        final_train_pred = w*train_pred_kineme + (1-w)*train_pred_au
        y_pred_train = ((final_train_pred > 0.5)+0).ravel()
        #same process for the test kinemes and AUs
        test_pred_kineme = Model_kineme.predict(test_kinemes)
        test_pred_au = Model_AU.predict(test_action)
        final_test_pred = w*test_pred_kineme + (1-w)*test_pred_au
        y_pred_test = ((final_test_pred > 0.5)+0).ravel()
        #append the values to train and test accuracy
        train_acc.append(accuracy_score(train_labels, y_pred_train))
        test_acc.append(accuracy_score(test_labels, y_pred_test))
        #find the weighted and macro f1 score and return all the values
        f1_w_epoch = f1_score(test_labels, y_pred_test, average='weighted')
        f1_m_epoch = f1_score(test_labels, y_pred_test, average='macro')
        fi_weighted.append(f1_w_epoch)
        fi_macro.append(f1_m_epoch)
    return np.asarray(train_acc), np.asarray(test_acc), np.asarray(fi_weighted), np.asarray(fi_macro)


# In[4]:


y_data_path = 'Chunk_level/Label_60_Overall.npy'
X_data_path = 'Chunk_level/Data_60_Overall_new.npy'

#create a weight matrix and define the accuracy adn f1 score lists
weight_matrix = np.arange(0.0, 0.51, 0.05)
train_acc_list = list()
test_acc_list = list()
f1_weighted_list = list()
f1_macro_list = list()
train_acc_std = list()
test_acc_std = list()
f1_weighted_std = list()
f1_macro_std = list()
weight_list = list()
i = 1
#start the loop for executing the fusion architecture using each weight
#append all the mean and std values to the created lists
for each_w in weight_matrix:
    print("*********************"+ str(i) + "*********************") 
    final_train_acc, final_test_acc, final_f1_weighted, final_f1_macro = training_lstm(y_data_path, X_data_path, "Overall: 59", 59, each_w)
    weight_list.append(each_w)
    train_acc_list.append(final_train_acc.mean())
    train_acc_std.append(final_train_acc.std())
    test_acc_list.append(final_test_acc.mean())
    test_acc_std.append(final_test_acc.std())
    f1_weighted_list.append(final_f1_weighted.mean())
    f1_weighted_std.append(final_f1_weighted.std())
    f1_macro_list.append(final_f1_macro.mean())
    f1_macro_std.append(final_f1_macro.std())
    i += 1

#create a dataframe to store all the lists and dave the dataframe    
Fusion_accuraciesO_1 = pd.DataFrame(list(zip(weight_list, train_acc_list, train_acc_std, test_acc_list, test_acc_std,
                           f1_weighted_list, f1_weighted_std, f1_macro_list, f1_macro_std)) , columns =['Weight value(Kineme)', 
                            'Training accuracy', 'Train std', 'Testing accuracy', 'Test std', 'F1 weighted', 'F1 weighted std', 'F1 Macro', 'F1 macro std'])


Fusion_accuraciesO_1.to_csv("DataFrames/Overall60_1.csv", index = False)
Fusion_accuraciesO_1


# In[5]:


y_data_path = 'Chunk_level/Label_60_Overall.npy'
X_data_path = 'Chunk_level/Data_60_Overall_new.npy'

weight_matrix = np.arange(0.55, 1.01, 0.05)
# weight_matrix = [0, 1]
train_acc_list = list()
test_acc_list = list()
f1_weighted_list = list()
f1_macro_list = list()
train_acc_std = list()
test_acc_std = list()
f1_weighted_std = list()
f1_macro_std = list()
weight_list = list()
i = 11
for each_w in weight_matrix:
    print("*********************"+ str(i) + "*********************") 
    final_train_acc, final_test_acc, final_f1_weighted, final_f1_macro = training_lstm(y_data_path, X_data_path, "Overall: 59", 59, each_w)
    weight_list.append(each_w)
    train_acc_list.append(final_train_acc.mean())
    train_acc_std.append(final_train_acc.std())
    test_acc_list.append(final_test_acc.mean())
    test_acc_std.append(final_test_acc.std())
    f1_weighted_list.append(final_f1_weighted.mean())
    f1_weighted_std.append(final_f1_weighted.std())
    f1_macro_list.append(final_f1_macro.mean())
    f1_macro_std.append(final_f1_macro.std())
    i += 1

    
Fusion_accuraciesO_2 = pd.DataFrame(list(zip(weight_list, train_acc_list, train_acc_std, test_acc_list, test_acc_std,
                           f1_weighted_list, f1_weighted_std, f1_macro_list, f1_macro_std)) , columns =['Weight value(Kineme)', 
                            'Training accuracy', 'Train std', 'Testing accuracy', 'Test std', 'F1 weighted', 'F1 weighted std', 'F1 Macro', 'F1 macro std'])


Fusion_accuraciesO_2.to_csv("DataFrames/Overall60_2.csv", index = False)
Fusion_accuraciesO_2

