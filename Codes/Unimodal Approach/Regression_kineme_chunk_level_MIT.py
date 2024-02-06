#!/usr/bin/env python
# coding: utf-8

# ## **Importing Libraries**

# In[38]:

#Importing of all libraries
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

# In[39]:


file_list = sorted(glob.glob('Path..........')) #Path for kineme files
la_path = 'Path............' #Path for labels


# In[40]:


len(file_list)  #This should be 138 for number of MIT files


# ## **Kineme Data Preparation**

# In[41]:


#FUNCTION FOR PREPARING CHUNKS OF COMPLETE VIDEO
def data_preprocess(chunk_time, input_file_list, label_path,l): 
    data_list = []
    total_num_chunks = 0
    label_file_index = 0
    final_label = []
    
    #print("Creating the input from csv files")
    for file_name in input_file_list:
    
        read_file = pd.read_csv(file_name) #to add column names
        #print(file_name + "-- Added this file")
        size_file = read_file.shape[1]
        #print(size_file)
        column_name = []
        for iter_num in range(0,size_file,1):
            column_name.append(iter_num) #loop for col_names
            
        final_data_file = pd.read_csv(file_name, header=None) #actual csv read
        #print(s1)
        final_data_file = final_data_file.T
        #num_chunk = int(size_file/chunk_time)
        num_chunk = int(size_file/chunk_time)
        total_num_chunks += num_chunk
        #print(num_chunk)
        
        for num in range(0, num_chunk):
            data_chunk = final_data_file[num*(chunk_time-1):((num+1)*(chunk_time-1))]
            data_chunk = data_chunk.to_numpy()
            data_list.append(data_chunk)
        
        label_data = pd.read_csv(label_path)
        updated_label = label_data[l]
        one_list = updated_label[label_file_index].repeat(num_chunk)
        final_label.extend(one_list)
        #print(final_label)
        #print(total_num_chunks, np.shape(final_label))
        label_file_index += 1
                    
    #Converting list to array
    data_array = np.array(data_list)
    #print("shape of data_array is" + str(np.shape(data_array)))
    
    #taking an array of random numbers to just create the dataset
    randnums= np.random.randint(1,101,chunk_time-1)
    randnums = randnums.reshape(-1,1) #for 1D to 2D
    
    new_data_array = data_array[0]
    updated_data = np.hstack((randnums, new_data_array)) #first 2 entry of new1
    for i in range(1,len(data_array),1):
        new_data_array = data_array[i]
        updated_data = np.hstack((updated_data, new_data_array))

    final_data = updated_data[:,1:]
    #print("Shape of final data" + str(np.shape(final_data)))
    
    final_data = final_data.T #final data to work on
    #print(final_data)
      
    #print("Read final labels")    
    final_label = pd.DataFrame(final_label, columns=(["data"]))
    #print(final_label)
    final_updated_label = final_label.get("data")
    #print(final_updated_label)
    
    #print(final_label)
    #print(np.shape(final_label))
    print("For size of " + str(chunk_time) + " each, we have total " + str(total_num_chunks) + " chunks for all file")
    print("Shape of input datatset: " + str(np.shape(final_data)) + " and target data shape" + str(np.shape(final_updated_label)) + " for Overall label")
    return  final_data, final_updated_label


# ## **Encoding of Input Data**

# In[42]:


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


# ## **LSTM Parameters**

# In[43]:


l_sec = 10   #Chunk size
l ='Excited'  #Label Name
# parameters
nKineme, seqLen, nClass = 16, l_sec-1, 1
BATCH_SIZE = 32
nNeuron = 20
#Data Formation
X_data, y_data = data_preprocess(l_sec, file_list, la_path,l)  #Calling for Data Preparation

y_data = np.around(y_data,2)   #Rounding of Labels


# In[44]:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
Model = Sequential()

Model.add(LSTM(20,activation="tanh",dropout=0.2,recurrent_dropout=0.0,input_shape=(seqLen, 16)))

Model.add(Dense(units = nClass, activation='linear'))
opt = keras.optimizers.Adam(learning_rate=0.01)

Model.compile(optimizer = opt, loss = 'mean_absolute_error')
Model.summary()

# In[45]:

#Lists for MAE and PCC
train_mse=[]
test_mse =[]
train_PCC =[]
test_PCC = []
train_Acc = []
test_Acc =[]
n=1

random_state = 42
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
for train_idx, test_idx in rkf.split(X_data):
    train_features, test_features, train_labels, test_labels = X_data[train_idx], X_data[test_idx], y_data[train_idx], y_data[test_idx] 
    train_features = ks_encoding(train_features, nKineme) #One hot encoding for kinemes
    test_features = ks_encoding(test_features, nKineme)
    zero_bias_history = Model.fit(train_features, train_labels, epochs = 30, batch_size = 32, validation_split = 0.1,callbacks=[callback])  
    
    y_pred_train = Model.predict(train_features)  #Model Preduction
    y_pred_train = np.around(y_pred_train,2)
    y_pred_test = Model.predict(test_features)
    y_pred_test = np.around(y_pred_test,2)
    train_mse.append(mean_absolute_error(train_labels, y_pred_train)) ##mean absoulte train error
    test_mse.append(mean_absolute_error(test_labels, y_pred_test)) #mean absolute test error
    train_Acc.append(1-mean_absolute_error(train_labels, y_pred_train))
    test_Acc.append(1-mean_absolute_error(test_labels, y_pred_test))
    y_train = train_labels.to_numpy().reshape(-1,1)
    b = np.corrcoef(y_train.T,y_pred_train.T)
    train_PCC.append(b[0][1])
    y = test_labels.to_numpy().reshape(-1,1)
    a = np.corrcoef(y.T,y_pred_test.T)
    test_PCC.append(a[0][1])
    print(n)
    n = n+1


    
# In[46]:

#Printing of all matrices
print("For time {0} and label {1}".format(l_sec,l))
print("Train MAE is:{0}±{1}".format(round(np.array(train_mse).mean(),3),round(np.array(train_mse).std(),2)))
print("Test MAE is:{0}±{1}".format(round(np.array(test_mse).mean(),3),round(np.array(test_mse).std(),2)))
#print("For time {0} and label {1}".format(l_sec,l))
print("Train-Accuracy is:{0}±{1}".format(round(np.array(train_Acc).mean(),3),round(np.array(train_Acc).std(),2)))
print("Test-Accuracy is:{0}±{1}".format(round(np.array(test_Acc).mean(),3),round(np.array(test_Acc).std(),2)))
print("PCC for training data is:{0}±{1}".format(round(np.array(train_PCC).mean(),3),round(np.array(train_PCC).std(),2)))
print("PCC for testing data is:{0}±{1}".format(round(np.array(test_PCC).mean(),3),round(np.array(test_PCC).std(),2)))


