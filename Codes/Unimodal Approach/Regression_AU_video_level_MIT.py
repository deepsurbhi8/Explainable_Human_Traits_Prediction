#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Importing of all libraries
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
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


# In[2]:


#Function to create kineme data matrix for train
def data_preprocess(chunk_time, input_file_kineme, label_val,num_chunk): 
    data_list = []
    total_num_chunks = 0
    label_file_index = 0
    final_label = []
    
    #print("Creating the input from csv files")
    final_data_file = input_file_kineme.T
    #num_chunk = int(size_file/chunk_time)
        
    for num in range(0, num_chunk):
        data_chunk = final_data_file[num*(chunk_time-1):((num+1)*(chunk_time-1))]
        data_chunk = data_chunk.to_numpy()
        data_list.append(data_chunk.flatten())
    #print(data_list)
    one_list = label_val.repeat(num_chunk)
    final_label.extend(one_list)
    #print(final_label)
    #print(total_num_chunks, np.shape(final_label))
    #label_file_index += 1                   
    return  data_list, final_label


#Function to create kineme data matrix for test
def data_preprocess_test(chunk_time, input_file_kineme,num_chunk): 
    data_list = []
    total_num_chunks = 0
    label_file_index = 0
    final_label = []
    
    #print("Creating the input from csv files")
    final_data_file = input_file_kineme.T
    #num_chunk = int(size_file/chunk_time)
        
    for num in range(0, num_chunk):
        data_chunk = final_data_file[num*(chunk_time-1):((num+1)*(chunk_time-1))]
        data_chunk = data_chunk.to_numpy()
        data_list.append(data_chunk.flatten())
    #print(data_list)
    '''one_list = label_val.repeat(num_chunk)
    final_label.extend(one_list)
    print(final_label)
    #print(total_num_chunks, np.shape(final_label))
    #label_file_index += 1 '''                  
    return  data_list



#Function to create AU Data Matrix
def max_encoding(input_file_au,threshold,chunk_size,total_chunks):
  count = 0
  duration = 0
  overlap = 10
  complete_vec = []
  intensity_var = threshold  
  read_data = np.array(input_file_au) 
  for c in range(0,total_chunks,1):           #total_chunks= 10 , c= 0 
      #print("Value of total chunks")
      #print(c)
      vector = []
      #print(c)
      for j in range(0, chunk_size-1, 1): #0 1 2 0 20 10 30 20 40
         #print(j)
         max_pool = [] 
         for i in range(5, 22, 1):
             max_value = np.max(read_data[duration:duration+20, i])
             #print("max value in this column")
             #print(max_value, i)
             if max_value <= intensity_var:
                 max_value = 0
             else:
                 max_value = 1
             max_pool.append(max_value)
             # print("vector for 2 seconds")
             # print(max_pool)
         duration = duration + overlap
         vector.append(max_pool)
         #print(vector)
      vec_flat = np.array(vector).flatten()
      #print(vec_flat)
      complete_vec.append(vec_flat)
      # print("duration and overlap")
      # print(duration, overlap)
      # print(vector)
      # complete_vec.append(vector)  
  return complete_vec


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
    encoded_features = np.asarray([np.asarray(onehot_encoding(ks[i], nKineme)) for i in range(m)])
    return encoded_features


# In[4]:


# parameters
chunk_size = 5  #like 5 sec data chunks to be taken from complete video
l_name = 'RecommendHiring'  #Label name
nKineme, seqLen, nClass = 16,chunk_size-1 , 1


# In[5]:
    
#LSTM Model
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

# In[6]:


file_list = sorted(glob.glob('Path for AU Data Files......../*.csv'))
label_data = pd.read_csv('Path for labels.............')
label = label_data[l_name] #L_name, label name to be passed for eg overall


# In[8]:


f = np.array(file_list).reshape(-1,1) #All data files


# In[9]:


label = pd.to_numeric(label, downcast='integer') #Float to integer conversion
label = label.to_numpy().reshape(-1,1)  
#print(label.shape)
file_to_work = np.concatenate((f, label),axis=1)   #File plus label in single array


# In[11]:


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
for train_idx, test_idx in rkf.split(file_to_work):
    train_features, test_features = file_to_work[train_idx], file_to_work[test_idx]  #train-test split
    #Calling for train-data
    file_path = train_features[:,0].tolist()  #File list for chunk preparation
    labels = train_features[:,1]  #labels for chunk preparation

    file_name = os.path.basename(file_path[0] )  #eds_report.csv  taking only basename

    AU_path = 'Path for AU data files of 10 fps......./' + file_name 
    AU1 = pd.read_csv(AU_path)

    label = labels[0]

    chun_au = int(AU1.shape[0]/(chunk_size*10))

    #Calling of functions
    au_res1 = max_encoding(AU1,1.5,chunk_size,total_size)
    au_res1 = np.array(au_res1)
    #Looping to create complete AU matrix
    for i in range(0,len(file_path)-1,1):
        file_name = os.path.basename(file_path[i+1] )  #eds_report.csv
        AU_path = 'Path for AU data files of 10 fps....../' + file_name 
        AU2 = pd.read_csv(AU_path)
        label2 = labels[i+1]
        chun_au = int(AU2.shape[0]/(chunk_size*10))
        au_res2 = max_encoding(AU2,1.5,chunk_size,total_size)
      
        #make array of results
        au_res2 = np.array(au_res2)       
        #Matrix merging
        au_res1 = np.vstack((au_res1,au_res2))
        label1.extend(label2)
    final = au_res1
    final_label = np.array(label1)
    final_label = [float(i) for i in final_label]   #String to int conversion
    final_label = np.array(final_label)
    train_labels = np.around(final_label,3)

    train_action = final[:, seqLen:]
    train_aus = train_action.reshape((train_action.shape[0], seqLen, 17)) #Reshaping to be input for LSTM
    zero_bias_history = Model.fit(train_aus, final_label, epochs = 30, batch_size = 32, validation_split = 0.1,callbacks=[callback])

    #Process for testing
    test_data = test_features[:,0].tolist()  #test data
    test_labels = test_features[:,1]  #test actual label
    test_labels = [float(i) for i in test_labels]   #String to int conversion
    test_labels = np.around(test_labels,3)
    y_pred_video = []  #List for prediction of test labels
    
    for i in range(0,len(test_data)):
        file_name = os.path.basename(test_data[i] )  #eds_report.csv
        AU_path = 'Path for AU data files of 10 fps....../' + file_name 

        AU2 = pd.read_csv(AU_path)  #Read csv file.......

        label2 = test_labels[i]
        chun_au = int(AU2.shape[0]/(chunk_size*10)) #chunk size to be taken

        l = label[0]

        au_res2 = max_encoding(AU2,1.5,chunk_size,total_size)
  
        #make array of results
        au_res2 = np.array(au_res2)
        final_test = au_res2

        test_action = final_test[:, seqLen:]
        test_aus = test_action.reshape((test_action.shape[0], seqLen, 17))
        y_pred = Model.predict(test_aus)
        y1 = np.mean(y_pred)   #Mean value over the chunks of a video to calculate the predicted label
        y_pred_video.append(y1)
    y_pred_video = np.around(y_pred_video,3)
    test_mse.append(1-mean_absolute_error(test_labels, y_pred_video)) #mean absolute test error
    print(test_mse)               
    y = np.array(test_labels).reshape(-1,1)
    a = np.corrcoef(y.T,np.array(y_pred_video).T)
    test_PCC.append(a[0][1])
    y_pred_train = Model.predict(train_aus)
    y_pred_train = np.around(y_pred_train,3)
    train_mse.append(1-mean_absolute_error(train_labels, y_pred_train)) ##mean absolute train error
    print(train_mse)
    y_train = train_labels.reshape(-1,1)
    b = np.corrcoef(y_train.T,y_pred_train.T)
    train_PCC.append(b[0][1])   #PCC value calculation
    print(n)
    n = n+1



# In[14]:

#Printing of all values

print("For 20 neurons {0} sec and label {1}".format(chunk_size,l_name))
print("Train accuracy {0}±{1}".format(round(np.array(train_mse).mean(),3),round(np.array(train_mse).std(),3)))
print("Test accuracy {0}±{1}".format(round(np.array(test_mse).mean(),3),round(np.array(test_mse).std(),3)))
print("Train PCC {0}±{1}".format(round(np.array(train_PCC).mean(),3),round(np.array(train_PCC).std(),3)))
print("Test PCC {0}±{1}".format(round(np.array(test_PCC).mean(),3),round(np.array(test_PCC).std(),3)))




