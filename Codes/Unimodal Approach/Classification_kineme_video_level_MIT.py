#!/usr/bin/env python
# coding: utf-8

# ## **Importing Libraries**

# In[1]:


#import required modules
#basic
import numpy as np
import pandas as pd
import glob
import os
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

# In[2]:


#Function to create kineme data matrix
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


#Function to create kineme data matrix
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


# In[5]:


from collections import Counter

#function to return the majority from a list of labels
def majority_vote(arr):
  freqDict = Counter(arr)
  size = len(arr)
  for (key, val) in freqDict.items():
    if (val > (size/2)):
      return key
    else:
      return np.random.randint(2)

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


# ## **LSTM Parameters**

# In[6]:


def training_lstm_video(label_name, seq_length):
    #path to all the kineme files
    file_list = sorted(glob.glob('MIT/Video_level/AU_10fps/*.csv'))
    label_data = pd.read_csv('MIT/Video_level/labels_traits_minmax_norm.csv')
    label = label_data[label_name]
    f = np.array(file_list).reshape(-1,1) #All data files
    label = pd.to_numeric(label, downcast='integer') #Float to integer conversion
    label = label.to_numpy().reshape(-1,1)  
    #print(label.shape)
    file_to_work = np.concatenate((f, label),axis=1)   #File plus label in single array


    # parameters
    nKineme, seqLen, nClass = 16, seq_length, 2
    EPOCHS = 30
    BATCH_SIZE = 32
    nNeuron = 12
    chunk_size = seqLen + 1
    nAction = 17


    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    Model = Sequential()

    Model.add(LSTM(nNeuron,activation="tanh",dropout=0.1,input_shape=(seqLen, nKineme)))
    #regressor.add(Dropout(0.0))

    Model.add(Dense(units = nClass,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.01)

    Model.compile(optimizer = opt, loss = 'binary_crossentropy',metrics=['accuracy'])
    Model.summary()


    from sklearn.metrics import accuracy_score
    test_loss=[]
    test_acc=[]
    train_loss = []
    train_acc = []
    fi_weighted=[]
    fi_macro=[]
    val_loss = []
    val_acc = []
    n=1
    random_state = 42
    # print(n)
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
    for train_idx, test_idx in rkf.split(file_to_work):
        train_features, test_features = file_to_work[train_idx], file_to_work[test_idx] 
        #Calling for train-data
        file_path = train_features[:,0].tolist()  #File list for chunk preparation
        labels = train_features[:,1]  #labels for chunk preparation

        file_name = os.path.basename(file_path[0] )  #eds_report.csv

        AU_path = 'MIT/Video_level/AU_10fps/' + file_name 
        Kine_Path = 'MIT/Video_level/Kinemes/' + file_name

        AU1 = pd.read_csv(AU_path)
        Kin1 = pd.read_csv(Kine_Path, header=None)
        label = labels[0]

        chun_au = int(AU1.shape[0]/(chunk_size*10))
        chun_ki = int(Kin1.shape[1]/(chunk_size))
        if chun_au<chun_ki:
            total_size = chun_au
        else:
            total_size = chun_ki
    #for AU directly put chunk size as total size from the chun_au

        #Calling of functions
        kin_res1,label1 = data_preprocess(chunk_size,Kin1,label,total_size)
        au_res1 = max_encoding(AU1,1.5,chunk_size,total_size)
        kin_res1 = np.array(kin_res1)
        au_res1 = np.array(au_res1)
        #Looping to create all matrix of kineme and AU
        for i in range(0,len(file_path)-1,1):
            file_name = os.path.basename(file_path[i+1] )  #eds_report.csv
            AU_path = 'MIT/Video_level/AU_10fps/' + file_name 
            Kine_Path = 'MIT/Video_level/Kinemes/' + file_name
            AU2 = pd.read_csv(AU_path)
            Kin2 = pd.read_csv(Kine_Path, header=None)
            label2 = labels[i+1]
            chun_au = int(AU2.shape[0]/(chunk_size*10))
            chun_ki = int(Kin2.shape[1]/(chunk_size))
            if chun_au<chun_ki:
                total_size = chun_au
            else:
                total_size = chun_ki

            kin_res2,label2 = data_preprocess(chunk_size,Kin2,label2,total_size)
            au_res2 = max_encoding(AU2,1.5,chunk_size,total_size)

            #make array of results
            kin_res2 = np.array(kin_res2)
            au_res2 = np.array(au_res2)

            #Matrix merging
            kin_res1 = np.vstack((kin_res1,kin_res2))
            au_res1 = np.vstack((au_res1,au_res2))
            label1.extend(label2)
        # final = np.concatenate((kin_res1.T, au_res1.T)).T
        final = np.concatenate((kin_res1.T, au_res1.T)).T
        final_label = np.array(label1)
        final_label = [float(i) for i in final_label]   #String to int conversion
        final_label = bin_labels(final_label)
        # print(final_label)
        final_label = np.array(final_label)
        train_labels = to_categorical(final_label)

        train_kinemes = ks_encoding(final[:,0:seqLen], 16)
        train_action = final[:, seqLen:]
        train_aus = train_action.reshape((train_action.shape[0], seqLen, 17))
        zero_bias_history = Model.fit(train_kinemes, train_labels, epochs = EPOCHS, batch_size = 32, validation_split = 0.1,callbacks=[callback])

        #Process for testing
        test_data = test_features[:,0].tolist()  #test data
        test_labels = test_features[:,1]  #test actual label
        test_labels = [float(i) for i in test_labels]   #String to int conversion
        test_labels = bin_labels(test_labels)
        y_pred_video = []
        for i in range(0,len(test_data)):
            file_name = os.path.basename(test_data[i] )  #eds_report.csv
            AU_path = 'MIT/Video_level/AU_10fps/' + file_name 
            Kine_Path = 'MIT/Video_level/Kinemes/' + file_name
            AU2 = pd.read_csv(AU_path)
            Kin2 = pd.read_csv(Kine_Path, header=None)
            label2 = test_labels[i]
            chun_au = int(AU2.shape[0]/(chunk_size*10))
            chun_ki = int(Kin2.shape[1]/(chunk_size))
            if chun_au<chun_ki:
                total_size = chun_au
            else:
                total_size = chun_ki
            l = label[0]
            kin_res2 = data_preprocess_test(chunk_size,Kin2,total_size)
            au_res2 = max_encoding(AU2,1.5,chunk_size,total_size)

            #make array of results
            kin_res2 = np.array(kin_res2)
            au_res2 = np.array(au_res2)
            final_test = np.concatenate((kin_res2.T, au_res2.T)).T
            test_kinemes = ks_encoding(final_test[:,0:seqLen], 16)
            test_action = final_test[:, seqLen:]
            test_aus = test_action.reshape((test_action.shape[0], seqLen, 17))
            y_pred = Model.predict(test_kinemes)
            y_pred = y_pred.argmax(axis=-1)
            y1 = majority_vote(y_pred)   #Voting for classification to change
            y_pred_video.append(y1)

        # print(test_labels)
        # print("Predicted")
        # print(y_pred_video)
        acc_val = accuracy_score(test_labels, y_pred_video)
        test_acc.append(acc_val)
        print('Test accuracy:', acc_val)   
        train_acc.append(np.array(zero_bias_history.history['accuracy']).mean()) 
        val_acc.append(np.array(zero_bias_history.history['val_accuracy']).mean())  

        f1_w_epoch = f1_score(test_labels, y_pred_video, average='weighted')
        f1_m_epoch = f1_score(test_labels, y_pred_video, average='macro')
        fi_weighted.append(f1_w_epoch)
        fi_macro.append(f1_m_epoch)
        print('F1 score weighted: {0} and macro: {1}'.format(f1_w_epoch, f1_m_epoch))         
        # y = np.array(test_labels).reshape(-1,1)
        # a = np.corrcoef(y.T,np.array(y_pred_video).T)
        # test_PCC.append(a[0][1])
        # y_pred_train = final_model.predict([train_kinemes, train_aus])
        # y_pred_train = np.around(y_pred_train,2)
        # train_mse.append(1-mean_absolute_error(train_labels, y_pred_train)) ##mean squarred train error
        # print(train_mse)
        # y_train = train_labels.reshape(-1,1)
        # b = np.corrcoef(y_train.T,y_pred_train.T)
        # train_PCC.append(b[0][1])
        print(n)
        n = n+1

    #print("For 1 layer lstm(relu function) with 0.1 dropout with 50 neurons and 15 sec data (Adam 0.01): on O")
    print("For: " + str(label_name) + " with chunk_size: " + str(seq_length))
    print("Train_accuracy {0}±{1}".format(round(np.array(train_acc).mean(),3),round(np.array(train_acc).std(),3)))
    print("Test_accuracy {0}±{1}".format(round(np.array(test_acc).mean(),3),round(np.array(test_acc).std(),3)))
    print("F1_Weighted {0}±{1}".format(round(np.array(fi_weighted).mean(),3),round(np.array(fi_weighted).std(),3)))
    print("F1_Macro {0}±{1}".format(round(np.array(fi_macro).mean(),3),round(np.array(fi_macro).std(),3)))


# In[7]:


training_lstm_video("Overall", 14)
training_lstm_video("Excited", 14)
training_lstm_video("RecommendHiring", 14)
training_lstm_video("EyeContact", 14)
training_lstm_video("Friendly", 14)
# training_lstm_video("Overall", 9)
# training_lstm_video("Excited", 9)
# training_lstm_video("RecommendHiring", 9)
# training_lstm_video("EyeContact", 9)
# training_lstm_video("Friendly", 9)
# training_lstm_video("Overall", 4)
# training_lstm_video("Excited", 4)
# training_lstm_video("RecommendHiring", 4)
# training_lstm_video("EyeContact", 4)
# training_lstm_video("Friendly", 4)


# In[8]:


training_lstm_video("Overall", 9)
training_lstm_video("Excited", 9)
training_lstm_video("RecommendHiring", 9)
training_lstm_video("EyeContact", 9)
training_lstm_video("Friendly", 9)
training_lstm_video("Overall", 4)
training_lstm_video("Excited", 4)
training_lstm_video("RecommendHiring", 4)
training_lstm_video("EyeContact", 4)
training_lstm_video("Friendly", 4)

