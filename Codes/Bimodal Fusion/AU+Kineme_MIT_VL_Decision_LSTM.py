#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import accuracy_score
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


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
#     print(count_0, count_1)
    return data_rec  


# In[3]:


#pass on the label name, sequence length and weight used in fusion
def training_lstm_video(label_name, seq_length, weight):
    #load y_data, convert to float and then convert the values to categorical labels as 0 or 1
    #path to all the kineme files
    file_list = sorted(glob.glob('Video_level/AU_10fps/*.csv'))
    label_data = pd.read_csv('Video_level/labels_traits_minmax_norm.csv')
    label = label_data[label_name]
    f = np.array(file_list).reshape(-1,1) #All data files
    label = pd.to_numeric(label, downcast='integer') #Float to integer conversion
    label = label.to_numpy().reshape(-1,1)  
    #print(label.shape)
    file_to_work = np.concatenate((f, label),axis=1)   #File plus label in single array


    # parameters
    nKineme, seqLen, nClass = 16, seq_length, 1
    EPOCHS = 30
    BATCH_SIZE = 32
    nNeuron = 12
    chunk_size = seqLen + 1
    nAction = 17
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    #model defined for kineme
    Model_kineme = Sequential()
    Model_kineme.add(LSTM(nNeuron,activation="tanh",dropout=0.2,recurrent_dropout=0.0,input_shape=(seqLen, nKineme)))
    Model_kineme.add(Dense(units = nClass, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    Model_kineme.compile(optimizer = opt, loss = 'binary_crossentropy',metrics=['accuracy'])
    Model_kineme.summary()
    
    #AU model architecture
    Model_AU = Sequential()
    Model_AU.add(LSTM(nNeuron,activation="tanh",dropout=0.2,recurrent_dropout=0.0,input_shape=(seqLen, nAction)))
    Model_AU.add(Dense(units = nClass, activation='sigmoid'))
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
    n=1
    random_state = 42
    # print(n)
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
    for train_idx, test_idx in rkf.split(file_to_work):
        train_features, test_features = file_to_work[train_idx], file_to_work[test_idx] 
        #Calling for train-data
        #for the first file followed by a loop over all the files
        file_path = train_features[:,0].tolist()  #File list for chunk preparation
        labels = train_features[:,1]  #labels for chunk preparation

        file_name = os.path.basename(file_path[0] )  #eds_report.csv

        AU_path = 'Video_level/AU_10fps/' + file_name 
        Kine_Path = 'Video_level/Kinemes/' + file_name

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
        #convert the training file data to chunks directly
        for i in range(0,len(file_path)-1,1):
            file_name = os.path.basename(file_path[i+1] )  #eds_report.csv
            AU_path = 'Video_level/AU_10fps/' + file_name 
            Kine_Path = 'Video_level/Kinemes/' + file_name
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
#         train_labels = to_categorical(final_label)
        train_labels = final_label

        train_kinemes = ks_encoding(final[:,0:seqLen], 16)
        train_action = final[:, seqLen:]
        train_aus = train_action.reshape((train_action.shape[0], seqLen, 17))
        #fit the kineme and AU model
        Model_kineme_history = Model_kineme.fit(train_kinemes, train_labels, epochs = EPOCHS, batch_size = 32, validation_split = 0.1,callbacks=[callback])
        Model_AU_history = Model_AU.fit(train_aus, train_labels, epochs = EPOCHS, batch_size = 32, validation_split = 0.1,callbacks=[callback])
        
        
        #Process for testing in case of video analysis
        #get the test labels and convert each label to 0 or 1
        test_data = test_features[:,0].tolist()  #test data
        test_labels = test_features[:,1]  #test actual label
        test_labels = [float(i) for i in test_labels]   #String to int conversion
        test_labels = bin_labels(test_labels)
        y_pred_video = []
        #a loop over the files in the test data
        for i in range(0,len(test_data)):
            file_name = os.path.basename(test_data[i] )  
            #find the AU and kineme values for each file and create chunks according to the taken size
            AU_path = 'Video_level/AU_10fps/' + file_name 
            Kine_Path = 'Video_level/Kinemes/' + file_name
            AU2 = pd.read_csv(AU_path)
            Kin2 = pd.read_csv(Kine_Path, header=None)
            label2 = test_labels[i]
            chun_au = int(AU2.shape[0]/(chunk_size*10))
            chun_ki = int(Kin2.shape[1]/(chunk_size))
            #get equla chunks from both AU and kineme matrices
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
            #concatenate the two matrices and then reshape it as per the training data and then predict values over the two matrices
            #find the weighted prediction and convert to 0 or 1 labels and take the majority vote over each video file
            final_test = np.concatenate((kin_res2.T, au_res2.T)).T
            test_kinemes = ks_encoding(final_test[:,0:seqLen], 16)
            test_action = final_test[:, seqLen:]
            test_aus = test_action.reshape((test_action.shape[0], seqLen, 17))
            y_pred_kineme = Model_kineme.predict(test_kinemes)
            y_pred_aus = Model_AU.predict(test_aus)
            final_test_pred = (weight*y_pred_kineme) + ((1-weight)* y_pred_aus)
            y_pred = ((final_test_pred > 0.5)+0).ravel()
            y1 = majority_vote(y_pred)   #Voting for classification to change
#             y_pred = Model.predict(test_kinemes)
#             y_pred = y_pred.argmax(axis=-1)
#             y1 = majority_vote(y_pred)   #Voting for classification to change
            y_pred_video.append(y1)

        #append the values to train and test accuracy
        test_acc.append(accuracy_score(test_labels, y_pred_video))
        #predict values for the training kinemes and AUs and find their weighted prediction and finally predict the label as 0 or 1 according to the final predicted value
        y_pred_train_kineme = Model_kineme.predict(train_kinemes)
        y_pred_train_aus = Model_AU.predict(train_aus)
        final_train_pred = (weight*y_pred_train_kineme) + ((1-weight)* y_pred_train_aus)
        y_pred_train = ((final_train_pred > 0.5)+0).ravel()
        train_acc.append(accuracy_score(train_labels, y_pred_train))
        #find the weighted and macro f1 score and return all the values
        f1_w_epoch = f1_score(test_labels, y_pred_video, average='weighted')
        f1_m_epoch = f1_score(test_labels, y_pred_video, average='macro')
        fi_weighted.append(f1_w_epoch)
        fi_macro.append(f1_m_epoch)
    return np.asarray(train_acc), np.asarray(test_acc), np.asarray(fi_weighted), np.asarray(fi_macro)


# In[4]:


y_data_path = 'Chunk_level/Label_5_Overall.npy'
X_data_path = 'Chunk_level/Data_5_Overall.npy'

#create a weight matrix and define the accuracy adn f1 score lists
weight_matrix = np.arange(0.0, 1.01, 0.1)
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
i = 1
#start the loop for executing the fusion architecture using each weight
#append all the mean and std values to the created lists
for each_w in weight_matrix:
    print("*********************"+ str(i) + "*********************") 
    final_train_acc, final_test_acc, final_f1_weighted, final_f1_macro = training_lstm_video("Overall", 4, each_w)
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
Fusion_accuracies = pd.DataFrame(list(zip(weight_list, train_acc_list, train_acc_std, test_acc_list, test_acc_std,
                      
                                          f1_weighted_list, f1_weighted_std, f1_macro_list, f1_macro_std)) , columns =['Weight value(Kineme)', 
                            'Training accuracy', 'Train std', 'Testing accuracy', 'Test std', 'F1 weighted', 'F1 weighted std', 'F1 Macro', 'F1 macro std'])


Fusion_accuracies.to_csv("DataFrames_Video/Overall_5.csv", index = False)
Fusion_accuracies

