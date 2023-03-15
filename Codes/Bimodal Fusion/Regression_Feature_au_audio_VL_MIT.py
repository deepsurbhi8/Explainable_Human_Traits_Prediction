# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob
import os
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
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input, Dense, concatenate
# from sklearn.linear_model import Ridge

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

#Function to create kineme data matrix for Train Data
def data_preprocess(chunk_time, input_file_kineme, label_val,num_chunk): 
    data_list = []
    total_num_chunks = 0
    label_file_index = 0
    final_label = [] 
    final_data_file = input_file_kineme.T     
    for num in range(0, num_chunk):
        data_chunk = final_data_file[num*(chunk_time-1):((num+1)*(chunk_time-1))]
        data_chunk = data_chunk.to_numpy()
        data_list.append(data_chunk.flatten())
    one_list = label_val.repeat(num_chunk)
    final_label.extend(one_list)                 
    return  data_list, final_label

#Function to create kineme data matrix for Test Data
def data_preprocess_test(chunk_time, input_file_kineme,num_chunk): 
    data_list = []
    total_num_chunks = 0
    label_file_index = 0
    final_label = []

    final_data_file = input_file_kineme.T 
    for num in range(0, num_chunk):
        data_chunk = final_data_file[num*(chunk_time-1):((num+1)*(chunk_time-1))]
        data_chunk = data_chunk.to_numpy()
        data_list.append(data_chunk.flatten())              
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
         duration = duration + overlap
         vector.append(max_pool)
      vec_flat = np.array(vector).flatten()
      complete_vec.append(vec_flat)
  return complete_vec

def chunks_formation(f, chunk_time, size_of_feature_set,num_chunks):
    data_list = []
    csv_file   = pd.read_csv(f, header=None)
    i = 0          
    entire_file_mat = []     
    while i < csv_file.shape[1]:
      new_frame = csv_file.loc[:, i:i+87]
      i = i + 44
      avg_value = new_frame.mean(axis=1)
      file_mat = pd.concat([avg_value], axis=1, ignore_index=True)
      file_mat = file_mat.to_numpy().flatten()
      file_mat = pd.DataFrame(file_mat)
      entire_file_mat = np.concatenate((entire_file_mat, file_mat), axis=None)
      new_array = entire_file_mat
      value_to_be_minus = new_array.shape[0]-(2*size_of_feature_set)
      new_array = new_array[0:value_to_be_minus] #To handle extra 2 sec data
    for num in range(0, num_chunks):
        data_chunk = new_array[num*(chunk_time-1)*size_of_feature_set:((num+1)*(chunk_time-1)*size_of_feature_set)]
        data_list.append(data_chunk)  
    data_array = np.array(data_list) 
    my_df = pd.DataFrame(data_array)
    Data = my_df.fillna(method='ffill')
    Data = Data.fillna(method='bfill')
    Data = np.asarray(Data)
    return Data

def chunks_formation_test(file_path, chunk_time, size_of_feature_set,scaler,num_chunks):
  f = file_path
  data_list = []
  csv_file   = pd.read_csv(f, header=None)
  # csv_file = csv_file.fillna(method='ffill')
  i = 0          
  entire_file_mat = []
  while i < csv_file.shape[1]:
    new_frame = csv_file.loc[:, i:i+87]
    i = i + 44
    avg_value = new_frame.mean(axis=1)
    file_mat = pd.concat([avg_value], axis=1, ignore_index=True)
    file_mat = file_mat.to_numpy().flatten()
    file_mat = pd.DataFrame(file_mat)
    entire_file_mat = np.concatenate((entire_file_mat, file_mat), axis=None)
    new_array = entire_file_mat
    value_to_be_minus = new_array.shape[0]-(2*size_of_feature_set)
    new_array = new_array[0:value_to_be_minus] #To handle extra 2 sec data
  for num in range(0, num_chunks):
      data_chunk = entire_file_mat[num*(chunk_time-1)*size_of_feature_set:((num+1)*(chunk_time-1)*size_of_feature_set)]
      data_list.append(data_chunk)
  data_array = np.asarray(data_list)
  data_df= pd.DataFrame(data_array)
  Data = data_df.fillna(method='ffill')
  Data = Data.fillna(method='bfill')
  Data = np.asarray(Data)
  # Data_final = Data
  scaled_Data = scaler.transform(Data)
  Data_final = scaled_Data
  return Data_final

def model_formation(chunk_size):
  seqLen = chunk_size-1
  #LSTM Model
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
  #Middle Input LSTM
  middle_branch_input = Input(shape=(seqLen,17), name='middle_input')
  middle_branch_output = LSTM(20, activation='tanh')(middle_branch_input)
  #Right Input LSTM
  right_branch_input = Input(shape=(seqLen,23), name='Right_input')
  right_branch_output = LSTM(20, activation='tanh')(right_branch_input)
  #Merged Layer
  merged = concatenate([middle_branch_output, right_branch_output], name='Concatenate')
  final_model_output = Dense(1, activation='linear')(merged)
  final_model = Model(inputs=[middle_branch_input,right_branch_input], outputs=final_model_output,name='Final_output')
  opt = keras.optimizers.Adam(learning_rate=0.01) #Optimizer
  final_model.compile(optimizer = opt, loss = 'mean_absolute_error')
  final_model.summary()
  return final_model,callback

def model_call(file_to_work, chunk_size,l_name, final_model,callback):
  seqLen = chunk_size-1
  #Lists to contain MAE and PCC
  train_mae =[]
  test_mae =[]
  train_PCC =[]
  test_PCC =[]
  n=1
  random_state = 42
  rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
  for train_idx, test_idx in rkf.split(file_to_work):
    print(n)
    train_features, test_features = file_to_work[train_idx], file_to_work[test_idx]
    file_path = train_features[:,0].tolist()  #File list for chunk preparation
    labels = train_features[:,1]  #labels for chunk preparation
    file_name = os.path.basename(file_path[0])  #eds_report.csv
    
    start_path = '/content/drive/MyDrive/Data/'
    #paths for files
    AU_path = start_path + 'AU_10fps/' + file_name 
    Kine_Path = start_path + 'Kineme_MIT (Given By Atanu)/' + file_name
    Audio_path = start_path + 'MIT_Data_CSV/' + file_name[0:-4] +'_audio.csv'
    
    AU1 = pd.read_csv(AU_path)
    Kin1 = pd.read_csv(Kine_Path, header=None)
    label = labels[0]
    
    chun_au = int(AU1.shape[0]/(chunk_size*10)) 
    chun_ki = int(Kin1.shape[1]/(chunk_size))
    if chun_au<chun_ki:
        total_size = chun_au
    else:
        total_size = chun_ki
    kin_res1,label1 = data_preprocess(chunk_size,Kin1,label,total_size)
    au_res1 = max_encoding(AU1,1.5,chunk_size,total_size)   
    audio_res1 = chunks_formation(Audio_path, chunk_time, 23,total_size)
   
    kin_res1 = np.array(kin_res1)
    au_res1 = np.array(au_res1)
    
    for i in range(0,len(file_path)-1,1):
        file_name = os.path.basename(file_path[i+1] )  
        start_path = '/content/drive/MyDrive/Data/'
        AU_path = start_path + 'AU_10fps/' + file_name 
        Kine_Path = start_path + 'Kineme_MIT (Given By Atanu)/' + file_name
        Audio_path = start_path + 'MIT_Data_CSV/' + file_name[0:-4] +'_audio.csv'
        

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
        audio_res2 = chunks_formation(Audio_path, chunk_time, 23,total_size)
        
        #make array of results
        au_res2 = np.array(au_res2)
                                     
        #Matrix merging
        au_res1 = np.vstack((au_res1,au_res2))
        audio_res1 = np.vstack((audio_res1,audio_res2))
        label1.extend(label2)

    scaler = preprocessing.StandardScaler().fit(audio_res1)
    final_audio = scaler.transform(audio_res1)

    final_label = np.array(label1)
    final_label = [float(i) for i in final_label]   
    final_label = np.array(final_label)
    train_labels = np.around(final_label,3)
   
    train_action = au_res1
    train_aus = train_action.reshape((train_action.shape[0], seqLen, 17))
    train_audio = final_audio.reshape((final_audio.shape[0], seqLen,23))
    
    zero_bias_history = final_model.fit([train_aus, train_audio], train_labels, epochs = 30, batch_size = 32, validation_split = 0.1,callbacks=[callback])
    # print(train_kinemes.shape, train_aus.shape, train_audio.shape)
    
    #Process for testing
    test_data = test_features[:,0].tolist()  #test data
    test_labels = test_features[:,1]  #test actual label
    test_labels = [float(i) for i in test_labels]   #String to int conversion
    test_labels = np.around(test_labels,3)
    y_pred_video = []
    
    for i in range(0,len(test_data)):
        file_name = os.path.basename(test_data[i] )  #eds_report.csv
        start_path = '/content/drive/MyDrive/Data/'
        AU_path = start_path + 'AU_10fps/' + file_name 
        Kine_Path = start_path + 'Kineme_MIT (Given By Atanu)/' + file_name
        Audio_path = start_path + 'MIT_Data_CSV/' + file_name[0:-4] +'_audio.csv'       
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
        au_res2 = max_encoding(AU2,1.5,chunk_size,total_size)
        audio_res2 = chunks_formation_test(Audio_path, chunk_time, 23,scaler,total_size)
  
        #make array of results
        au_res2 = np.array(au_res2)
        test_action = au_res2
        test_aus = test_action.reshape((test_action.shape[0], seqLen, 17))
        test_audio = audio_res2.reshape((audio_res2.shape[0], seqLen,23))
        
        # print(test_kinemes.shape, test_aus.shape, test_audio.shape)
        
        y_pred = final_model.predict([test_aus, test_audio])
        y1 = np.mean(y_pred)   #Voting for classification to change
        y_pred_video.append(y1)
        
    y_pred_video = np.around(y_pred_video,3)
    # print("test labels and predicted labels")
    # print(test_labels, y_pred_video)
    test_mae.append(1-mean_absolute_error(test_labels, y_pred_video)) #mean squarred test error            
    y = np.array(test_labels).reshape(-1,1)
    a = np.corrcoef(y.T,np.array(y_pred_video).T)
    test_PCC.append(a[0][1])
    # print(test_PCC)
    y_pred_train = final_model.predict([train_aus, train_audio])
    y_pred_train = np.around(y_pred_train,3)

    # print("train labels and predicted labels")
    # print(train_labels, y_pred_train)
    train_mae.append(1-mean_absolute_error(train_labels, y_pred_train)) ##mean squarred train error
    # print(train_mae)
    y_train = train_labels.reshape(-1,1)
    b = np.corrcoef(y_train.T,y_pred_train.T)
    train_PCC.append(b[0][1])
    print(n)
    n = n+1
    # n = n+1
  print("For label {0} and chunk_time {1}".format(l_name,chunk_time))
  print("Train-accuracy Test-accuracy Train-PCC Test-PCC")
  print("{0}±{1} {2}±{3} {4}±{5} {6}±{7}".format(round(np.array(train_mae).mean(),3),round(np.array(train_mae).std(),2), round(np.array(test_mae).mean(),3),round(np.array(test_mae).std(),2),round(np.array(train_PCC).mean(),3),round(np.array(train_PCC).std(),2),round(np.array(test_PCC).mean(),3),round(np.array(test_PCC).std(),2)))
  
  value_save = []
  value_save.append(round(np.array(train_mae).mean(),3))
  value_save.append(round(np.array(train_mae).std(),3))
  value_save.append(round(np.array(test_mae).mean(),3))
  value_save.append(round(np.array(test_mae).std(),3))
  value_save.append(round(np.array(train_PCC).mean(),3))
  value_save.append(round(np.array(train_PCC).std(),3))
  value_save.append(round(np.array(test_PCC).mean(),3))
  value_save.append(round(np.array(test_PCC).std(),3))

  start_path = '/content/drive/MyDrive/Data/'
  path_save = start_path + 'Bimodal_fusion_Au_Ad_'  + str(chunk_size) + '_' + l_name + '.csv'
  pd.DataFrame(value_save).to_csv(path_save)

l_name = 'Friendly'
chunk_size = 15
chunk_time = chunk_size
file_list = sorted(glob.glob('/content/drive/MyDrive/Data/AU_10fps/*.csv'))
label_data = pd.read_csv('/content/drive/MyDrive/Data/labels_for_MIT.csv')
label = label_data[l_name]
f = np.array(file_list).reshape(-1,1) #All data files
label = pd.to_numeric(label, downcast='integer') #Float to integer conversion
label = label.to_numpy().reshape(-1,1)  
file_to_work = np.concatenate((f, label),axis=1)   #File plus label in single array
print(file_to_work.shape)
model,callback = model_formation(chunk_size)
model_call(file_to_work, chunk_size,l_name, model,callback)
