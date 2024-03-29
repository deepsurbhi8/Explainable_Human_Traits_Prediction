
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

from google.colab import drive
drive.mount('/content/drive')

# onehot encoding of kineme sequence
def onehot_encoding(ks, nKineme):
    #print(ks)
    onehot_encoded = list()
    for k in ks:
        #print(k)
        vec = [0 for _ in range(nKineme)]
        vec[k-1] = 1
        onehot_encoded.append(vec)

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
      for j in range(0, chunk_size-1, 1):
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
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

  Model_AU = Sequential()
  Model_AU.add(LSTM(20,activation="tanh",dropout=0.2,recurrent_dropout=0.0,input_shape=(seqLen, 17)))
  Model_AU.add(Dense(units = 1, activation='linear'))
  opt = keras.optimizers.Adam(learning_rate=0.01)
  Model_AU.compile(optimizer = opt, loss = 'mean_absolute_error')
  Model_AU.summary()

  Model_audio = Sequential()
  Model_audio.add(LSTM(20,activation="tanh",dropout=0.2,recurrent_dropout=0.0,input_shape=(seqLen, 23)))
  Model_audio.add(Dense(units = 1, activation='linear'))
  opt = keras.optimizers.Adam(learning_rate=0.01)
  Model_audio.compile(optimizer = opt, loss = 'mean_absolute_error')
  Model_audio.summary()
  return Model_AU,  Model_audio, callback

def model_call(file_to_work, chunk_size,l_name, Model_AU, Model_audio, callback):
  seqLen = chunk_size-1
  #Lists to contain MAE and PCC
  train_mae1, test_mae1, train_PCC1, test_PCC1 =[], [], [], []
  train_mae2, test_mae2, train_PCC2, test_PCC2 =[], [], [], []
  train_mae3, test_mae3, train_PCC3, test_PCC3 =[], [], [], []


  n=0
  random_state = 42
  rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
  for train_idx, test_idx in rkf.split(file_to_work):
    print(n)
    train_features, test_features = file_to_work[train_idx], file_to_work[test_idx]
    file_path = train_features[:,0].tolist()  #File list for chunk preparation
    labels = train_features[:,1]  #labels for chunk preparation
    file_name = os.path.basename(file_path[0])  #eds_report.csv
    # os.path.split(file_name)[-1]

    #paths for files
    start_path = '/content/drive/MyDrive/Data/'
    AU_path = start_path + 'AU_10fps/' + file_name
    Kine_Path = start_path + 'Kineme_MIT (Given By Atanu)/' + file_name
    Audio_path = start_path + 'MIT_Data_CSV/' + file_name[0:-4] +'_audio.csv'

    AU1 = pd.read_csv(AU_path)
    Kin1 = pd.read_csv(Kine_Path, header=None)
    label = labels[0]

    chun_au = int(AU1.shape[0]/(chunk_size*10)) #10 is because we are taking 10fps and for every 10 value we will need one row
    chun_ki = int(Kin1.shape[1]/(chunk_size))
    if chun_au<chun_ki:
        total_size = chun_au
    else:
        total_size = chun_ki
    kin_res1,label1 = data_preprocess(chunk_size,Kin1,label,total_size)
    au_res1 = max_encoding(AU1,1.5,chunk_size,total_size)
    audio_res1 = chunks_formation(Audio_path, chunk_time, 23,total_size)

    # kin_res1 = np.array(kin_res1)
    au_res1 = np.array(au_res1)

    for i in range(0,len(file_path)-1,1):
        file_name = os.path.basename(file_path[i+1] )  #eds_report.csv

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
        # kin_res2 = np.array(kin_res2)
        au_res2 = np.array(au_res2)

        #Matrix merging
        # kin_res1 = np.vstack((kin_res1,kin_res2))
        au_res1 = np.vstack((au_res1,au_res2))
        audio_res1 = np.vstack((audio_res1,audio_res2))
        label1.extend(label2)
    # final = np.concatenate((kin_res1.T, au_res1.T)).T

    scaler = preprocessing.StandardScaler().fit(audio_res1)
    final_audio = scaler.transform(audio_res1)
    # print('Original Data')
    # print(audio_res1)
    # print('Scaled Data')
    # print(final_audio)

    final_label = np.array(label1)
    final_label = [float(i) for i in final_label]   #String to float conversion
    final_label = np.array(final_label)
    train_labels = np.around(final_label,3)

    # train_kinemes = ks_encoding(kin_res1, 16)
    train_action = au_res1
    train_aus = train_action.reshape((train_action.shape[0], seqLen, 17))
    train_audio = final_audio.reshape((final_audio.shape[0], seqLen,23))

    # zero_bias_history = final_model.fit([train_kinemes, train_aus,train_audio], train_labels, epochs = 30, batch_size = 32, validation_split = 0.1,callbacks=[callback])
    # print(train_kinemes.shape, train_aus.shape, train_audio.shape)
    # kineme_history = Model_kineme.fit(train_kinemes, train_labels, epochs = 30, batch_size = 32, validation_split=0.1,callbacks=[callback])  #Fitting the model
    # print("Kineme Model Training is Done")
    AU_history = Model_AU.fit(train_aus, train_labels, epochs = 30, batch_size = 32, validation_split=0.1,callbacks=[callback])
    print("AU Model Training is Done")
    audio_history = Model_audio.fit(train_audio, train_labels, epochs = 30, batch_size = 32, validation_split=0.1, callbacks=[callback])
    print("Audio model training is done")


    # print("Training prediction:............shape")
    # print(y_pred_train.shape)
    #Process for testing
    test_data = test_features[:,0].tolist()  #test data
    test_labels = test_features[:,1]  #test actual label
    test_labels = [float(i) for i in test_labels]   #String to int conversion
    test_labels = np.around(test_labels,3)
    y_pred_video_w1 = []
    y_pred_video_w2 = []
    y_pred_video_w3 = []


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
        # kin_res2 = data_preprocess_test(chunk_size,Kin2,total_size)
        au_res2 = max_encoding(AU2,1.5,chunk_size,total_size)
        audio_res2 = chunks_formation_test(Audio_path, chunk_time, 23,scaler,total_size)

        #make array of results
        # kin_res2 = np.array(kin_res2)
        au_res2 = np.array(au_res2)
        # final_test = np.concatenate((kin_res2.T, au_res2.T)).T

        # test_kinemes = ks_encoding(kin_res2, 16)
        test_action = au_res2
        test_aus = test_action.reshape((test_action.shape[0], seqLen, 17))
        test_audio = audio_res2.reshape((audio_res2.shape[0], seqLen,23))

        # test_pred_kineme = Model_kineme.predict(test_kinemes)
        test_pred_au = Model_AU.predict(test_aus)
        test_pred_audio = Model_audio.predict(test_audio)


        # final_test_pred1 = 0.33*test_pred_kineme +  0.33*test_pred_audio
        # y1 = np.mean(final_test_pred1)   #Voting for classification to change
        # y_pred_video_w1.append(y1)
        y_pred_video_w11.append(np.mean(0*test_pred_au +  1*test_pred_audio))
        y_pred_video_w1.append(np.mean(0.1*test_pred_au +  0.9*test_pred_audio))
        y_pred_video_w2.append(np.mean(0.2*test_pred_au +  0.8*test_pred_audio))







    y_pred_video_w1 = np.around(y_pred_video_w1,3)
    y_pred_video_w2 = np.around(y_pred_video_w2,3)
    y_pred_video_w3 = np.around(y_pred_video_w3,3)


    # train_pred_kineme = Model_kineme.predict(train_kinemes)
    train_pred_au = Model_AU.predict(train_aus)
    train_pred_audio = Model_audio.predict(train_audio)

    final_train_pred1 = 0.1*train_pred_au + 0.9*train_pred_audio
    final_train_pred2 = 0.2*train_pred_au  + 0.8*train_pred_audio
    final_train_pred3 = 0.3*train_pred_au + 0.7*train_pred_audio



    # final_train_pred = ((x*train_pred_kineme) + (y*train_pred_au) + (z*train_pred_audio)) #change this -- whether we need two weighing parameters or three
    # final_train_pred1 = final_train_pred1[:,0]
    # y_pred_train1 = np.around(final_train_pred1,3)
    y_pred_train1 = np.around(final_train_pred1[:,0],3)
    y_pred_train2 = np.around(final_train_pred2[:,0],3)
    y_pred_train3 = np.around(final_train_pred3[:,0],3)


    test_mae1.append(1-mean_absolute_error(test_labels, y_pred_video_w1)) #mean squarred test error
    test_mae2.append(1-mean_absolute_error(test_labels, y_pred_video_w2)) #mean squarred test error
    test_mae3.append(1-mean_absolute_error(test_labels, y_pred_video_w3)) #mean squarred test error


    y11 = np.array(test_labels).reshape(-1,1)

    a1 = np.corrcoef(y11.T,np.array(y_pred_video_w1).T)
    a2 = np.corrcoef(y11.T,np.array(y_pred_video_w2).T)
    a3 = np.corrcoef(y11.T,np.array(y_pred_video_w3).T)


    test_PCC1.append(a1[0][1])
    test_PCC2.append(a2[0][1])
    test_PCC3.append(a3[0][1])



    train_mae1.append(1-mean_absolute_error(train_labels, y_pred_train1)) ##mean squarred train error
    train_mae2.append(1-mean_absolute_error(train_labels, y_pred_train2)) ##mean squarred train error
    train_mae3.append(1-mean_absolute_error(train_labels, y_pred_train3)) ##mean squarred train error



    y_train = train_labels.reshape(-1,1)
    b1 = np.corrcoef(y_train.T,y_pred_train1.T)
    b2 = np.corrcoef(y_train.T,y_pred_train2.T)
    b3 = np.corrcoef(y_train.T,y_pred_train3.T)


    train_PCC1.append(b1[0][1])
    train_PCC2.append(b2[0][1])
    train_PCC3.append(b3[0][1])

    print(n)
    n = n+1


  # print(train_mae1,train_mae2,train_mae3,train_mae4,train_mae5,train_mae6,train_mae7)
  # print(test_mae1,test_mae2,test_mae3,test_mae4,test_mae5,test_mae6,test_mae7)
  # print(train_PCC1,train_PCC2,train_PCC3,train_PCC4,train_PCC5,train_PCC6,train_PCC7)
  # print(test_PCC1,test_PCC2,test_PCC3,test_PCC4,test_PCC5,test_PCC6,test_PCC7)
  print("For label {0} and chunk_time {1}".format(l_name,chunk_time))
  print("Train-accuracy Test-accuracy Train-PCC Test-PCC")
  print("{0}±{1} {2}±{3} {4}±{5} {6}±{7}".format(round(np.array(train_mae1).mean(),3),round(np.array(train_mae1).std(),2), round(np.array(test_mae1).mean(),3),round(np.array(test_mae1).std(),2),round(np.array(train_PCC1).mean(),3),round(np.array(train_PCC1).std(),2),round(np.array(test_PCC1).mean(),3),round(np.array(test_PCC1).std(),2)))
  print("{0}±{1} {2}±{3} {4}±{5} {6}±{7}".format(round(np.array(train_mae2).mean(),3),round(np.array(train_mae2).std(),2), round(np.array(test_mae2).mean(),3),round(np.array(test_mae2).std(),2),round(np.array(train_PCC2).mean(),3),round(np.array(train_PCC2).std(),2),round(np.array(test_PCC2).mean(),3),round(np.array(test_PCC2).std(),2)))
  print("{0}±{1} {2}±{3} {4}±{5} {6}±{7}".format(round(np.array(train_mae3).mean(),3),round(np.array(train_mae3).std(),2), round(np.array(test_mae3).mean(),3),round(np.array(test_mae3).std(),2),round(np.array(train_PCC3).mean(),3),round(np.array(train_PCC3).std(),2),round(np.array(test_PCC3).mean(),3),round(np.array(test_PCC3).std(),2)))

l_name = 'Overall'
chunk_size = 60
chunk_time = chunk_size
file_list = sorted(glob.glob('/content/drive/MyDrive/Data/AU_10fps/*.csv'))
label_data = pd.read_csv('/content/drive/MyDrive/Data/labels_for_MIT.csv')
label = label_data[l_name]
# label = label[0:10]
f = np.array(file_list).reshape(-1,1) #All data files
label = pd.to_numeric(label, downcast='integer') #Float to integer conversion
label = label.to_numpy().reshape(-1,1)
#print(label.shape)
file_to_work = np.concatenate((f, label),axis=1)   #File plus label in single array
# file_to_work = file_to_work[0:10,:]
print(file_to_work.shape)
Model_AU, Model_audio,callback = model_formation(chunk_size)
model_call(file_to_work, chunk_size,l_name, Model_AU, Model_audio,callback)
