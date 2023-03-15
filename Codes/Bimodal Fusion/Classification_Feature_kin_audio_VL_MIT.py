# -*- coding: utf-8 -*-

#the imports
import glob
import os

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input, Dense, concatenate


chunk_size = 30
label_name = "Overall"
#all parameters
seqLen = chunk_size-1
nKineme, nClass = 16, 1
nAction, nAudio = 17, 23
threshold_action_units = 1.5
epochs_par, batch_size_par = 30, 32
nNeuron = 12
chunk_time = chunk_size

# onehot encoding of kineme sequence
def onehot_encoding(ks, nKineme):
    onehot_encoded = list()
    for k in ks:
        vec = [0 for _ in range(nKineme)]
        vec[k-1] = 1
        onehot_encoded.append(vec)
    return onehot_encoded
def ks_encoding(ks, nKineme):
    m, n = ks.shape # ks is a numpy ndarray
    ks = ks.tolist() #converted to list
    encoded_features = np.asarray(
        [np.asarray(onehot_encoding(ks[i], nKineme)) for i in range(m)])
    return encoded_features

#Function to create kineme data matrix for Train Data
def data_preprocess(chunk_time, input_file_kineme, label_val, num_chunk): 
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
def data_preprocess_test(chunk_time, input_file_kineme, num_chunk): 
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

#function to return the majority from a list of labels
def majority_vote(arr):
    vote_count = Counter(arr)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return np.random.randint(2)
    return top_two[0][0]
#function to convert continous labels into binary labels
def bin_labels(data_rec):             
    count_0 = 0
    count_1 = 0
    median_value = np.median(data_rec)
    print("Inside bin labels function: " + str(median_value))
    for it in range(0,len(data_rec),1):
        if data_rec[it]<=median_value :
            count_0 += 1
            data_rec[it]=0
        else:
            count_1 += 1
            data_rec[it]=1
    print(count_0, count_1)
    return data_rec
def find_median(data_rec):
    median_value = np.median(data_rec)
    print("Inside med function: " + str(median_value))
    return median_value

#function to convert continous labels into binary labels
def bin_labels_test(data_rec, train_median):             
    count_0 = 0
    count_1 = 0
    # median_value = train_median
    for it in range(0,len(data_rec),1):
        if data_rec[it]<=train_median :
            count_0 += 1
            data_rec[it]=0
        else:
            count_1 += 1
            data_rec[it]=1
    print(count_0, count_1)
    return data_rec 

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
        for j in range(0, chunk_size-1, 1): #0 1 2 0 20 10 30 20 40
            max_pool = [] 
            for i in range(5, 22, 1):
                max_value = np.max(read_data[duration:duration+20, i])
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
# function to create audio features
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
    scaled_Data = scaler.transform(Data)
    Data_final = scaled_Data
    return Data_final



#model for the feature fusion
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#left input LSTM
kineme_branch_input = Input(shape=(seqLen, nKineme), name="kineme_input")
kineme_branch_output = LSTM(nNeuron, activation = 'tanh')(kineme_branch_input)
#right input LSTM
audio_branch_input = Input(shape=(seqLen, nAudio), name='audio_input')
audio_branch_output = LSTM(nNeuron, activation = 'tanh')(audio_branch_input)
#Merging all layers
merged_layer = concatenate([kineme_branch_output, audio_branch_output], name='concatenate')
final_model_output = Dense(nClass, activation = 'sigmoid')(merged_layer)
final_model = Model(inputs=[kineme_branch_input, audio_branch_input], outputs=final_model_output, name='Final_output')
opt = keras.optimizers.Adam(learning_rate=0.01)
final_model.compile(optimizer = opt, loss = 'binary_crossentropy',metrics=['accuracy'])
final_model.summary()

#path to all the kineme files
# path for implementation on other systems
file_list = sorted(glob.glob('/content/drive/MyDrive/Data/AU_10fps/*.csv'))
label_data = pd.read_csv('/content/drive/MyDrive/Data/labels_for_MIT.csv')
label = label_data[label_name]
f = np.array(file_list).reshape(-1,1) #All data files
label = pd.to_numeric(label, downcast='integer') #Float to integer conversion
label = label.to_numpy().reshape(-1,1)  
file_to_work = np.concatenate((f, label),axis=1)   #File plus label in single array
# file_to_work = file_to_work[0:10,:]

#define lists for saving the results
test_loss=[]
test_acc=[]
train_loss = []
train_acc = []
fi_weighted=[]
fi_macro=[]
val_loss = []
val_acc = []

#count for counting the number of iterations
#file_to_work contains all files to be divided into train and test data for each of the splits of validation
count = 1
random_state = 42
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
for train_idx, test_idx in rkf.split(file_to_work):
    print("@@@@@@@@ On the iteration " + str(count) + " @@@@@@@@")
    train_features, test_features = file_to_work[train_idx], file_to_work[test_idx]
    file_path = train_features[:,0].tolist()  #File list for chunk preparation
    labels = train_features[:,1]  #labels for chunk preparation
    file_name = Path(file_path[0]).stem 
    #paths for files
    outer_directory  = '/content/drive/MyDrive/Data'
    AU_path = outer_directory + '/AU_10fps/' + file_name + '.csv'
    Kine_Path = outer_directory + '/Kineme_MIT (Given By Atanu)/' + file_name + '.csv'
    Audio_path = outer_directory + '/MIT_Data_CSV/' + file_name +'_audio.csv'
    #creation of data frames with the initial csv paths of kineme and AU and label; this is basically done because 
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
    audio_res1 = chunks_formation(Audio_path, chunk_time, nAudio,total_size)
    kin_res1 = np.array(kin_res1)
    
    for i in range(0,len(file_path)-1,1):
        file_name = Path(file_path[i+1]).stem   #eds_report.csv
        AU_path = outer_directory + '/AU_10fps/' + file_name + '.csv'
        Kine_Path = outer_directory + '/Kineme_MIT (Given By Atanu)/' + file_name + '.csv'
        Audio_path = outer_directory + '/MIT_Data_CSV/' + file_name +'_audio.csv'
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
        audio_res2 = chunks_formation(Audio_path, chunk_time, 23,total_size)
        #make array of results
        kin_res2 = np.array(kin_res2)
        #Matrix merging
        kin_res1 = np.vstack((kin_res1,kin_res2))
        audio_res1 = np.vstack((audio_res1,audio_res2))
        label1.extend(label2)
    scaler = preprocessing.StandardScaler().fit(audio_res1)
    final_audio = scaler.transform(audio_res1)
    final_label = np.array(label1)
    final_label = [float(i) for i in final_label]   #String to float conversion
    train_median = find_median(final_label)
    final_label = bin_labels(final_label)
    train_labels = np.array(final_label)
    train_kinemes = ks_encoding(kin_res1, nKineme)
    train_audio = final_audio.reshape((final_audio.shape[0], seqLen, nAudio))
    zero_bias_history = final_model.fit([train_kinemes, train_audio], train_labels, epochs = epochs_par, batch_size = batch_size_par, validation_split = 0.1,callbacks=[callback])
    

    #Process for testing
    test_data = test_features[:,0].tolist()  #test data
    test_labels = test_features[:,1]  #test actual label
    test_labels = [float(i) for i in test_labels]   #String to int conversion
    test_labels = bin_labels_test(test_labels, train_median)
    y_pred_video = []
    
    for i in range(0,len(test_data)):
        file_name = Path(test_data[i]).stem   #eds_report.csv
        AU_path = outer_directory + '/AU_10fps/' + file_name + '.csv'
        Kine_Path = outer_directory + '/Kineme_MIT (Given By Atanu)/' + file_name + '.csv'
        Audio_path = outer_directory + '/MIT_Data_CSV/' + file_name +'_audio.csv'

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
        # au_res2 = max_encoding(AU2,1.5,chunk_size,total_size)
        audio_res2 = chunks_formation_test(Audio_path, chunk_time, 23,scaler,total_size)

        #make array of results
        kin_res2 = np.array(kin_res2)

        test_kinemes = ks_encoding(kin_res2, nKineme)
        test_audio = audio_res2.reshape((audio_res2.shape[0], seqLen, nAudio))
        y_pred = final_model.predict([test_kinemes,  test_audio])
        y_pred = ((y_pred > 0.5)+0).ravel()
        y1 = majority_vote(y_pred)   #Voting for classification to change
        y_pred_video.append(y1)
    acc_val = accuracy_score(test_labels, y_pred_video)
    test_acc.append(acc_val)
    print('Test accuracy:', acc_val)   
    train_acc.append(np.array(zero_bias_history.history['accuracy']).mean()) 
    val_acc.append(np.array(zero_bias_history.history['val_accuracy']).mean()) 
    f1_w_epoch = f1_score(test_labels, y_pred_video, average='weighted')
    f1_m_epoch = f1_score(test_labels, y_pred_video, average='macro')
    fi_weighted.append(f1_w_epoch)
    fi_macro.append(f1_m_epoch)
    count = count+1
    
print("Feature fusion  Bimodal result for Kin and Audio")
print("For label " + str(label_name) + " and chunk size " + str(chunk_size))
print(str(round(np.array(train_acc).mean(),3)) + "±" +str(round(np.array(train_acc).std(),3)) + "," + str(round(np.array(test_acc).mean(),3)) + "±" +str(round(np.array(test_acc).std(),3)) + "," + str(round(np.array(fi_weighted).mean(),3)) + "±" +str(round(np.array(fi_weighted).std(),3)) + "," + str(round(np.array(fi_macro).mean(),3)) + "±" +str(round(np.array(fi_macro).std(),3)))
