# -*- coding: utf-8 -*-

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
from sklearn import preprocessing
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
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.set_visible_devices(physical_devices[0:1], 'GPU')
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

#function to convert continous labels into binary labels
def bin_labels(data_rec):             
    count_0 = 0
    count_1 = 0
    median_value = np.median(data_rec)
    print(median_value)
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

def training_lstm(label_name, chunk_size, w, z, t):
    # parameters
    seqLen = chunk_size - 1
    nKineme, nClass = 16, 1
    nAction, nAudio = 17, 23
    EPOCHS = 30
    BATCH_SIZE = 32
    nNeuron = 12

    #load dataset and labels
    #dataset path
    outer_directory  = '/content/drive/MyDrive/Research_Related/Ongoing_Research/Kineme_Project/Kineme_Files/FICS/Data_files'
    y_train_path = outer_directory +  '/Label_chunk_' + str(chunk_size) + '_train_' + label_name + '.npy'
    X_train_path_audio = outer_directory +  '/Audio_chunk_' + str(chunk_size) + '_train.npy'
    X_train_path_au = outer_directory +  '/AU_chunk_' + str(chunk_size) + '_train.npy'
    X_train_path_kineme = outer_directory +  '/Kineme_chunk_' + str(chunk_size) + '_train.npy'
    y_test_path = outer_directory +  '/Label_chunk_' + str(chunk_size) + '_test_' + label_name + '.npy'
    X_test_path_audio = outer_directory +  '/Audio_chunk_' + str(chunk_size) + '_test.npy'
    X_test_path_au = outer_directory +  '/AU_chunk_' + str(chunk_size) + '_test.npy'
    X_test_path_kineme = outer_directory +  '/Kineme_chunk_' + str(chunk_size) + '_test.npy'
    y_val_path = outer_directory +  '/Label_chunk_' + str(chunk_size) + '_val_' + label_name + '.npy'
    X_val_path_audio = outer_directory +  '/Audio_chunk_' + str(chunk_size) + '_val.npy'
    X_val_path_au = outer_directory +  '/AU_chunk_' + str(chunk_size) + '_val.npy'
    X_val_path_kineme = outer_directory +  '/Kineme_chunk_' + str(chunk_size) + '_val.npy'
    
    
    #check if we need to change categorisation separately for each of train test and val??
    #check with taking the median in both the cases
    y_train = np.load(y_train_path)
    print(y_train.shape)
    y_train = y_train.astype(np.float)
    y_train = bin_labels(y_train)
    train_mat_audio = np.load(X_train_path_audio)
    train_mat_au = np.load(X_train_path_au)
    train_mat_kineme = np.load(X_train_path_kineme)
    y_test = np.load(y_test_path)
    y_test = y_test.astype(np.float)
    y_test = bin_labels(y_test)
    test_mat_audio = np.load(X_test_path_audio)
    test_mat_au = np.load(X_test_path_au)
    test_mat_kineme = np.load(X_test_path_kineme)
    y_val = np.load(y_val_path)
    y_val = y_val.astype(np.float)
    y_val = bin_labels(y_val)
    val_mat_audio = np.load(X_val_path_audio)
    val_mat_au = np.load(X_val_path_au)
    val_mat_kineme = np.load(X_val_path_kineme)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    Model_kineme = Sequential()
    Model_kineme.add(LSTM(nNeuron,activation="tanh",dropout=0.1,input_shape=(seqLen, nKineme)))
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
    #Audio model architecture
    Model_audio = Sequential()
    Model_audio.add(LSTM(nNeuron,activation="tanh",dropout=0.1,input_shape=(seqLen, nAudio)))
    Model_audio.add(Dense(units = nClass,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    Model_audio.compile(optimizer = opt, loss = 'binary_crossentropy',metrics=['accuracy'])
    Model_audio.summary()

    #passing the features and labels data to their respective variables as train, test or validation
    train_features,  train_labels = train_mat_audio, y_train
    test_features,  test_labels = test_mat_audio, y_test
    val_features,  val_labels = val_mat_audio, y_val
    
    #extract the kinemes and convert each value to the encoding using the ks_encoding method
    #do the same for train, test and validation data
    train_kinemes = ks_encoding(train_mat_kineme, nKineme)
    test_kinemes = ks_encoding(test_mat_kineme, nKineme)
    val_kinemes = ks_encoding(val_mat_kineme, nKineme)
    
    #reshape the action unit to a 3d matrix, similar to that of kinemes
    train_aus = train_action.reshape((train_mat_au.shape[0], seqLen, nAction))
    test_aus = test_action.reshape((test_mat_au.shape[0], seqLen, nAction))
    val_aus = val_action.reshape((val_mat_au.shape[0], seqLen, nAction))
    
    train_audio = train_features.reshape((train_features.shape[0], seqLen, nAudio))
    test_audio = test_features.reshape((test_features.shape[0], seqLen, nAudio))
    val_audio = val_features.reshape((val_features.shape[0], seqLen, nAudio))



    test_loss=[]
    test_acc=[]
    train_loss = []
    train_acc = []
    fi_weighted=[]
    fi_macro=[]
    val_loss = []
    val_acc = []

    kineme_history = Model_kineme.fit(train_kinemes, train_labels, epochs = EPOCHS, batch_size = 32, validation_data=(val_kinemes, val_labels),callbacks=[callback])  #Fitting the model 
    print("Kineme Model Training is Done")
    AU_history = Model_AU.fit(train_aus, train_labels, epochs = EPOCHS, batch_size = 32, validation_data=(val_aus, val_labels),callbacks=[callback])
    print("AU Model Training is Done")
    audio_history = Model_audio.fit(train_audio, train_labels, epochs = EPOCHS, batch_size = 32, validation_data=(val_audio, val_labels), callbacks=[callback])
    print("Audio model training is done")    
        
    train_pred_kineme = Model_kineme.predict(train_kinemes)
    train_pred_au = Model_AU.predict(train_aus)
    train_pred_audio = Model_audio.predict(train_audio) 
    final_train_pred = w*train_pred_kineme + z*train_pred_au + t*train_pred_audio 

    y_pred_train = ((final_train_pred > 0.5)+0).ravel()

    test_pred_kineme = Model_kineme.predict(test_kinemes)
    test_pred_au = Model_AU.predict(test_aus)
    test_pred_audio = Model_audio.predict(test_audio)
    final_test_pred = w*test_pred_kineme + z*test_pred_au + t*test_pred_audio
    y_pred_test = ((final_test_pred > 0.5)+0).ravel()
    train_acc.append(accuracy_score(train_labels, y_pred_train))
    test_acc.append(accuracy_score(test_labels, y_pred_test))
    f1_w_epoch = f1_score(test_labels, y_pred_test, average='weighted')
    f1_m_epoch = f1_score(test_labels, y_pred_test, average='macro')
    fi_weighted.append(f1_w_epoch)
    fi_macro.append(f1_m_epoch)
    return np.asarray(train_acc), np.asarray(test_acc), np.asarray(fi_weighted), np.asarray(fi_macro)

# Parameters and lists
chunk_size = 7
label = "O"
train_acc_list, test_acc_list, f1_macro_list, f1_weighted_list = list(), list(), list(), list()
train_acc_std, test_acc_std, f1_weighted_std, f1_macro_std = list(), list(), list(), list()
weight_list = list()
i = 1
# Function calling
for a, b, c in zip(each_w, each_z, each_t):
    print("*********************"+ str(i) + "*********************")  
    final_train_acc, final_test_acc, final_f1_weighted, final_f1_macro = training_lstm(label, chunk_size, a, b, c)
    weight_list.append([a,b,c])
    train_acc_list.append(final_train_acc.mean())
    train_acc_std.append(final_train_acc.std())
    test_acc_list.append(final_test_acc.mean())
    test_acc_std.append(final_test_acc.std())
    f1_weighted_list.append(final_f1_weighted.mean())
    f1_weighted_std.append(final_f1_weighted.std())
    f1_macro_list.append(final_f1_macro.mean())
    f1_macro_std.append(final_f1_macro.std())
    i += 1

Fusion_accuraciesEX_1 = pd.DataFrame(list(zip(weight_list, train_acc_list, train_acc_std, test_acc_list, test_acc_std,
                           f1_weighted_list, f1_weighted_std, f1_macro_list, f1_macro_std)) , columns =['Weight value', 
                            'Training accuracy', 'Train std', 'Testing accuracy', 'Test std', 'F1 weighted', 'F1 weighted std', 'F1 Macro', 'F1 macro std'])

path_to_save = './../../Results/local_audio_features/Feature_result_' + label + '.csv'
Fusion_accuraciesEX_1.to_csv(path_to_save, index = False)