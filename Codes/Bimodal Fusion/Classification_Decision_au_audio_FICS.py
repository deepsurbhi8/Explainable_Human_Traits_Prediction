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

def training_lstm_15(label_name, chunk_size, w):
    # parameters
    seqLen = chunk_size - 1
    nKineme, nClass = 16, 1
    nAction, nAudio = 17, 23
    EPOCHS = 30
    BATCH_SIZE = 32
    nNeuron = 12

    #load dataset and labels
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
    
    
    y_train = np.load(y_train_path)
    print(y_train.shape)
    y_train = y_train[:, 1].astype(float)
    y_train = bin_labels(y_train)
    train_mat_audio = np.load(X_train_path_audio)
    train_mat_au = np.load(X_train_path_au)
    train_mat_kineme = np.load(X_train_path_kineme)
    y_test = np.load(y_test_path)
    y_test = y_test[:, 1].astype(float)
    y_test = bin_labels(y_test)
    test_mat_audio = np.load(X_test_path_audio)
    test_mat_au = np.load(X_test_path_au)
    test_mat_kineme = np.load(X_test_path_kineme)
    y_val = np.load(y_val_path)
    y_val = y_val[:, 1].astype(float)
    y_val = bin_labels(y_val)
    val_mat_audio = np.load(X_val_path_audio)
    val_mat_au = np.load(X_val_path_au)
    val_mat_kineme = np.load(X_val_path_kineme)
    
    
    



    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout  
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

    AU_history = Model_AU.fit(train_aus, train_labels, epochs = EPOCHS, batch_size = 32, validation_data=(val_aus, val_labels),callbacks=[callback])
    print("AU Model Training is Done")
    audio_history = Model_audio.fit(train_audio, train_labels, epochs = EPOCHS, batch_size = 32, validation_data=(val_audio, val_labels), callbacks=[callback])
    print("Audio model training is done")    
        
    train_pred_au = Model_AU.predict(train_aus)
    train_pred_audio = Model_audio.predict(train_audio) 

    y_pred_train = (((w*train_pred_au + (1-w)*train_pred_audio) > 0.5)+0).ravel()
    test_pred_au = Model_AU.predict(test_aus)
    test_pred_audio = Model_audio.predict(test_audio)
    y_pred_test = (((w*test_pred_au + (1-w)*test_pred_audio) > 0.5)+0).ravel()
    train_acc = accuracy_score(train_labels, y_pred_train)
    test_acc = accuracy_score(test_labels, y_pred_test)
    f1_w_epoch = f1_score(test_labels, y_pred_test, average='weighted')
    f1_m_epoch = f1_score(test_labels, y_pred_test, average='macro')

   
    return train_acc, test_acc, f1_w_epoch, f1_m_epoch

# Parameters
chunk_time = 15
seqLen = 14
label ='O'
train_acc = []
test_acc = []
fi_weighted = []
f1_macro = []
for w in weight_list:
    print(w)
    train_acc1, test_acc1, f1_weight, f1_mac = training_lstm_15(label, chunk_time, w)
    train_acc.append(round(train_acc1,3))
    test_acc.append(round(test_acc1,3))
    fi_weighted.append(round(f1_weight,3))
    f1_macro.append(round(f1_mac,3))
    
#create a dataframe to store all the lists and save the dataframe    
final_frame = pd.DataFrame(list(zip(weight_list, train_acc, test_acc, fi_weighted, f1_macro)), 
                           columns = ['Weights', 'Train accuracy', 'Test accuracy', 'F1-weighted', 'F1-macro'])

path = '/content/drive/MyDrive/Data/C_FICS_DF_AU_Aud_weight_' + str(chunk_time) + str(label) + '.csv'
final_frame.to_csv(path, index=False)
