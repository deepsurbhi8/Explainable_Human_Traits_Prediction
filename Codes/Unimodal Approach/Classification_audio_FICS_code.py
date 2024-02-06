# -*- coding: utf-8 -*-


#the imports
import pandas as pd
import numpy as np

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

from google.colab import drive
drive.mount('/content/drive')

outer_directory  = '/content/drive/MyDrive/Kineme_Project/Kineme_Files/'
y_train_path = outer_directory + 'FICS/Data_files/Label_chunk_' + str(15) + '_train_O.npy'
print(y_train_path)
y_train = np.load(y_train_path)
print(y_train.shape)
y_train = y_train.astype(np.float)

def implement_lstm(chunk_size, label_name):

    #fn to converting continous labels into binary labels
    def bin_labels(data_rec):
        count_0, count_1 = 0, 0
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
        return data_rec       #returning binary labeled data

    #LSTM parameters
    nNeuron = 12
    nClass = 1
    seqLen, nValues = chunk_size-1, 23
    EPOCHS = 30
    BATCH_SIZE = 32


    #load dataset and labels
    #dataset path
    outer_directory  = '/content/drive/MyDrive/Kineme_Project/Kineme_Files/FICS/Data_files'
    y_train_path = outer_directory +  '/Label_chunk_' + str(chunk_size) + '_train_' + label_name + '.npy'
    X_train_path = outer_directory + '/Audio_chunk_' + str(chunk_size) + '_train.npy'
    y_test_path = outer_directory + '/Label_chunk_' + str(chunk_size) + '_test_' + label_name + '.npy'
    X_test_path = outer_directory + '/Audio_chunk_' + str(chunk_size) + '_test.npy'
    y_val_path = outer_directory + '/Label_chunk_' + str(chunk_size) + '_val_' + label_name + '.npy'
    X_val_path = outer_directory + '/Audio_chunk_' + str(chunk_size) + '_val.npy'

    # reading the data nad labels for train, test and val data
    y_train = np.load(y_train_path)
    print(y_train.shape)
    y_train = y_train[:, 1].astype(np.float)
    y_train = bin_labels(y_train)
    train_mat = np.load(X_train_path)
    y_test = np.load(y_test_path)
    y_test = y_test[:, 1].astype(np.float)
    y_test = bin_labels(y_test)
    test_mat = np.load(X_test_path)
    y_val = np.load(y_val_path)
    y_val = y_val[:, 1].astype(np.float)
    y_val = bin_labels(y_val)
    val_mat = np.load(X_val_path)

    #scaling the data
    scaler = preprocessing.StandardScaler().fit(train_mat)
    train_matrix = scaler.transform(train_mat)
    train_matrix_scaled = pd.DataFrame(train_matrix)
    #scaling test data
    test_matrix_scaled = scaler.transform(test_mat)
    val_matrix_scaled = scaler.transform(val_mat)
    # scaling validation data
    train_matrix_scaled = np.array(train_matrix_scaled)
    test_matrix_scaled = np.array(test_matrix_scaled)
    val_matrix_scaled = np.array(val_matrix_scaled)

    #check shapes
    # print(train_matrix_scaled.shape)
    print(y_train.shape)


    #LSTM model architecture
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    Model = Sequential()
    Model.add(LSTM(nNeuron, activation="tanh", dropout=0.1, recurrent_dropout=0.0, input_shape=(seqLen, nValues)))
    Model.add(Dense(units=nClass, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    Model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    Model.summary()


    #lists to add the results
    test_acc = []
    train_acc = []
    fi_weighted = []
    fi_macro = []
    val_acc = []

    # creating features and calling the function
    train_features,  train_labels = train_mat, y_train
    test_features,  test_labels = test_mat, y_test
    val_features,  val_labels = val_mat, y_val

    # reshaping the features
    train_features = train_features.reshape((train_features.shape[0], seqLen, nValues))
    test_features = test_features.reshape((test_features.shape[0], seqLen, nValues))
    val_features = val_features.reshape((val_features.shape[0], seqLen, nValues))
    zero_bias_history = Model.fit(train_features, train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data=(val_features, val_labels), callbacks=[callback])
    score = Model.evaluate(test_features, test_labels, verbose=0)

    # Accuracy and results
    print('Test accuracy:', score[1])
    test_acc.append(score[1])
    train_acc.append(np.array(zero_bias_history.history['accuracy']).mean())
    val_acc.append(np.array(zero_bias_history.history['val_accuracy']).mean())
    y_testpred = ((Model.predict(test_features) > 0.5)+0).ravel()
    f1_w_epoch = f1_score(test_labels, y_testpred, average='weighted')
    f1_m_epoch = f1_score(test_labels, y_testpred, average='macro')
    fi_weighted.append(f1_w_epoch)
    fi_macro.append(f1_m_epoch)

    print("Train Accuracy    Test Accuracy    F1 Score- Weighted    F1 Score- Macro")
    print(str(round(np.array(train_acc).mean(),3)) + " ± " +str(round(np.array(train_acc).std(),3)) + "," + str(round(np.array(test_acc).mean(),3)) + " ± " +str(round(np.array(test_acc).std(),3)) + "," + str(round(np.array(fi_weighted).mean(),3)) + " ± " +str(round(np.array(fi_weighted).std(),3)) + "," + str(round(np.array(fi_macro).mean(),3)) + " ± " +str(round(np.array(fi_macro).std(),3)))
    return train_acc, test_acc, fi_weighted, fi_macro

# Function calling
seq_len = 15

label_name = "O"
train_acc_O15, test_acc_O15, f1_weight_O15, f1_macro_O15 = implement_lstm(seq_len, label_name)
print(str(round(np.array(train_acc_O15).mean(),3)) + " ± " +str(round(np.array(train_acc_O15).std(),3)) + "," + str(round(np.array(test_acc_O15).mean(),3)) + " ± " +str(round(np.array(test_acc_O15).std(),3)) + "," + str(round(np.array(f1_weight_O15).mean(),3)) + " ± " +str(round(np.array(f1_weight_O15).std(),3)) + "," + str(round(np.array(f1_macro_O15).mean(),3)) + " ± " +str(round(np.array(f1_macro_O15).std(),3)))


label_name = "C"
train_acc_C15, test_acc_C15, f1_weight_C15, f1_macro_C15 = implement_lstm(seq_len, label_name)
print(str(round(np.array(train_acc_C15).mean(),3)) + " ± " +str(round(np.array(train_acc_C15).std(),3)) + "," + str(round(np.array(test_acc_C15).mean(),3)) + " ± " +str(round(np.array(test_acc_C15).std(),3)) + "," + str(round(np.array(f1_weight_C15).mean(),3)) + " ± " +str(round(np.array(f1_weight_C15).std(),3)) + "," + str(round(np.array(f1_macro_C15).mean(),3)) + " ± " +str(round(np.array(f1_macro_C15).std(),3)))


label_name = "E"
train_acc_E15, test_acc_E15, f1_weight_E15, f1_macro_E15 = implement_lstm(seq_len, label_name)
print(str(round(np.array(train_acc_E15).mean(),3)) + " ± " +str(round(np.array(train_acc_E15).std(),3)) + "," + str(round(np.array(test_acc_E15).mean(),3)) + " ± " +str(round(np.array(test_acc_E15).std(),3)) + "," + str(round(np.array(f1_weight_E15).mean(),3)) + " ± " +str(round(np.array(f1_weight_E15).std(),3)) + "," + str(round(np.array(f1_macro_E15).mean(),3)) + " ± " +str(round(np.array(f1_macro_E15).std(),3)))


label_name = "A"
train_acc_A15, test_acc_A15, f1_weight_A15, f1_macro_A15 = implement_lstm(seq_len, label_name)
print(str(round(np.array(train_acc_A15).mean(),3)) + " ± " +str(round(np.array(train_acc_A15).std(),3)) + "," + str(round(np.array(test_acc_A15).mean(),3)) + " ± " +str(round(np.array(test_acc_A15).std(),3)) + "," + str(round(np.array(f1_weight_A15).mean(),3)) + " ± " +str(round(np.array(f1_weight_A15).std(),3)) + "," + str(round(np.array(f1_macro_A15).mean(),3)) + " ± " +str(round(np.array(f1_macro_A15).std(),3)))


label_name = "N"
train_acc_N15, test_acc_N15, f1_weight_N15, f1_macro_N15 = implement_lstm(seq_len, label_name)
print(str(round(np.array(train_acc_N15).mean(),3)) + " ± " +str(round(np.array(train_acc_N15).std(),3)) + "," + str(round(np.array(test_acc_N15).mean(),3)) + " ± " +str(round(np.array(test_acc_N15).std(),3)) + "," + str(round(np.array(f1_weight_N15).mean(),3)) + " ± " +str(round(np.array(f1_weight_N15).std(),3)) + "," + str(round(np.array(f1_macro_N15).mean(),3)) + " ± " +str(round(np.array(f1_macro_N15).std(),3)))
