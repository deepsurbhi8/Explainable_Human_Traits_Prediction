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

#fn to converting continous labels into binary labels
def bin_labels(y1):
    for i in range(0,len(y1),1):
        if y1[i]<y1.median():
            y1[i]=0
        else:
            y1[i]=1
    return y1      #returning binary labeled data



#load dataset and labels
#dataset path
mit_features_path = '/content/drive/MyDrive/Kineme_extension/Data_local_features/MIT/Audio_Local_Chunk_wise/MIT_Local_2min.csv'
#labels path
mit_labels_path = '/content/drive/MyDrive/MIT Dataset/Openface_extracted_files/labels for OCEAN traits_minmax_norm.csv'

#read the files and convert into the required format by handling null values
mit_features = pd.read_csv(mit_features_path, header = None)
mit_features = mit_features.fillna(method = "ffill")
mit_features = mit_features.T
# scaling entire data
scaler = preprocessing.StandardScaler()
d = scaler.fit_transform(mit_features)
mit_features_scaled = pd.DataFrame(d)

#Converting the features into an np array
mit_feature_arr = np.array(mit_features_scaled)
mit_feature_arr = mit_feature_arr[:, 0:]

#reads the file along with labels
mit_labels = pd.read_csv(mit_labels_path)
#Converting the labels to binary labels
# print(mit_labels['Overall'])
mit_labels['Overall'] = bin_labels(mit_labels['Overall'] )
mit_labels['RecommendHiring'] = bin_labels(mit_labels['RecommendHiring'] )
mit_labels['Excited'] = bin_labels(mit_labels['Excited'] )
mit_labels['EyeContact'] = bin_labels(mit_labels['EyeContact'] )
mit_labels['Friendly'] = bin_labels(mit_labels['Friendly'] )
# print(mit_labels['Overall'])
# print(mit_labels)
#Converting test labels to an array and taking all test labels as type float
mit_labels = np.array(mit_labels)
mit_labels = mit_labels[:, 2:]
mit_labels = mit_labels.astype('float')

#check shapes
print(mit_feature_arr.shape)
print(mit_labels.shape)

#LSTM parameters
nNeuron = 32
nClass = 1
seqLen, nValues = 119, 23

#LSTM model formation
#parameters: nclass = 1, activation=tanh, dropout-0.2, nNeurons=32, learning rate=0.01, loss=binary_crossentropy, epochs=30
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
Model = Sequential()
Model.add(LSTM(nNeuron, activation="tanh", dropout=0.2, recurrent_dropout=0.0, input_shape=(seqLen, nValues)))
Model.add(Dense(units=nClass, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.01)
Model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
Model.summary()


test_acc = []
train_acc = []
f1_weight = []
f1_macro = []
val_acc = []
random_s = 42
randnums= np.random.randint(1,101,138)
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_s)
for train_idx, test_idx in rkf.split(mit_feature_arr):
  all_features = mit_feature_arr.reshape((mit_feature_arr.shape[0], seqLen, nValues))
  # Considering overall trait for the MIT dataset
  final_mit_labels = mit_labels[:, 0]
  # final_mit_labels = to_categorical(final_mit_labels)
  x_train, x_test, y_train, y_test = all_features[train_idx], all_features[test_idx], final_mit_labels[train_idx], final_mit_labels[test_idx]
  zero_bias_history = Model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split=0.1,steps_per_epoch=2, callbacks=[callback])  #Fitting the model
  # y_train_pred = Model.predict(x_train) -- not needed as we can get this value directly from the model history created while fitting the model
  y_test_pred = ((Model.predict(x_test) > 0.5)+0).ravel()
  # y_test_pred=np.argmax(y_test_pred, axis=1)
  # y_test=np.argmax(y_test, axis=1)
  #save the accuracies in the created lists
  train_acc.append(np.array(zero_bias_history.history['accuracy']).mean())
  test_acc.append(accuracy_score(y_test, y_test_pred))
  f1_weight.append(f1_score(y_test, y_test_pred, average='weighted'))
  f1_macro.append(f1_score(y_test, y_test_pred, average='macro'))



print("")
print("Train accuracy {0}±{1}".format(round(np.array(train_acc).mean(),3),round(np.array(train_acc).std(),3)))
print("Test accuracy {0}±{1}".format(round(np.array(test_acc).mean(),3),round(np.array(test_acc).std(),3)))
print("F1 Weighted {0}±{1}".format(round(np.array(f1_weight).mean(),3),round(np.array(f1_weight).std(),3)))
print("F1 Macro {0}±{1}".format(round(np.array(f1_macro).mean(),3),round(np.array(f1_macro).std(),3)))
print("Validation accuracy {0}±{1}".format(round(np.array(val_acc).mean(),3),round(np.array(val_acc).std(),3)))

print("")
print("Train accuracy {0}±{1}".format(round(np.array(train_acc).mean(),3),round(np.array(train_acc).std(),3)))
print("Test accuracy {0}±{1}".format(round(np.array(test_acc).mean(),3),round(np.array(test_acc).std(),3)))
print("F1 Weighted {0}±{1}".format(round(np.array(f1_weight).mean(),3),round(np.array(f1_weight).std(),3)))
print("F1 Macro {0}±{1}".format(round(np.array(f1_macro).mean(),3),round(np.array(f1_macro).std(),3)))
