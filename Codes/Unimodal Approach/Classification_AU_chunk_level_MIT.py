
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

#define the data and label file path for each label
y_data_path = 'MIT/Chunk_level/Label_60_Overall.npy'
X_data_path = 'MIT/Chunk_level/Data_60_Overall_new.npy'
training_lstm(y_data_path, X_data_path, "Overall")
y_data_path = 'MIT/Chunk_level/Label_60_Excited.npy'
X_data_path = 'MIT/Chunk_level/Data_60_Excited.npy'
training_lstm(y_data_path, X_data_path, "Excited")


def training_lstm(y_data_path, X_data_path, Label_class):
    # parameters
    #nKineme contains the number of kinemes clusters created
    #seqLen defines the length/size of eaach chunk
    #nAction is the size of vector for action units; as we extract 17 action units from openface
    nKineme, seqLen, nClass = 16, 59, 2
    nAction = 17
    EPOCHS = 30
    BATCH_SIZE = 32
    nNeuron = 12

    #load the data and convert labels to categorical
    y_data = np.load(y_data_path)
    y_data = y_data.astype(np.float)
    y_data = bin_labels(y_data)
    X_data = np.load(X_data_path)



    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    Model = Sequential()
    Model.add(LSTM(nNeuron,activation="tanh",dropout=0.1,input_shape=(seqLen, nAction)))
    #regressor.add(Dropout(0.0))
    Model.add(Dense(units = nClass,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    Model.compile(optimizer = opt, loss = 'binary_crossentropy',metrics=['accuracy'])
    Model.summary()

#     print("abc")

    test_loss=[]
    test_acc=[]
    train_loss = []
    train_acc = []
    fi_weighted=[]
    fi_macro=[]
    val_loss = []
    val_acc = []

    random_state = 42
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)      #repeat kfold function
    for train_idx, test_idx in rkf.split(X_data):
        train_features, test_features, train_labels, test_labels = X_data[train_idx], X_data[test_idx], y_data[train_idx], y_data[test_idx] 
        train_action = train_features[:, seqLen:]
        test_action = test_features[:, seqLen:]
        train_action = train_action.reshape((train_action.shape[0], seqLen, nAction))
        test_action = test_action.reshape((test_action.shape[0], seqLen, nAction))
        # convert labels into categorical
        train_labels = to_categorical(train_labels)   
        zero_bias_history = Model.fit(train_action, train_labels, epochs = EPOCHS, batch_size = 32, validation_split=0.1, callbacks=[callback]) 
        score = Model.evaluate(test_action, to_categorical(test_labels), verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        test_loss.append(score[0])
        test_acc.append(score[1])
        train_acc.append(np.array(zero_bias_history.history['accuracy']).mean())
        val_acc.append(np.array(zero_bias_history.history['val_accuracy']).mean())
        y_testpred = Model.predict_classes(test_action)
        f1_w_epoch = f1_score(test_labels, y_testpred, average='weighted')
        f1_m_epoch = f1_score(test_labels, y_testpred, average='macro')
        fi_weighted.append(f1_w_epoch)
        fi_macro.append(f1_m_epoch)


    print("For:" + str(Label_class))
    print("Train_accuracy {0}±{1}".format(round(np.array(train_acc).mean(),3),round(np.array(train_acc).std(),3)))
    print("Test_accuracy {0}±{1}".format(round(np.array(test_acc).mean(),3),round(np.array(test_acc).std(),3)))
    print("F1_Weighted {0}±{1}".format(round(np.array(fi_weighted).mean(),3),round(np.array(fi_weighted).std(),3)))
    print("F1_Macro {0}±{1}".format(round(np.array(fi_macro).mean(),3),round(np.array(fi_macro).std(),3)))

