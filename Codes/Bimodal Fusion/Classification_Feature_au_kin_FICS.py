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


# In[4]:


# onehot encoding of kineme sequence
def onehot_encoding(ks, nKineme):
    onehot_encoded = list()
    for k in ks:
        vec = [0 for _ in range(nKineme)]
        vec[k-1] = 1
        onehot_encoded.append(vec)
    return onehot_encoded


def ks_encoding(ks, nKineme):
    # ks is a numpy ndarray
    m, n = ks.shape 
    ks = ks.tolist() 
    encoded_features = np.asarray(
        [np.asarray(onehot_encoding(ks[i], nKineme)) for i in range(m)]
    )
    return encoded_features

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

#define path for the training, testing and validation labels and data for the label Opneness of the FICS dataset
y_train_path = 'Kinemes/train_O.npy'
X_train_path = 'Kinemes/train_kineme_5990.npy'
y_test_path = 'Kinemes/test_O.npy'
X_test_path = 'Kinemes/test_kineme_1997.npy'
y_val_path = 'Kinemes/val_O.npy'
X_val_path = 'Kinemes/val_kineme_1995.npy'
#pass the above data paths as arguments to the training lstm function
training_lstm(y_train_path, X_train_path, y_test_path, X_test_path, y_val_path, X_val_path)


# In[9]:


def training_lstm(y_train_path, X_train_path, y_test_path, X_test_path, y_val_path, X_val_path):
    # parameters
    #sequence length for FICS dataset is 14 for all videos
    nKineme, seqLen, nClass = 16, 14, 2
    nAction = 17
    EPOCHS = 50
    BATCH_SIZE = 32
    nNeuron = 32

    #load all the data and label files and convert the labels to 0 or 1 depending on the median
    y_train = np.load(y_train_path)
    y_train = y_train[:,1].astype(np.float)
    y_train = bin_labels(y_train)
    train_mat = np.load(X_train_path)
    y_test = np.load(y_test_path)
    y_test = y_test[:,1].astype(np.float)
    y_test = bin_labels(y_test)
    test_mat = np.load(X_test_path)
    y_val = np.load(y_val_path)
    y_val = y_val[:,1].astype(np.float)
    y_val = bin_labels(y_val)
    val_mat = np.load(X_val_path)
    


    #Model architecture with different layers

    #Using two different lstm layers for the kineme and action unit data separately and concatenating them at the end
    #kineme lstm implementation; input is the encoding of kineme sequence of 16 and 14 is the time step for each file; sequence of kinemes from each file
    #we provide the input shape to a lstm layer with activation
    left_branch_input = Input(shape=(seqLen,nKineme), name='Left_input')
    left_branch_output = LSTM(32, activation='relu')(left_branch_input)

    #in right side, we have action units as the input sequences with each as a 17 dimensional vector
    right_branch_input = Input(shape=(seqLen,nAction), name='Right_input')
    right_branch_output = LSTM(32, activation='relu')(right_branch_input)

    #merging the two layers using a dense layer and then having one final output layer and the optimizer, compile and summary
    merged = concatenate([left_branch_output, right_branch_output], name='Concatenate')
    final_model_output = Dense(2, activation='sigmoid')(merged)
    final_model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                        name='Final_output')
    opt = keras.optimizers.Adam(learning_rate=0.01)
    final_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=['accuracy'])
    final_model.summary()


    #define the loss and accuracy lists
    test_loss=[]
    test_acc=[]
    train_loss = []
    train_acc = []
    fi_weighted=[]
    fi_macro=[]
    val_loss = []
    val_acc = []

    #passing the features and labels data to their respective variables as train, test or validation
    train_features,  train_labels = train_mat, y_train
    test_features,  test_labels = test_mat, y_test
    val_features,  val_labels = val_mat, y_val
    
    #extract the kinemes and convert each value to the encoding using the ks_encoding method
    #do the same for train, test and validation data
    train_kinemes = ks_encoding(train_features[:,0:seqLen], nKineme)
    test_kinemes = ks_encoding(test_features[:,0:seqLen], nKineme)
    val_kinemes = ks_encoding(val_features[:,0:seqLen], nKineme)
    
    #extract the remaining values for action units
    train_action = train_features[:, seqLen:]
    test_action = test_features[:, seqLen:]
    val_action = val_features[:, seqLen:]
    
    #reshape the action unit to a 3d matrix, similar to that of kinemes
    train_aus = train_action.reshape((train_action.shape[0], seqLen, nAction))
    test_aus = test_action.reshape((test_action.shape[0], seqLen, nAction))
    val_aus = val_action.reshape((val_action.shape[0], seqLen, nAction))
    
    # convert labels into categorical
    train_labels = to_categorical(train_labels)   
    val_labels = to_categorical(val_labels)  
    
    #model training y passing both kineme and AU data along with the training labels
    zero_bias_history = Model.fit([train_kinemes, train_aus], train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data=([val_kinemes, val_aus], val_labels), callbacks=[callback]) 
    #evaluate the model using the test data
    score = Model.evaluate([test_kinemes, test_aus], to_categorical(test_labels), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    #append the test, train and val accuracy to the above lists
    test_loss.append(score[0])
    test_acc.append(score[1])
    train_acc.append(np.array(zero_bias_history.history['accuracy']).mean())
    val_acc.append(np.array(zero_bias_history.history['val_accuracy']).mean())
    
    #find the prediction values that will be used for finding the f1 score 
    y_testpred = Model.predict_classes([test_kinemes, test_aus])
    f1_w_epoch = f1_score(test_labels, y_testpred, average='weighted')
    f1_m_epoch = f1_score(test_labels, y_testpred, average='macro')
    fi_weighted.append(f1_w_epoch)
    fi_macro.append(f1_m_epoch)


    #print the accuracies and F1 scores
    print("For:" + str(Label_class))
    print("Train_accuracy {0}±{1}".format(round(np.array(train_acc).mean(),3),round(np.array(train_acc).std(),3)))
    print("Test_accuracy {0}±{1}".format(round(np.array(test_acc).mean(),3),round(np.array(test_acc).std(),3)))
    print("F1_Weighted {0}±{1}".format(round(np.array(fi_weighted).mean(),3),round(np.array(fi_weighted).std(),3)))
    print("F1_Macro {0}±{1}".format(round(np.array(fi_macro).mean(),3),round(np.array(fi_macro).std(),3)))

