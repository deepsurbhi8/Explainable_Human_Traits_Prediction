# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:18:09 2021

@author: SURBHI MADAL
"""
import numpy as np
import pandas as pd
import glob


chunk_time, size_of_feature_set  = 10,23
all_files = sorted(glob.glob('E:/Kineme_Audio/MIT_Data_CSV/*.csv'))

# all_files = all_files[0:4]
chunks_array = np.load('E:/Kineme_Audio/Chunk_Array_For_Audio/Label_Count_10.npy') #We created this list from au+kin formed matrix to be consistent with chunks formed 
    
data_list = []
k = 0
for f in all_files:
    csv_file   = pd.read_csv(f, header=None)
    csv_file = csv_file.fillna(method='ffill')
    # len_file = csv_file.shape[1]
    i = 0          
    entire_file_mat = []
    
    while i < csv_file.shape[1]:
      new_frame = csv_file.loc[:, i:i+87]
      i = i + 44
      avg_value = new_frame.mean(axis=1)
      file_mat = pd.concat([avg_value], axis=1, ignore_index=True)
      # print(file_mat.shape)
      file_mat = file_mat.to_numpy().flatten()
      file_mat = pd.DataFrame(file_mat)
      entire_file_mat = np.concatenate((entire_file_mat, file_mat), axis=None)
      # print(np.shape(entire_file_mat))
      new_array = entire_file_mat
      value_to_be_minus = new_array.shape[0]-(2*size_of_feature_set)
      new_array = new_array[0:value_to_be_minus] #To handle extra 2 sec data
      
    num_chunks = chunks_array[k]  #Main change to the number of chunks
    for num in range(0, num_chunks):
        data_chunk = entire_file_mat[num*(chunk_time-1)*size_of_feature_set:((num+1)*(chunk_time-1)*size_of_feature_set)]
        data_list.append(data_chunk)
    print(k)
    k +=1
   
data_array = np.array(data_list) 
my_df = pd.DataFrame(data_array)
Data = my_df.fillna(method='ffill') #to fill nan values
Data = np.asarray(Data)


Data_AU_Kin = np.load('E:/Kineme_Audio/LSTM_AU_Kineme-20211122T162529Z-001/LSTM_AU_Kineme/Audio_Data/Data_10_Excited.npy') #Just to check with proper number of chuks

