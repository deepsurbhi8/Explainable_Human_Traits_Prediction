#Function to create kineme data matrix for Train Data
# chunk_time - chunk duration considered for the analysis
# input_file_kineme - kineme values csv file
# label_val - label value for the file
# num_chunk - number of chunks in the entire video file
def data_preprocess(chunk_time, input_file_kineme, label_val,num_chunk): 
    data_list = []
    total_num_chunks = 0
    label_file_index = 0
    final_label = []
    
    #print("Creating the input from csv files")
    final_data_file = input_file_kineme.T
    #num_chunk = int(size_file/chunk_time)

    # dividing the entire data into the num of chunks
    for num in range(0, num_chunk):
        data_chunk = final_data_file[num*(chunk_time-1):((num+1)*(chunk_time-1))]
        data_chunk = data_chunk.to_numpy()
        data_list.append(data_chunk.flatten())
    #print(data_list)
    one_list = label_val.repeat(num_chunk)
    final_label.extend(one_list)
    #print(final_label)
    #print(total_num_chunks, np.shape(final_label))
    #label_file_index += 1                   
    return  data_list, final_label


#Function to create kineme data matrix for Test Data
def data_preprocess_test(chunk_time, input_file_kineme,num_chunk): 
    data_list = []
    total_num_chunks = 0
    label_file_index = 0
    final_label = []
    
    #print("Creating the input from csv files")
    final_data_file = input_file_kineme.T
    #num_chunk = int(size_file/chunk_time)
        
    for num in range(0, num_chunk):
        data_chunk = final_data_file[num*(chunk_time-1):((num+1)*(chunk_time-1))]
        data_chunk = data_chunk.to_numpy()
        data_list.append(data_chunk.flatten())
    #print(data_list)
    '''one_list = label_val.repeat(num_chunk)
    final_label.extend(one_list)
    print(final_label)
    #print(total_num_chunks, np.shape(final_label))
    #label_file_index += 1 '''                  
    return  data_list
