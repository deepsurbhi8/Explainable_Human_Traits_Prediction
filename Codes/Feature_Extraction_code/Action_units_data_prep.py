#Function to create AU Data Matrix
def max_encoding(input_file_au,threshold,chunk_size,total_chunks):
  count = 0
  duration = 0
  overlap = 10
  complete_vec = []
  intensity_var = threshold  
  read_data = np.array(input_file_au) 
  for c in range(0,total_chunks,1):           #total_chunks= 10 , c= 0 
      #print("Value of total chunks")
      #print(c)
      vector = []
      #print(c)
      for j in range(0, chunk_size-1, 1): #0 1 2 0 20 10 30 20 40
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
             # print("vector for 2 seconds")
             # print(max_pool)
         duration = duration + overlap
         vector.append(max_pool)
         #print(vector)
      vec_flat = np.array(vector).flatten()
      #print(vec_flat)
      complete_vec.append(vec_flat)
      # print("duration and overlap")
      # print(duration, overlap)
      # print(vector)
      # complete_vec.append(vector)  
  return complete_vec