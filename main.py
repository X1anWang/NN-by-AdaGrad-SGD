import pandas as pd
import NeuralNetwork
import numpy as np

data_files = ['data1.csv']

for file in range(len(data_files)):
    # initialize parameters
    path = data_files[file]
    lr = 0.05
    epoch = 1
    limit = 300000
    dim = 200
    a = 0.75
    n_negative = 5
    max_size_np = 1e8
    
    NN = NeuralNetwork.NeuralNetwork(path, lr, epoch, limit, dim, 'unigram-table',
                                      a, n_negative, 'adagrad', 'input',
                                      max_size_np)
    
    i = 0
    vectors = []
    while True:
        i += 1
        if i % 10000 == 0:
            print(str(i) + " processed...")
        vector = NN.next_packet()
        if len(vector) == 0:
            break
        vectors.append(vector)

    
    vectors = np.array(vectors)
    vectors = np.mat(vectors)
    np.save(str(file)+".npy", vectors)