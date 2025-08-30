# imports

import torch
import numpy as np

def generate_task_data(num_samples, sequence_length):
    
    # input size
    vocab_size = 3
    
    # random array
    input_signals = np.random.randint(1, vocab_size, size=(num_samples,))
    
    X = np.zeros((num_samples, sequence_length, vocab_size), dtype=np.float32)
    y = np.array(input_signals, dtype=np.int64)
    
    # data generation
    for i in range (num_samples):
        X[i, 0, input_signals[i]]=1
        
    return torch.from_numpy(X), torch.from_numpy(y)
    