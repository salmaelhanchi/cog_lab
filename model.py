# imports
import torch
import torch.nn as nn


# class with Module inherited
class MemoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        
        super(MemoryRNN, self).__init__()
        
        # setting up the RNN
        # hyperparameters
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            # memory units
            hidden_size=hidden_size, 
            batch_first=True
        )
        # fully connected hidden state
        self.fc = nn.Linear(hidden_size, output_size)
    
    # h_n final hidden state    
    def forward(self, x):
        _ , h_n = self.rnn(x)
        # standarize for liear layer
        output = self.fc(h_n.squeeze(0))
        return output