# In model.py

import torch
import torch.nn as nn

class MemoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MemoryRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # An LSTM's hidden state is a tuple: (hidden_state, cell_state)
        # We only need the final hidden_state for our classifier.
        _ , (h_n, c_n) = self.lstm(x) 
        
        output = self.fc(h_n.squeeze(0))
        return output