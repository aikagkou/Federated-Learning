import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
      # input_size : number of features in input at each time step
      # hidden_size : Number of LSTM units
      # num_layers : number of LSTM layers
      # batch_first= True: input data will have the batch size as the first dimension
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,dropout_prob=0.2):
        super(LSTMModel, self).__init__() #initializes the parent class nn.Module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x): # defines forward pass of the neural network
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out)
        return out
