import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

#Attempt at a dynamic CNN model creater class, probably need to add maxpool and relu for final edition

class Dynamic_CNN(nn.Module):
    def __init__(self, in_dim, out_dim, layers):
        super(Dynamic_CNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        
        self.conv_layers = nn.ModuleList()
        in_channels = in_dim[0]  # Extract the number of channels
        
        for out_channels, kernel_size in layers:
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=kernel_size)
            )
            self.conv_layers.append(conv_layer)
            in_channels = out_channels
        
        self.fc = nn.Linear(in_channels * in_dim[1] * in_dim[2], out_dim)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Dynamic_CNN2(nn.Module):
    def __init__(self, in_dim, out_dim, layers):
        super(Dynamic_CNN2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        
        self.layers = nn.ModuleList()
        in_chan = in_dim

        for out_chan, kernel_size in layers:
            conv_layer = nn.Sequential(
                            nn.Conv2d(in_chan, out_chan, kernel_size),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=kernel_size))
            self.layers.append(conv_layer)
            in_chan = out_chan
        
        self.fc = nn.Linear(in_chan, out_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x