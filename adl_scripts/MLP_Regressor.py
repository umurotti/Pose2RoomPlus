import torch
from torch import nn


class MLP_Regressor(nn.Module):

  def __init__(self, input_size, output_size, layer_sizes):
    super().__init__()
    
    layerlist = []
    layerlist.append(nn.Linear(input_size, layer_sizes[0]))
    for index in range(0, len(layer_sizes) - 1):
        layerlist.append(nn.ReLU())
        layerlist.append(nn.Linear(layer_sizes[index],layer_sizes[index + 1]))
    layerlist.append(nn.ReLU())
    layerlist.append(nn.Linear(layer_sizes[-1], output_size))
    
    self.layers = nn.ModuleList(layerlist)


  def forward(self, x):
    '''Forward pass'''
    for layer in self.layers:
        x = layer(x)
    return x