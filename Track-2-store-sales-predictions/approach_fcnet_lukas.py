# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:49:56 2021

@author: luki
"""

import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd


#*** PREPROCESSING ***
df = pd.read_csv("train_data.csv")

df['month'] = pd.DatetimeIndex(df['Date']).month
df = pd.get_dummies(data=df, columns=['AssortmentType', 'StoreType', 'StateHoliday', "month", "DayOfWeek"])
del df["Date"]
del df["Store"]

Y = torch.tensor(df["Sales"].values)
del df["Sales"]

X = torch.tensor(df.values)

data = []
for i in range(len(X)):
    data.append([X[i], Y[i]])
    
dataloader = DataLoader(data, batch_size=128, shuffle=True)




#*** FCNet ***#
class FCNetwork(nn.Module):
    """ Network with fully-connected layers. """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Number of input dimensions.
        out_features : int
            Number of output dimensions.
        """
        super(FCNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, out_features)
        self.tanh_fc1 = nn.Tanh()
        self.tanh_fc2 = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh_fc1(self.fc1(x))
        x = self.tanh_fc2(self.fc2(x))
        return self.fc3(x)


def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> list:
    """
    Evaluate the performance of a network on some metric.
    
    Parameters
    ----------
    network : nn.Module
        Pytorch module representing the network.
    data : DataLoader
        Pytorch dataloader that is able to 
        efficiently sample mini-batches of data.
    metric : callable
        Function that computes a scalar metric
        from the network logits and true data labels.
        The function should expect pytorch tensors as inputs.

    Returns
    -------
    errors : list
        The computed metric for each mini-batch in `data`.
    """
    error_list = []
    with torch.no_grad():
        for x, y in data:
            
            prediction = network.forward(x)
            error_list.append(metric(prediction, y))
        
    return torch.tensor(error_list)
    

def update(network: nn.Module, data: DataLoader, loss: nn.Module, 
           opt: optim.Optimizer) -> list:
    """
    Update the network to minimise some loss using a given optimiser.
    
    Parameters
    ----------
    network : nn.Module
        Pytorch module representing the network.
    data : DataLoader
        Pytorch dataloader that is able to 
        efficiently sample mini-batches of data.
    loss : nn.Module
        Pytorch function that computes a scalar loss
        from the network logits and true data labels.
    opt : optim.Optimiser
        Pytorch optimiser to use for minimising the objective.

    Returns
    -------
    errors : list
        The computed loss for each mini-batch in `data`.
    """
    error_list = []
    counter = 0
    for x, y in data:
        
        print(x.float()[0])
        pred = network(x.float())
        print(pred, y.float())

        le = loss(pred, y.float())
        error_list.append(le)
        opt.zero_grad()
        le.backward()
        opt.step()
        counter += 1
        
    print("Loss: ", torch.mean(torch.tensor(error_list)) )
    
    return torch.tensor(error_list)


epochs = 15
lr = 0.001
batch_size = 128
shuffle = True
momentum = 0.9


fc_net = FCNetwork(34, 1)
update(fc_net, dataloader, loss=nn.MSELoss(), opt=torch.optim.Adam(fc_net.parameters(), lr=lr))


