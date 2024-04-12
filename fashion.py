import numpy as np
import matplotlib.pyplot as plt

import torch 
import torchvision # load our datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # building block for neural net
import torch.nn.functional as F # convolution functions like Relu
import torch.optim as optim # optimizer


## Loading and normalizing data
#python image library has range [0,1] we are converting to [-1,1]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.50,0.5))]
)

batch_size=4
num_workers = 2

#loading train data
trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                        download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True,num_workers=num_workers)

#loading test data
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
