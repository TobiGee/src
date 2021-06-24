"""
Company: Reutlingen University
Lecture: Machine Vision & Artificial Intelligence
Created: May 22, 2021
License: MIT

In this file, the Generator and the Discriminator model are defined.
"""

import torch.nn as nn
import torch
import os
#####################################################################
# nn.Module: base class for all neural network modules
# nn.Sequential: sequential container, modules will be added to it 
#   in the order they are passed in the constructor
#nn.Linear: Applies a linear transformation to the incoming data y=xAT+by = xA^T + by=xAT+b 
#nn.LeakyReLU: Applies the element-wise fnction: LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
#nn.Sigmoid: Applies the element-wise function Sigmoid(x)=σ(x)=1/(1+exp(−x))
#nn.Tanh: Applies the element-wise function: Tanh(x)=tanh(x)=(exp(x)-exp(−x))/(exp(x)−exp(−x)​)
class Discriminator(nn.Module):
    """
    The class Discriminator inherits from nn.Module and represents the Discriminator network for a GAN.
    """
    def __init__(self, img_dim):
        """
        Constructor for the Discriminator class. It specifies the structure of the neuronal network.
        Args:
        img_dim: dimension of the images from the dataset
        """
        super().__init__()
        self.training_iterations = 0
        self.loss_over_iterations = None
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Default forward method for the neuronal network which defines the network transformation.
        Args:
        x: input tensor
        Returns:
        A predicted tensor
        """
        return self.disc(x)

    def save_model(self,config):
        """
        Method to save the Modelparameters to the in the config specified path. 
        If the file already exists override it, otherwise create it.
        Args:
        config: This method uses: 
                    {'discpath': absolute path to saving directory, 
                     'train_iterations': number ot training iterations}
        """
        filepath = config['discpath']
        if(os.path.exists(filepath)):
            print('Replace existing model')
            os.remove(filepath) #delete old model to replace it with the new one
        else:
            print("File do not exist")
            path_head, path_tail = os.path.split(filepath)
            if(not os.path.exists(path_head)):
                print('Path do not exist, create Path: ' + path_head)
                os.makedirs(path_head)
        self.training_iterations += config['train_iterations']
        torch.save({'state_dict': self.state_dict(), 
                    'iterations': self.training_iterations}, filepath)

    def load_model(self, config):
        """
        Method to load the Modelparameters from the in the config specified path. 
        If the file not exists create a new model.
        Args:
        config: This method uses: 
                    {'discpath': absolute path to the directory}
        """
        filepath = config['discpath']
        if(os.path.exists(filepath)):
            print("Load Discriminator from " + filepath)
            model = torch.load(filepath)
            self.load_state_dict(model['state_dict'])
            self.training_iterations = model['iterations']
            print("Trained iterations: ", self.training_iterations)
        else:
            print("No such file found, create new Discriminator")


class Generator(nn.Module):
    """
    The class Generator inherits from nn.Module and represents the Generator network for a GAN.
    """
    def __init__(self, z_dim, img_dim):
        """
        Constructor for the Generator class. It specifies the structure of the neuronal network.
        Args:
        z_dim: z-Dimension for the first linear layer
        img_dim: dimension of the images from the dataset
        """
        super().__init__()
        self.training_iterations = 0
        self.loss_over_iterations = None
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Default forward method for the neuronal network which defines the network transformation.
        Args:
        x: input tensor
        Returns:
        A predicted tensor
        """
        return self.gen(x)
    
    def save_model(self, config):
        """
        Method to save the Model to the in the config specified path. 
        If the file already exists override it, otherwise create it.
        Args:
        config: This method uses: 
                    {'genpath': absolute path to saving directory, 
                     'train_iterations': number ot training iterations}
        """
        filepath = config['genpath']
        if(os.path.exists(filepath)):
            print('Replace existing model')
            os.remove(filepath) #delete old model to replace it with the new one
        else:
            print("File do not exist")
            path_head, path_tail = os.path.split(filepath)
            if(not os.path.exists(path_head)):
                print('Path do not exist, create Path: ' + path_head)
                os.makedirs(path_head)
        self.training_iterations += config['train_iterations']
        torch.save({'state_dict': self.state_dict(), 
                    'iterations': self.training_iterations}, filepath)

    def load_model(self, config):
        """
        Method to load the Modelparameters from the in the config specified path. 
        If the file not exists create a new model.
        Args:
        config: This method uses: 
                    {'genpath': absolute path to the directory}
        """
        filepath = config['genpath']
        if(os.path.exists(filepath)):
            print("Load Generator from: " + filepath)
            model = torch.load(filepath)
            self.load_state_dict(model['state_dict'])
            self.training_iterations = model['iterations']
            print("Trained iterations: ", self.training_iterations)
        else:
            print("No such file found, create new Generator")