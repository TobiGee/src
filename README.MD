MV Projekt: simple GAN Implementation in Python using the framework PyTorch
---

This project is part of the lecture "Machine Vision and Artifficial intelligence" on reutlingen university.
It aims to build a toolbox for using generative adversary networks. Moreover it puts focus on experimenting with GANs in a simple way. This project is based on the GAN implementation by Aladdin Persson but modified and implemented into a own written framework. You can find his implementation under the following links.
  
- YouTube video: https://www.youtube.com/watch?v=OljTVUVzPpM
- GitHub repository: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/1.%20SimpleGAN)

## Prerequisites
To use this program, you will need python and the frameworks pytorch and torchvision installed on your computer.
It's recommended to use an anaconda environment for the frameworks. For installation guides please visit https://www.anaconda.com/ and https://pytorch.org/. In addition you have to install the cudatoolkit and tensorboard too.

## Introduction
In this project a simple, modular GAN is implemented for experimentation with an example configuration for the MNIST-Dataset. You can train the GAN networks based on your configuration and test the trained Generator. In addition it will allow you to save and load the trained neural networks. 
The program is based on three python source files and one config.ini file. The following subsections explain the functionality, structre and content of these files.

### main.py
The main.py file includes besides the main function the test and train functionality encapsulated into two different functions train(config) and test(config). A third function load_Dataset(config) loads the dataset. All of them need a dictionary named config which contains the parameters from the config.ini file and the paramters out of the commandline.

### networks.py
In this file the Generator and Discriminator networks are defined in classes. They inherit from torch.nn.Module and have two additional methods for saving and loading the model based on the configuration dictionary.

### utils.py
To read the configuration out of the config.ini file and the commandline, three functions are written in the utils.py file. The read_config() function returns the config dictionary which is used in the whole project. To create read and create all parameters, this function uses the other two functions named parse_args() and createLogPaths(). parse_args() will accept the following three paramters in the command line:

 --iteration: how many iterations do you want to train the GAN (default: 10)

 --phase: do you want to test or train the GAN? (default: train)

 --hypset: the name of the generic term in the config.ini file which summarizes the configuration which you want to use for the program run.

### config.ini
This file specifies the configuration for the program usage. It has to own the following parameters subordinated to a generic term. The necessary config-structure is listed below.

['generic term']
  - **datatype**: type of the dataset(MNIST | CIFAR10 | ImageFolder)
  - **device**: device which should process the training or testing(gpu | cpu)
  - **learning_rate_gen_ada**: learning rate of the adam algorithm for the generator
  - **learning_rate_disc_ada**: learning rate of the adam algorithm for the discriminator
  - **z_dim**: z-dimension for the neural network
  - **image_dim_x**: intended image x-direction
  - **image_dim_y**: intended image y-direction
  - **batch_size**: number of batches per iteration
  - **gen_model_path**: relativ path for saving and loading the Generator model
  - **disc_model_path**: relativ path for saving and loading the Discriminator model
  - **dataset_path**: relativ path to the dataset
  - **logging_path**: relativ path where the tensorboard logs should be

For an example take a look into the file where you can find the configuration for the MNIST-dataset.

## How to start and use the program

### First usage

First download the program or clone the git repository with the following command:
```bash
https://github.com/TobiGee/src.git
```
To take a first look to the program you can start it in your python environment by calling:
```bash
python main.py
```
It will create a Generator and a Discriminator network, train them for 10 iterations and saves the results. Therefore the program will download the MNIST-Dataset and create a models and runs folder in the src folder. 
* The models folder will contain a Discriminator.pth and a Generator.pth file with the trained models. 
* The runs folder will contain log files for the run which you can display with the following command inside the python environment:
```bash
tensorboard --logdir runs/MNIST/train/{foldername(timestamp)}
```
It will open a local host website where you can view the results.

### Custom GAN training
For custom GAN training you only have to make or change a configuration set in the config.ini file. Of course, you can customise the neural networks too.
The command to start the training with the custom configuration and dataset can be look like the followed one:
```bash
python main.py --iteration 100 --phase train --hypset CIFAR10
```
