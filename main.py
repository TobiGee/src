"""
Company: Reutlingen University
Lecture: Machine Vision & Artificial Intelligence
Created: May 22, 2021

This is the main file for the implementation of a simple GAN in python using the framework pytorch.
It uses following additional functions and classes:
 -> read_config from utils.py
 -> Discriminator from networks.py
 -> Generator from networks.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets #only for testing, later with custom data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from networks import Discriminator
from networks import Generator
from utils import read_config

def load_Dataset(config):
    """
    This function returns a loaded Dataset which is specified in the config dictionary. It may download
    the dataset and create the specified path if it is not able to find the set.

    Args:
    config : A dictionary with the extracted configuration out of the config.ini file and the commandlineparser. 
        Following dictionary-values are necessary:

        {'image_dim_x': intended image x-dimension, 
         'image_dim_y': intended image y-dimension,
         'datatype':    datatype of the dataset (MNIST | CIFAR10 | ImageFolder),
         'datapath':    absolut path to the dataset}


    Returns:
    A DataLoader object with the dataset specified in the config dictionary. 
    """
    dataset = None
    transform = transforms.Compose(
        [transforms.Resize((config["image_dim_x"], config["image_dim_y"])), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))]
    )
    if(not os.path.exists(config['datapath'])):
        if(config['datatype'] == "MNIST" or config['datatype'] == "CIFAR10"):
            print("Create Path")
            os.makedirs(config['datapath'])
        else:
            print("no dataset available")
            exit()

    if(config['datatype'] == "MNIST"):
        dataset = datasets.MNIST(root=config['datapath'], transform=transform, download = True)
    elif(config['datatype'] == "CIFAR10"):
        dataset = datasets.CIFAR10(root=config['datapath'], transform=transform, download = True)
    elif(config['datatype'] == "ImageFolder"):
        print("Imagefolder found")
        dataset = datasets.ImageFolder(root = config['datapath'], transform = transform)
    else:
        print("no dataset available")
        exit()
    
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle = True)
    return loader

def test(config):
    """
    This function tests a trained neuronal network. In this case the generator of the GAN.

    Args:
    config : A dictionary with the extracted configuration out of the config.ini file and the commandlineparser. 
        Following dictionary-values are necessary:

        {'image_dim_x': intended image x-dimension, 
         'image_dim_y': intended image y-dimension,
         'image_dim':   image_dim_x * image_dim_y,
         'batch_size':   size of the batch per iteration
         'datatype':    datatype of the dataset (MNIST | CIFAR10 | ImageFolder),
         'device':      device which should execute the test (gpu | cpu),
         'z_dim':       Hyperparameter for the Generator,
         'logpathfake': logpath for tensorboard SummaryWriter (--logdir)}
    """
    
    #load Generator  
    gen = Generator(config['z_dim'], config['image_dim']).to(config['device'])
    gen.load_model(config)

    #load Dataset
    print("Load dataset...")
    loader = load_Dataset(config)

    #initialize tensorboard summarywriter
    writer_fake = SummaryWriter(config['logpathfake'])
    writer_real = SummaryWriter(config['logpathreal'])
    trained_iterations = gen.training_iterations
    step_gen = gen.training_iterations
    #Testing trained Generator 
    print("Testing...")
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, config['image_dim']).to(config['device'])
        batch_size = real.shape[0]

        if batch_idx == 0:
            with torch.no_grad():
                noise = torch.randn(config['batch_size'], config['z_dim']).to(config['device'])
                fake = gen(noise).reshape(-1, 1, config['image_dim_x'], config['image_dim_y'])
                data = real.reshape(-1, 1, config['image_dim_x'], config['image_dim_y'])
                img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
                img_grid_real = torchvision.utils.make_grid(data, normalize = True)
                writer_fake.add_image(
                    "Mnist generated fake images out of test", img_grid_fake, global_step = trained_iterations
                )
                writer_real.add_image(
                    "Mnist reference Images", img_grid_real, global_step = 0
                )

def train(config): 
    """
    This function trains a GAN network with one Generator and one Discriminator in the simplest way.

    Args:
    config : A dictionary with the extracted configuration out of the config.ini file and the commandlineparser. 
        Following dictionary-values are necessary:

        {'image_dim_x':         intended image x-dimension, 
         'image_dim_y':         intended image y-dimension,
         'image_dim':           image_dim_x * image_dim_y,
         'batch_size':          size of the batch per iteration
         'datatype':            datatype of the dataset (MNIST | CIFAR10 | ImageFolder),
         'datapath':            absolut path to the dataset,
         'genpath':             path of the file where to save/load the Generator model,
         'discpath':            path of the file where to save/load the Discriminator model,
         'device':              device which should execute the test (gpu | cpu),
         'z_dim':               Hyperparameter for the Generator,
         'lr_gen':              Learning rate for the Generator Adam optimizer,
         'lr_disc':             Learning rate for the Discriminator Adam optimizer,
         'train_iterations':    training iterations for the GAN,
         'logpathfake':         logpath for tensorboard SummaryWriter (--logdir) to save the generated fake images,
         'logpathreal':         logpath for tensorboard SummaryWriter (--logdir) to save the real images used for training,
         'logpathgraph':        logpath for tensorboard SummaryWriter (--logdir) to save loss behavior of the networks}
    """

    #load dataset
    print("Load dataset...")
    loader = load_Dataset(config)


    #initialize GAN
    #torch.randn: returns a tensor filled with random numbers
    #optim.Adam: first order grandient-based optimization of stocastic objective functions
    #nn.BCELoss: Creates a criterion that measures the Binary Cross Entropy between the target and the output
    #SummaryWriter: initialize visualization with tensorboard
    print("Initialize GAN...")

    disc = Discriminator(config['image_dim']).to(config['device'])
    disc.load_model(config)
    
    gen = Generator(config['z_dim'], config['image_dim']).to(config['device'])
    gen.load_model(config)
    step_gen = gen.training_iterations
    step_disc = disc.training_iterations
    fixed_noise = torch.randn((config['batch_size'], config['z_dim'])).to(config['device'])

    opt_disc = optim.Adam(disc.parameters(), lr=config['lr_disc'])
    opt_gen = optim.Adam(gen.parameters(), lr=config['lr_gen'])
    criterion = nn.BCELoss()
    writer_fake = SummaryWriter(config['logpathfake'])
    writer_real = SummaryWriter(config['logpathreal'])
    writer_graphs = SummaryWriter(config['logpathgraph'])

    #Learning GAN
    #.view: allows a tensor to be a View of an existing tensor, avoids explicit data copy
    #.ones_like: returns a tensor filled with the scalar value 1 and the size of the input
    #.zeros_like: returns a tensor filled with the scalar value 0 and the size of the input
    #criterion(input, target(desired result))
    #.detach(): returns a new Tensor, detached from the current graph
    #.zero_grad(): set all gradients of all model parameters to zero
    #Tensor.backward: compute the gradient of current tensor
    #Adam.step(): perform a single optimization step
    print("Learning...")
    for iteration in range(config['train_iterations']):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, config['image_dim']).to(config['device'])
            batch_size = real.shape[0]
            
            #Train Discriminator
            noise = torch.randn(config['batch_size'], config['z_dim']).to(config['device'])
            fake = gen(noise)
            #train disc with real images
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real)) #label fakes with 1
            #give disc the fake images
            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) #label fakes with 0
            #calc loss
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph = True)
            opt_disc.step()

            #Train Generator
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output)) #label fakes with 1
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            #set tensorboard and console output
            if batch_idx == 0:
                print(
                    f"Iteration [{iteration}/{config['train_iterations']}] \ "
                    f"Loss D: {lossD:0.4f}, Loss G: {lossG: 0.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise).reshape(-1, 1, config['image_dim_x'], config['image_dim_y'])
                    data = real.reshape(-1, 1, config['image_dim_x'], config['image_dim_y'])
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize = True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step = step_gen+1
                    )
                    writer_real.add_image(
                        "Mnist real Images", img_grid_real, global_step = step_disc+1
                    )
                    writer_graphs.add_scalar("Loss Discriminator", lossD, step_disc + 1)
                    writer_graphs.add_scalar("Loss Generator", lossG,  step_gen + 1)
                    step_gen += 1
                    step_disc += 1

    #save model
    disc.save_model(config)
    gen.save_model(config)
    print("Discriminator trained finally " + str(disc.training_iterations) + " iterations")
    print("Generator trained finally " + str(gen.training_iterations) + " iterations")
    writer_fake.close()
    writer_real.close()
    writer_graphs.close()

if __name__=="__main__":
    print('Let\'s go!')
    config = read_config()

    if(config['phase'] == 'train'):
        train(config)
    elif (config['phase'] == 'test'):
        test(config)
   
