"""
Company: Reutlingen University
Lecture: Machine Vision & Artificial Intelligence
Created: May 22, 2021
License: MIT

In this utils-file are functions implemented which are used for reading the configurations.
"""
import argparse
from configparser import ConfigParser
import datetime
import torch.cuda
import os

def parse_args():
    """
    This function parses commandline arguments. You have the following input possiblities:
        --phase:        determines if you want to test or train your model (default: train)
        --iteration:    determines how many training iterations you want to train. This do not affect tests.
        --hypset:       name from the header in the config.ini file which summarises the configuration


    Returns:
    The parsed arguments assigned to a namespace.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='train or test?')
    parser.add_argument('--iteration', type=int, default=10, help='The number of training iterations')
    parser.add_argument('--hypset', type=str, default='MNIST', help='The name of the hypset in the config.ini file which should be used')
    return parser.parse_args()


def createLogPaths(relative_path, phase):
    """
    This function creates three absolut paths for the logging directory for:
        1. logging the fake images
        2. logging the real images
        3. logging some graphs
    For automatic generic order, the actual timestamp will be used.
    Args:
    relative_path: Relative path to the intended logging folder
    pahse: phase for logging (test or train)
    Returns:
    The created paths.
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path_real = os.path.join(os.path.dirname(os.path.realpath(__file__)) + relative_path +'/' + phase +'/'+ current_time + '/real')
    log_path_fake = os.path.join(os.path.dirname(os.path.realpath(__file__)) + relative_path + '/' + phase +'/'+ current_time + '/fake')
    log_path_graph = os.path.join(os.path.dirname(os.path.realpath(__file__)) + relative_path + '/' + phase +'/'+ current_time + '/graphs')
    return log_path_real, log_path_fake, log_path_graph

def read_config():
    """
    This function reads the configuration for the training or testing process out of the config.ini file 
    and the commandline. Therefore you have to specifiy the following values in a set in the config file:
        ['set']
        datatype                = str{type of the dataset(MNIST | CIFAR10 | ImageFolder)}
        device                  = str{device which should process the training or testing(gpu | cpu)}
        learning_rate_gen_ada   = int{learning rate of the adam algorithm for the generator}
        learning_rate_disc_ada  = int{learning rate of the adam algorithm for the discriminator}
        z_dim                   = int{z-dimension for the neuronal network}
        image_dim_x             = int{intended image x-direction}
        image_dim_y             = int{intended image y-direction}
        batch_size              = int{number of batches per iteration}
        gen_model_path          = str{relativ path for saving and loading the Generator model}
        disc_model_path         = str{relativ path for saving and loading the Discriminator model}
        dataset_path            = str{relativ path to the dataset}
        logging_path            = str{relativ path where the tensorboard logs should be}


    Returns:
    A dictionary with the following values as a configuration for training or testing a GAN:
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

    config = {}

    print("Read config from command line...")
    args = parse_args()
    hypset = args.hypset
    config['train_iterations'] = args.iteration
    config['phase'] = args.phase

    print("Read config from config.ini...")
    config_parser = ConfigParser()
    configFile_path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + '\config.ini')
    config_parser.read(configFile_path)
    config['datatype'] = config_parser[hypset]['datatype']
    config['genpath'] = os.path.join(os.path.dirname(os.path.realpath(__file__)) 
                        + config_parser[hypset]['gen_model_path'])
    config['discpath'] = os.path.join(os.path.dirname(os.path.realpath(__file__)) 
                        + config_parser[hypset]['disc_model_path'])
    config['datapath'] = os.path.join(os.path.dirname(os.path.realpath(__file__)) 
                        + config_parser[hypset]['dataset_path'])
    config['logpathreal'], config['logpathfake'], config['logpathgraph'] = createLogPaths(config_parser[hypset]['logging_path'], config['phase'])
    device = str(config_parser[hypset]['device'])
    #check if cuda if available, if the config file choose gpu as the computing device
    if(device == 'gpu'):    
        if(torch.cuda.is_available()):
           config['device'] = 'cuda'
        else:
            config['device'] = 'cpu'
    config['lr_gen'] = float(config_parser[hypset]['learning_rate_gen_ada'])
    config['lr_disc'] = float(config_parser[hypset]['learning_rate_disc_ada'])
    config['z_dim'] = int(config_parser[hypset]['z_dim'])    
    config['image_dim_x'] = int(config_parser[hypset]['image_dim_x'])
    config['image_dim_y'] = int(config_parser[hypset]['image_dim_y'])
    config['image_dim'] =  config['image_dim_x'] * config['image_dim_y']
    config['batch_size'] = int(config_parser[hypset]['batch_size'])

    return config
