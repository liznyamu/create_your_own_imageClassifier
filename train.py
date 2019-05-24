# Train a new network on a dataset with train.py and save a checkpoint

# Basic usage:
#   python train.py data_directory
#   Prints out training loss, validation loss, and validation accuracy as the network trains

# Options:
# Set directory to save checkpoints:
#   python train.py data_dir --save_dir save_directory
#   python train.py flowers --save_dir checkpoint.pth

# Choose architecture:
#   python train.py data_dir --arch "vgg13"
#   python train.py flowers --arch "vgg19"

# Set hyperparameters:
#   python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#   python train.py flowers --learning_rate 0.003 --hidden_units 1024 --epochs 10

# Use GPU for training:
#   python train.py data_dir --gpu

# imports here
import torch
from torch import nn, optim
from helpers.our_model import (load_data, load_arch, train_model,
set_classifier, save_checkpoint)
import argparse

# handling options and arguments
parser = argparse.ArgumentParser(
    description='Train a new network on a dataset and save a checkpoint'
)

parser.add_argument('data_dir', action='store', default = 'flowers',
                    help='Store path to train, test and valid images')

parser.add_argument('--save_dir', action='store', dest='save_dir',
                    default='checkpoint.pth',
                    help='Store checkpoint file path')

parser.add_argument('--arch', action='store', dest='arch',
                    default='vgg19',
                    help='Store pretrained network architecture')

parser.add_argument('--learning_rate', action='store', dest='learning_rate',
                    default=0.003, type=float,
                    help='Store learning_rate for the trained model')

parser.add_argument('--hidden_units', action='store', dest='hidden_units',
                    default=1024, type=int,
                    help='Store hidden units for the trained model')

parser.add_argument('--epochs', action='store', dest='epochs',
                    default=10, type=int,
                    help='Store number of epochs for the trained model')

parser.add_argument('--gpu', action='store_true', dest='gpu',
                    default=False, help='Whether or not to use GPU in training the model')


results = parser.parse_args()

# directory to images
data_dir = results.data_dir         # 'flowers'
# checkpoint save path/directory
save_directory = results.save_dir   # 'vgg19_wM_checkpoint.pth' # 'vgg19_woM_checkpoint.pth' - TODO change on predict

# pre-trained model to use
arch = results.arch                 # 'vgg19'

# hyperparameters on training
learning_rate = results.learning_rate   # model's learning rate : 0.003
hidden_units = results.hidden_units # number of hidden layer units : 1024
epochs = results.epochs     # number of runs through the entire train dataset : 10

# gpu if device is set
gpu = results.gpu

# Starting to train the models
print('\nSetting Training environment', ' ... '*6)

# Use GPU if it's available and user requests to use GPU during training
if gpu and torch.cuda.is_available():
    print('Using GPU to train the model', ' ... '*3)
    device = torch.device("cuda")
else:
    print('Using CPU to train the model', ' ... '*3)
    print('''\nWARN : This could take some time to train the model
             use option `--gpu` and switch on GPU mode for faster training.\n\n
    ''')
    device = torch.device("cpu")

# load the data - from image folders
print('Loading the datasets - from image folders', ' ... '*3)
dataloaders, image_datasets, train_loader, valid_loader, test_loader = load_data(data_dir)

# load a pre-trained network model - default vgg19
print(f'Loading pre-trained network : {arch}', ' ... '*3)
model = load_arch(arch)


# add our classifier with user's hidden units
print(f'Adding our classifier to our model :\nHidden layer units {hidden_units}')
model, classifier = set_classifier(model, hidden_units)


# Define the loss criterion - Negative Log-likelihood loss
print('Defining the loss criterion - Negative Log-likelihood loss')
criterion = nn.NLLLoss()

# Now we train the classifier parameters - as feature parameters are frozen --- device and loaders**
print('Setting Optimizer - Negative Log-likelihood loss')
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

# move the model to available device
print('Moving the model to the available device ', ' ... '*3)
model.to(device)

# train model
print('\nStarting to train the model', ' ... '*6)
train_model(model, optimizer, criterion, train_loader, valid_loader, device, epochs, learning_rate)


# save checkpoint
print(f'\n\nSaving the trained model to a checkpoint labeled : {save_directory}', ' ... '*6)
save_checkpoint(model, image_datasets, optimizer, save_directory, arch, hidden_units, epochs)

# finish training and saving the model checkpoint
print('We have finished training and saving the model checkpoint - Bye!')
