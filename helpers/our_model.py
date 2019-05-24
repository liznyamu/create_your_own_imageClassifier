# Imports here
import torch
from torch import nn, optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models

# non- PyTorch imports
from collections import OrderedDict
import time

# own modules
from helpers.images import process_image

# Keep the session active
from workspace_utils import active_session



# TODO : load the data
def load_data(data_dir = 'flowers'):
    '''
        Return dataloaders dictionary, 'train', 'valid' and 'test' data loaders
    '''
    # Image directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {'train' : train_data,
                     'valid': valid_data,
                     'test': test_data}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)

    dataloaders = {'train' : train_loader,
                  'valid' : valid_loader,
                  'test' : test_loader}

    return dataloaders, image_datasets, train_loader, valid_loader, test_loader

# TODO : load the data
def load_testdata(data_dir = 'flowers'):
    '''
        Return dataloaders dictionary, 'train', 'valid' and 'test' data loaders
    '''
    # Image directories
    test_dir = data_dir + '/test'

    # Define your transforms for the testing sets
    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return test_loader

# TODO get pretrained network model
def load_arch(arch):
    '''
        Load a pre-trained network, freeze its parameters

        return pretrained model
    '''
    # use Python getattr() - http://bit.ly/2HzYG3X
    # Load a pre-trained network - VGG-19
    model = getattr(models, arch)(pretrained=True) #model = models.vgg19(pretrained=True)

    # Freeze the pre-trained model parameters so we don't backpropagate through them
    for param in model.parameters():
        param.require_grad = False

    return model


# TODO : define a method to add our classifier
def set_classifier(model, hidden_units = 1024):
    '''
        Define a new, untrained feed-forward network as a classifier,
        using ReLU activations and dropout

        return model with our classifier
    '''
    # dropout rate
    dropout = 0.2
    # model's classifier [25088, 1024, 102]
    classifier = nn.Sequential(OrderedDict([('fc_1', nn.Linear(25088,hidden_units)),
                                           ('relu', nn.ReLU()),
                                           ('drop', nn.Dropout(p=dropout)),
                                            ('fc_2', nn.Linear(hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    # set classifier on pre-trained network
    model.classifier = classifier

    return model, classifier



# Train the model - on the 'train' images and make validations using 'valid' images
# TODO : add time output - from start to stop
def train_model(model, optimizer, criterion, train_loader, valid_loader, device, epochs = 10, learning_rate = 0.003):

    # start timer
    startTimer = time.time()

    with active_session():
    # do long-running work here

        # running loss on training our model
        running_loss = 0

        # track training loss, validation/inference loss and model's accuracy
        # within the training batches
        steps, print_every = 0, 80

        for epoch in range(epochs):
            # use the train_loader to train the model
            for inputs, labels in train_loader:
                steps += 1

                # Move input and label tensors to the default/available device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)

                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # view training loss, validation loss and model's accuracy
                if steps % print_every == 0:
                    # reset the testing metrics
                    test_loss = 0
                    accuracy = 0

                    # set model in evaluation mode
                    model.eval()

                    # turn off gradient descent on validation
                    with torch.no_grad():
                        # use the valid_loader to validate the model
                        for inputs, labels in valid_loader:

                            # Move input and label tensors to the default/available device
                            inputs, labels = inputs.to(device), labels.to(device)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)

                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # print our metrics
                    print(f'Epoch {epoch+1}/{epochs}..',
                         f'Train loss: {running_loss/print_every:.3f}..',
                         f'Validation loss: {test_loss/len(valid_loader):.3f}',
                         f'Validation accuracy: {accuracy/len(valid_loader):.3f}')

                    #reset metrics
                    running_loss = 0

                    # set model to training mode
                    model.train()

    # print how long it took to train the model :
    endTimer = time.time()
    time_elapsed = endTimer - startTimer
    print("\nTotal Training Period: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))


# TODO : Test the models accuracy
def test_model(model, test_loader, device):
    # TODO: Do validation on the test set
    accuracy = 0

    # set model in evaluation mode
    model.eval()

    # turn off gradient descent on testing phase
    with torch.no_grad():

        for inputs, labels in test_loader:

            # Move input and label tensors to the default/available device
            inputs, labels = inputs.to(device), labels.to(device)

            # get the log probabilities
            logps = model.forward(inputs)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f'Test accuracy: {accuracy/len(test_loader):.3f}')


#TODO : Save checkpoints
def save_checkpoint(model, image_datasets, optimizer , checkpoint_path='checkpoint.pth', arch ='vgg19', hidden_units = 1024, epochs = 10):
    # include below on the checkpoint :
    # trained model's mapping of flower classes to predicted indices
    model.class_to_idx = image_datasets['train'].class_to_idx

    # number of epochs, optimizer state_dict, model's state_dict, input-hidden-output sizes
    # TODO : we can remove the model_state_dict as it was not required (as per instructions)
    # TODO : see if we can load the classifier too
    # TODO : remove input-hidden-output sizes if classifier works
    # TODO : add the pretrained network used ----------------------------------------------------***
    checkpoint = {'epoch' : epochs,
                  'arch' : arch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'dropout' : 0.2,
                  'input_size' : 25088,
                  'hidden_layer' : hidden_units,
                  'output_size' : 102
                 }

    # Save the checkpoint - with reference to - http://bit.ly/2VxhJ2K
    torch.save(checkpoint, checkpoint_path)

# TODO: Implement the code to predict the class from an image file
def predict(image_path, model, topk=5):
    '''
        Predict the class (or topk classes) of an image
        using a trained deep learning model.
    '''

    # get normalized image (numpy ndarray)
    img = process_image(image_path)

    # convert numpy ndarray to pytorch tensor - http://bit.ly/2WhCLqp
    #  call .float() on your inputs, doubles will run slower and use more memory - http://bit.ly/2YFg397
    img = torch.from_numpy(img).float()

    # add a dimension of batch size at dimension 1 (ie index 0) - http://bit.ly/2Wibi88
    img.unsqueeze_(0)

    # get probabilities - on normalized image
    log_ps = model.forward(img)
    ps = torch.exp(log_ps)

    # get topk probabilities and indices - across multiple columns
    top_k, top_idx = ps.topk(topk, dim=1)

    # convert pytorch tensors to list - and get first element (ie index 0) - http://bit.ly/2YIC2w8
    top_k = top_k.tolist()[0]
    top_idx = top_idx.tolist()[0]

    # invert model.class_to_idx dictionary - http://bit.ly/2M3RIZf
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}

    # get topk classes from idx_to_class - http://bit.ly/2WbpdwL
    top_class = [idx_to_class.get(x) for x in top_idx]

    # round the probabilities to 8 decimal places - http://bit.ly/2HudlgW
    top_k = [round(x, 8) for x in top_k]

    return top_k, top_class


# TODO: Implement the code to predict the class from an image file
def predict_device(image_path, model, gpu, topk=5):
    '''
        Predict the class (or topk classes) of an image
        using a trained deep learning model.
    '''

    if gpu and torch.cuda.is_available():
        print('\nUsing GPU - to make inferences or predictions')
        device = torch.device("cuda")
    else:
        print('\nUsing CPU - to make inferences or predictions')
        device = torch.device("cpu")

    # TODO: set model to available device
    model.to(device)
    
    # get normalized image (numpy ndarray)
    img = process_image(image_path)

    # convert numpy ndarray to pytorch tensor - http://bit.ly/2WhCLqp
    #  call .float() on your inputs, doubles will run slower and use more memory - http://bit.ly/2YFg397
    img = torch.from_numpy(img).float()

    # TODO : set model to avaible device
    img = img.to(device)

    # add a dimension of batch size at dimension 1 (ie index 0) - http://bit.ly/2Wibi88
    img.unsqueeze_(0)

    # get probabilities - on normalized image
    log_ps = model.forward(img)
    ps = torch.exp(log_ps)

    # get topk probabilities and indices - across multiple columns
    top_k, top_idx = ps.topk(topk, dim=1)

    # convert pytorch tensors to list - and get first element (ie index 0) - http://bit.ly/2YIC2w8
    top_k = top_k.tolist()[0]
    top_idx = top_idx.tolist()[0]

    # invert model.class_to_idx dictionary - http://bit.ly/2M3RIZf
    idx_to_class = {value:key for key, value in model.class_to_idx.items()}

    # get topk classes from idx_to_class - http://bit.ly/2WbpdwL
    top_class = [idx_to_class.get(x) for x in top_idx]

    # round the probabilities to 8 decimal places - http://bit.ly/2HudlgW
    top_k = [round(x, 8) for x in top_k]

    return top_k, top_class



# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(file_path):
    '''
    Loads a checkpoint and rebuild the trained model
    based on a pre-trained network

        returns the trained model and checkpoint
    '''
    # load the saved checkpoint
    # we want to load the model on CPU after it was trained on GPU
    #device = torch.device('cpu')
    checkpoint = torch.load(file_path, map_location = 'cpu')

    # use Python getattr() - http://bit.ly/2HzYG3X
    # load the pre-trained model used during training
    model = getattr(models, checkpoint['arch'])(pretrained=True) #model = models.vgg19(pretrained=True)

    # freeze the pre-trained network parameters
    for param in model.parameters():
        param.require_grad = False

    # rebuild classifier
    classifier = nn.Sequential(OrderedDict([('fc_1', nn.Linear(checkpoint['input_size'],\
                                                               checkpoint['hidden_layer'])),
                                       ('relu', nn.ReLU()),
                                       ('drop', nn.Dropout(p=checkpoint['dropout'])),
                                        ('fc_2', nn.Linear(checkpoint['hidden_layer'],\
                                                           checkpoint['output_size'])),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ]))

    # set model classifier
    model.classifier = classifier

    # set model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    # map flower classes to predicted indices
    model.class_to_idx = checkpoint['class_to_idx']

    # return loaded checkpoint
    return model, checkpoint
