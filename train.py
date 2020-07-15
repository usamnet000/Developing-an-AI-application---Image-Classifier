# Imports from here
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

def training_input():
    
    ''' 
    Input: None, automatically leading to filepaths
    Operation: Transforms training and validation data to input for model
    Output: traindataloader, validdataloader, train_image_datasets
    '''
    
    # Data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    
    # Transform pictures
    
    train_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.485, 0.456, 0.406])])
    
    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.485, 0.456, 0.406])])
    
    valid_data_transforms = test_data_transforms
    
    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=test_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    traindataloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    validdataloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
    testdataloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)   
    
    return traindataloader, validdataloader, train_image_datasets


def pretrained_model(architecture):
    
    ''' 
    Input: architecture
    Operation: Loads pretrained model and freezes its parameters
    Output: model
    '''
    possible_archs = {'vgg13': 25088, 'vgg16': 25088, 'vgg19': 25088, 'densenet121': 1024, 'densenet161': 2208, 'alexnet': 9216}
    
    if architecture in possible_archs:
        
        input_layer = possible_archs.get(architecture)
        
        # Remove string for further operations
        save_architecture = architecture
        architecture = architecture.replace("",'')
        model = getattr(models, architecture)(pretrained = True)
       
        for param in model.parameters():
            param.requires_grad = False
            
            
    else: 
        print( "Try again! Please give a valid architecture. \nValid architectures: vgg13, vgg16, vgg19, alexnet, densenet121, densenet161")
        return
    
    return model,  save_architecture, input_layer


def classifier(hiddenunits, learningrate, model, input_layer):
    
    ''' 
    Input: hidden_layers, learningrate, model
    Operation: Define the classifier network, setting optimizer and errorfunction
    Output: model.classifier, criterion, optimizer
    '''
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(4096, 512),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(512, 102),
                           nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    # define our criterion for loss
    criterion = nn.NLLLoss()
    # define our optimizer for only the created classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningrate)
    
    return model, criterion, optimizer



def training_network(epochs, device, model, trainloader, validloader, criterion, optimizer):
    '''
    Input: epochs, device, model, trainloader, validloader, criterion, optimizer
    Operation: Trains the classifier part of the CNN and validates the model every 20 steps
    Output: trained model with graphic
    '''
    import time
    #Make a validation step after every 20 iterations
    step_every = 20
    model.to(device) 

    # Start measuring time
    start = time.time()
    print("Start training ...")

    for epoch in range(epochs): 
        steps = 0
        running_loss = 0
        print("Epoch", epoch +1)
        
        # Training the classifier
        for images, labels in trainloader:
            steps += 1 
            model.train()
            images, labels = images.to(device), labels.to(device) 
        
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            #Validation testing
            if steps % step_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
                
                for imagesval, labelsval in validloader: 
                    
                    imagesval, labelsval = imagesval.to(device) , labelsval.to(device) 
                    
                    with torch.no_grad(): 
                        outputs = model.forward(imagesval)
                        test_loss = criterion(outputs,labelsval)
                        ps = torch.exp(outputs).data
                        equal = (labelsval.data == ps.max(1)[1])
                        accuracy += equal.type_as(torch.FloatTensor()).mean()

                # Calculating the training loss, validation loss and accuracy
                test_loss /= len(validloader)
                accuracy /= len(validloader)
                train_loss = running_loss/step_every
            
                running_loss = 0

                print("Epoch {}/{}  Training Loss: {:.3f}  Validation Loss: {:.3f}  Accuracy: {:.2f}%".format((epoch + 1), 
                epochs, train_loss, test_loss, accuracy * 100))

    
    end = time.time() 
    time = end - start
    print("Total training time : {}m {}s".format(int(time // 60), int(time % 60)))
    
    return model
   
   
def saving_model(filepath, train_image_datasets, learningrate, epochs, model, optimizer):
    
    '''
    Input : filepath, train_data, learningrate, epochs, model
    Operation : Saves trained model to checkpoint.pth
    Output : Statement where file was saved
    '''
    model.class_to_idx = train_image_datasets.class_to_idx

    checkpoint = {'input size': 25088,
                  'output size': 102,
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'classifier': model.classifier,
                  'learningrate': learningrate,
                  'optimizer': optimizer.state_dict(),
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    # Saving the model and its hyperparameters to the filepath
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    os.path.join(filepath, 'checkpoint.pth')
    
    store = "{}checkpoint.pth".format(filepath)
    torch.save(checkpoint, store)
    
    print("saved model checkpoint under ", store)
    
    return store

 
parser = argparse.ArgumentParser()

parser.add_argument('--arch', type = str, default = 'vgg19', help = 'which CNN Model should be used for pretraining, choose between vgg13, vgg16, vgg19, densenet121, densenet161, alexnet | (default = vgg19)')
parser.add_argument('--save_directory', type = str, default = 'SavedModel/', help = 'directory to save trained model | (default = SavedModel/)')
parser.add_argument('--learningrate', type = float, default = 0.001, help = 'give learningrate as a float | (default = 0.001)')
parser.add_argument('--hidden_units', type = int, default = 508, help = 'give number of hidden units as an integer | (default = 508)')
parser.add_argument('--epochs', type = int, default = 1, help = 'give number of epochs as an integer | (default = 1)')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'cuda or cpu | (default = cuda)')

args = parser.parse_args()

# Run functions 

print(" Selection: Training a CNN \n Using a pretrained {} architecture \n With hyperparameters: learningrate {}, {} hidden units and {} epoch(s) \n The trained model is stored under {}checkpoint.pth \n The training is performed in {} mode".format(args.arch,args.learningrate, args.hidden_units, args.epochs, args.save_directory, args.gpu))

trainloader, validloader, train_data = training_input()

model,  save_architecture, input_layer = pretrained_model(args.arch)

model, criterion, optimizer = classifier(args.hidden_units, args.learningrate, model, input_layer)

model = training_network(args.epochs, args.gpu, model, trainloader, validloader, criterion, optimizer)

store = saving_model(args.save_directory, train_data, args.learningrate, args.epochs, model, optimizer)


# Save architecture model and filepath

os.path.join(args.save_directory, 'save_progress.txt')
with open("save_progress.txt", "w") as output:
    output.write(str(store) + "\n" + str(args.arch))