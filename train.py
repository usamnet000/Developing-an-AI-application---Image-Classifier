# Imports from here
import argparse

import datetime
from time import time, sleep

from TrainModel import TrainModel
from Parameters import Parameters

def command(string):
    return str (input( string ))

def greetUser():
    currentH = int( datetime.datetime.now().hour )
    if currentH >= 0 and currentH < 12:
        print( 'Good Morning!' )

    if currentH >= 12 and currentH < 18:
        print( 'Good Afternoon!' )

    if currentH >= 18 and currentH != 0:
        print( 'Good Evening!' )

# This function retrieves 6 Command Line Arugments from user as input from
# the user running the program from a terminal window. This function returns
# the collection of these command line arguments from the function call as
# the variable in_arg
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type = str, default = 'vgg19', help = 'which CNN Model should be used for pretraining, choose between vgg13, vgg16, vgg19, densenet121, densenet161, alexnet | (default = vgg19)')
parser.add_argument('--save_directory', type = str, default = 'SavedModel/', help = 'directory to save trained model | (default = SavedModel/)')
parser.add_argument('--learningrate', type = float, default = 0.001, help = 'give learningrate as a float | (default = 0.001)')
parser.add_argument('--hidden_units', type = int, default = 508, help = 'give number of hidden units as an integer | (default = 508)')
parser.add_argument('--epochs', type = int, default = 1, help = 'give number of epochs as an integer | (default = 1)')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'cuda or cpu | (default = cuda)')
          
arg_in = parser.parse_args()
parameters = Parameters({"arch":arg_in.arch,"epochs":arg_in.epochs,"gpu":arg_in.gpu,"hidden_units":arg_in.hidden_units,"learningrate":arg_in.learningrate,"save_directory":arg_in.save_directory})

# Function that greet user
greetUser()
# Function that checks command line arguments using in_arg  
parameters.displayParameters('training')
query = command("Please Enter to continue")
print( 'Do you want to modify input values?' )
while True:
    print('‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡')
    print("## What you Want modify?                 ##")
    print("## 0 : modify a architecture             ##")
    print("## 1 : modify a learningrate             ##")
    print("## 2 : modify a hidden units             ##")
    print("## 3 : modify a epoch(s)                 ##")
    print("## 4 : modify a {} location of model save##".format(arg_in.save_directory))
    print("## 5 : modify a mode(s)                  ##")
    print("## 6 : Print  Parameters                 ##")
    print("## 7 : training                          ##")
    print("## 8 : exit                              ##")
    print('‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡')
    query = int(command("command: "))
    if query == 0 :
        query = command("Choice: vgg13, vgg16, vgg19, alexnet, densenet121, densenet161 , the defualt ({})".format(parameters.in_arg['arch']))
        while True:
            if query not in ["vgg13", "vgg16", "vgg19", "alexnet", "densenet121", "densenet161"]:
                print( "Try again! Please give a valid architecture. \nValid architectures: vgg13, vgg16, vgg19, alexnet, densenet121, densenet161")
                query = command("command: ")
            else:
                parameters.in_arg['arch'] = query
                break
    elif query == 1:
        while True:
            query = command("enter learningrate, the defualt ({})".format(parameters.in_arg['learningrate']))
            if query.replace('.', '', 1).isdigit():
                parameters.in_arg['learningrate'] = float(query)
                break
    elif query == 2:
        while True:
            query = command("enter hidden units, the defualt ({})".format(parameters.in_arg['hidden_units']))
            if query.isdigit():
                parameters.in_arg['hidden_units'] = int(query)
                break
    elif query == 3:
        while True:
            query = command("enter epochs, the defualt ({})".format(parameters.in_arg['epochs']))
            if query.isdigit():
                parameters.in_arg['epochs'] = int(query)
                break
    elif query == 4:
        query = command("enter save directory of model, the defualt ({})".format(parameters.in_arg['save_directory']))
        parameters.in_arg['save_directory'] = query
    elif query == 5:
        while True:
            query = command("enter cuda or cpu , the defualt ({})".format(parameters.in_arg['gpu']))
            if query not in ["cuda", "cpu"]:
                print("Try again! Please give a valid processor. \nValid processor: cuda, cpu")
                query = command("command: ")
            else:
                parameters.in_arg['gpu'] = query
                break
    elif query == 6:
        parameters.displayParameters('training')
    elif query == 7:
        modelObj = TrainModel()
        # prepare Dataset And Load
        trainloader, validloader, train_data = modelObj.prepareDatasetAndLoad()
        # Building network with Pytorch
        modelObj.define_model(parameters.in_arg['arch'])
        criterion, optimizer = modelObj.classifier(parameters.in_arg['hidden_units'], parameters.in_arg['learningrate'])
        # Training network to compare image input
        modelObj.training(parameters.in_arg['epochs'], parameters.in_arg['gpu'], trainloader, validloader, criterion, optimizer)
        # Save model file
        store = modelObj.save(parameters.in_arg['save_directory'], train_data, optimizer,parameters.in_arg['arch'])
    elif query == 8:
        break
    