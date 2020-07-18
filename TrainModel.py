import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import numpy as np
from PIL import Image

class TrainModel:
    def __init__(self):
        self.model = None
        
    def prepareDatasetAndLoad(self):
        
        # location of dataset
        data_dir = 'flowers'    
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        # define a transform to normalize the data 
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
        # # after download datasets is load dataset
        traindataloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True) # get 64 image and label for traindataloader
        validdataloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64)
        testdataloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64)   
        
        return traindataloader, validdataloader, train_image_datasets

    def define_model(self, model_name):
        models_ = {'vgg13': 25088, 'vgg16': 25088, 'vgg19': 25088, 'densenet121': 1024, 'densenet161': 2208, 'alexnet': 9216}
        if model_name in models_:
            # apply model to input
            #self.model = models_[model_name]
            self.model = getattr(models, model_name.replace("",''))(pretrained = True)
           
            for param in self.model.parameters():
                param.requires_grad = False
        
        else: 
            print( "Try again! Please give a valid model.")
            

    def classifier(self,hiddenunits, learningrate):
        classifier = nn.Sequential(nn.Linear(25088, 4096),# input layer in network
                           nn.ReLU(), # first layer is using relu activation function
                           nn.Dropout(0.5), # again, dropout value received from model above
                           nn.Linear(4096, 512),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(512, 102),# output thirdlayer is softmax 
                           nn.LogSoftmax(dim=1))# batch size dim 1 by column and actual vector passing through is 
        
        self.model.classifier = classifier             
        
        # We can get a loss by looking at a criterion
        # we try minimize loss to get better result in output 
        criterion = nn.NLLLoss()
        # define our optimizer for only the created classifier
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=learningrate) # Find out how to use these gradients to really refresh our weights
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        return criterion, optimizer

    # Measuring performance in the test dataset The goal of validation is to measure the performance of our model
    def validation(self,running_loss,epoch,epochs,steps,validloader,device,criterion):
        #Make a validation step after every 20 iterations
        step_every = 20
        if steps % step_every == 0:
            test_loss = 0
            valid_accuracy  = 0

            # puts model in evaluation mode
            # instead of (default)training mode
            # When we try to make forecasts with our network, we want all of our units to be available so we want to stop the leak when we check validation
            self.model.eval()
            
            for imagesval, labelsval in validloader: 
                imagesval, labelsval = imagesval.to(device) , labelsval.to(device)  
                
                # Turn off gradients to speed up this part
                # As for validation, we will not conduct any training, so we will not need to be graduated
                with torch.no_grad(): 
                    outputs = self.model.forward(imagesval) # now to see the image output to predicted in Validation 
                    test_loss = criterion(outputs,labelsval)
                    ps = torch.exp(outputs).data # Take the exponent to get the actual possibilities
                    # Validation by compare labels and highest probability of correct prediction of a class given the image
                    equals = (labelsval.data == ps.max(1)[1]) # you can use topk instead max 
                    # convert to FloatTensor to work and Calculate mean
                    valid_accuracy += equals.type_as(torch.FloatTensor()).mean() # Accuracy is the number of correct classifications that we made during the previous step(ones) to make our model compare to all expectations
                    
            # Calculating the training loss, validation loss and accuracy
            test_loss /= len(validloader)
            valid_accuracy /= len(validloader)
            train_loss = running_loss/step_every # show in result loss start decrease
            running_loss = 0

            # print our results so far
            print("Epoch {}/{}  Training Loss: {:.3f}  Validation Loss: {:.3f}  Valid Accuracy : {:.2f}%".format((epoch + 1), 
                    epochs, train_loss, test_loss, valid_accuracy  * 100))
        
    def training(self,epochs, device, trainloader, validloader, criterion, optimizer):
        # Measures total program runtime by collecting start time
        from time import time, sleep
        start_time = time()

        print("training ...")
        
        #Each pass through a whole training dataset is called epoch
        for epoch in range(epochs): 
            steps = 0
            running_loss = 0 # after calculte loss we can track loss and logged
            print("Epoch", epoch +1)
            
            # Training the classifier
            for images, labels in trainloader:  # to get images and labels in trainloader
                steps += 1 
                self.model.train()
                images, labels = images.to(device), labels.to(device) 
                
                # The training permit has four different steps
                # Forward pass ,and use outputs to get loss, then backward pass , then update weights
                optimizer.zero_grad() # Clear the gradients, do this because gradients are accumulated and Necessary for the network to function properly
                
                outputs = self.model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward() # generate weight grad values to use it in train network 
                optimizer.step() # if pass optimizer step the weights is change 
                
                running_loss += loss.item() # get loss
                self.validation(running_loss,epoch,epochs,steps,validloader,device,criterion)
                
        # Measure total program runtime by collecting end time
        end_time = time()
        
        # Computes overall runtime in seconds & prints it in hh:mm:ss format
        tot_time = end_time - start_time #calculate difference between end time and start time
        
        print("\n** Total Elapsed Runtime:",
        str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
        +str(int((tot_time%3600)%60)) ) 

    def save(self,filepath, train_image_datasets, optimizer,model_name):
        self.model.class_to_idx = train_image_datasets.class_to_idx     
        checkpoint = {'input size': 25088,
                  'output size': 102,
                  'state_dict': self.model.state_dict(),
                  'arch': model_name,
                  'classifier': self.model.classifier,
                  'model_state_dict': self.model.state_dict(),
                  'class_to_idx': self.model.class_to_idx}
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        os.path.join(filepath, 'checkpoint.pth')
        
        # Saving the model
        store = "{}checkpoint.pth".format(filepath)
        torch.save(checkpoint, store)
        print("saved model checkpoint ", store)
        
        return store
